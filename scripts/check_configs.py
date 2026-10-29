"""Check the stage configs before a GPU run. Pod-only checks are warnings unless --pod."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
import types
import typing
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from configure_judge import judge_env_problems
from layout import BASE_BF16_DIR, REPO_ROOT, STAGE1_EXPORT_DIR, STAGES
from pydantic import BaseModel

NUM_ATTENTION_HEADS = 64
NUM_KV_HEADS = 8
ALLOWED_TP = (1, 2, 4, 8)
DEFAULT_SEARCH_DIMS = 9
DEFAULT_MIN_WARMUP_PER_DIM = 4.4
ALLOWED_ADDED_KEYS = frozenset({"optimization.sampler_seed"})
PROMPT_SECTIONS = ("benign_prompts", "target_prompts", "benign_eval_prompts", "target_eval_prompts")
EXPECTED_MODEL_ID = {"stage1": BASE_BF16_DIR, "stage2": STAGE1_EXPORT_DIR}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def flatten_keys(table: dict[str, Any], prefix: str = "") -> Iterator[str]:
    # max_memory's keys depend on the GPU count, so treat it as a single key.
    for key, value in table.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and dotted != "model.max_memory":
            yield from flatten_keys(value, dotted + ".")
        else:
            yield dotted


def check_tensor_parallel(stage: str, model: dict[str, Any], report: Report) -> None:
    tp = model.get("tensor_parallel_size")
    if not isinstance(tp, int) or isinstance(tp, bool):
        report.error(f"{stage}: model.tensor_parallel_size must be an integer, got {tp!r}")
        return
    if tp == 3:
        report.error(f"{stage}: tensor_parallel_size=3 is invalid (64 heads / 8 KV heads)")
        return
    if tp not in ALLOWED_TP or NUM_ATTENTION_HEADS % tp or NUM_KV_HEADS % tp:
        report.error(
            f"{stage}: tensor_parallel_size={tp} must be one of {ALLOWED_TP} and divide "
            f"{NUM_ATTENTION_HEADS} attention heads and {NUM_KV_HEADS} KV heads"
        )
        return
    max_memory = model.get("max_memory")
    if not isinstance(max_memory, dict):
        report.error(f"{stage}: model.max_memory must list one entry per GPU")
        return
    gpu_keys = sorted(str(k) for k in max_memory if str(k) != "cpu")
    expected = [str(i) for i in range(tp)]
    if gpu_keys != expected:
        report.error(
            f"{stage}: max_memory GPU keys {gpu_keys} do not match tensor_parallel_size={tp} "
            f"(expected {expected})"
        )


def check_trials(
    stage: str, optimization: dict[str, Any], search_dims: int, min_per_dim: float, report: Report
) -> None:
    trials = optimization.get("num_trials")
    warmup = optimization.get("num_warmup_trials")
    if not isinstance(trials, int) or not isinstance(warmup, int):
        report.error(f"{stage}: num_trials and num_warmup_trials must be integers")
        return
    if not 0 <= warmup < trials:
        report.error(f"{stage}: num_warmup_trials ({warmup}) must be < num_trials ({trials})")
        return
    per_dim = warmup / search_dims
    if per_dim < min_per_dim:
        report.warn(
            f"{stage}: {warmup} warmup trials / {search_dims} search dims = {per_dim:.2f} "
            f"per dim, below {min_per_dim}"
        )
    if optimization.get("sampler_seed") != 42:
        report.error(f"{stage}: optimization.sampler_seed must be 42")


def check_key_subset(stage: str, cfg: dict[str, Any], upstream: dict[str, Any], report: Report):
    added = set(flatten_keys(cfg)) - set(flatten_keys(upstream)) - ALLOWED_ADDED_KEYS
    for key in sorted(added):
        report.error(f"{stage}: key {key!r} does not exist in the upstream config")


def _model_class(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if (found := _model_class(arg)) is not None:
                return found
    return None


def unknown_schema_keys(table: dict[str, Any], model: type[BaseModel], prefix: str = ""):
    # abliterix silently ignores unknown keys inside sections, so find them here.
    for key, value in table.items():
        info = model.model_fields.get(key)
        if info is None:
            yield f"{prefix}{key}"
            continue
        sub = _model_class(info.annotation)
        if sub is not None and isinstance(value, dict):
            yield from unknown_schema_keys(value, sub, f"{prefix}{key}.")


def abliterix_config(cfg: dict[str, Any]) -> Any:
    # Validate `cfg` only, ignoring command-line arguments and AX_* environment variables.
    from abliterix.settings import AbliterixConfig

    class FileOnlyConfig(AbliterixConfig):
        @classmethod
        def settings_customise_sources(cls, settings_cls, init_settings, **sources):
            return (init_settings,)

    return FileOnlyConfig(**cfg)


def check_abliterix_schema(stage: str, cfg: dict[str, Any], report: Report) -> None:
    from abliterix.settings import AbliterixConfig
    from pydantic import ValidationError

    for key in unknown_schema_keys(cfg, AbliterixConfig):
        report.error(f"{stage}: key {key!r} is not an abliterix config field")
    try:
        abliterix_config(cfg)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            report.error(f"{stage}: abliterix rejects config at {loc}: {err['msg']}")


def check_environment(stage: str, cfg: dict[str, Any], pod: bool, report: Report) -> None:
    flag = report.error if pod else report.warn
    if cfg.get("detection", {}).get("llm_judge"):
        for problem in judge_env_problems(os.environ):
            flag(f"{stage}: LLM judge: {problem} (run scripts/configure_judge.py)")
    datasets = {cfg.get(section, {}).get("dataset") for section in PROMPT_SECTIONS}
    for dataset in sorted(datasets, key=str):
        if not dataset or not (REPO_ROOT / dataset).is_dir():
            flag(f"{stage}: dataset {dataset!r} not found (run scripts/setup_data.py)")
    model_id = cfg.get("model", {}).get("model_id", "")
    if not (REPO_ROOT / model_id).is_dir():
        flag(f"{stage}: model directory {model_id!r} does not exist yet")


def check_stage(
    stage: str,
    cfg: dict[str, Any],
    upstream: dict[str, Any],
    *,
    search_dims: int = DEFAULT_SEARCH_DIMS,
    min_warmup_per_dim: float = DEFAULT_MIN_WARMUP_PER_DIM,
    pod: bool = False,
) -> Report:
    report = Report()
    check_tensor_parallel(stage, cfg.get("model", {}), report)
    check_trials(stage, cfg.get("optimization", {}), search_dims, min_warmup_per_dim, report)
    check_key_subset(stage, cfg, upstream, report)
    check_abliterix_schema(stage, cfg, report)
    model_id = cfg.get("model", {}).get("model_id")
    if model_id != EXPECTED_MODEL_ID[stage]:
        report.error(
            f"{stage}: model.model_id is {model_id!r}, expected {EXPECTED_MODEL_ID[stage]!r}"
        )
    check_environment(stage, cfg, pod, report)
    return report


def check_all(
    configs: dict[str, dict[str, Any]],
    upstreams: dict[str, dict[str, Any]],
    **kwargs: Any,
) -> Report:
    report = Report()
    for stage in STAGES:
        sub = check_stage(stage, configs[stage], upstreams[stage], **kwargs)
        report.errors += sub.errors
        report.warnings += sub.warnings
    dirs = {configs[s].get("optimization", {}).get("checkpoint_dir") for s in STAGES}
    if len(dirs) != len(STAGES):
        report.error("stage1 and stage2 must use different optimization.checkpoint_dir values")
    tps = {configs[s].get("model", {}).get("tensor_parallel_size") for s in STAGES}
    if len(tps) != 1:
        report.error("stage1 and stage2 must use the same tensor_parallel_size (same pod)")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", action="store_true", help="treat pod-only checks as errors")
    parser.add_argument("--search-dims", type=int, default=DEFAULT_SEARCH_DIMS)
    parser.add_argument("--min-warmup-per-dim", type=float, default=DEFAULT_MIN_WARMUP_PER_DIM)
    args = parser.parse_args(argv)

    configs = {stage: load_toml(paths[0]) for stage, paths in STAGES.items()}
    upstreams = {stage: load_toml(paths[1]) for stage, paths in STAGES.items()}
    report = check_all(
        configs,
        upstreams,
        search_dims=args.search_dims,
        min_warmup_per_dim=args.min_warmup_per_dim,
        pod=args.pod,
    )
    for msg in report.warnings:
        print(f"WARNING: {msg}")
    for msg in report.errors:
        print(f"ERROR: {msg}")
    print(f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
