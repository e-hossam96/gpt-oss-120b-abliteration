import copy

import pytest
from check_configs import (
    Report,
    check_all,
    check_key_subset,
    check_stage,
    check_tensor_parallel,
    check_trials,
    flatten_keys,
    load_toml,
)
from layout import STAGES


@pytest.fixture(scope="module")
def configs():
    return {stage: load_toml(paths[0]) for stage, paths in STAGES.items()}


@pytest.fixture(scope="module")
def upstreams():
    return {stage: load_toml(paths[1]) for stage, paths in STAGES.items()}


def test_repo_configs_have_no_errors(configs, upstreams):
    report = check_all(configs, upstreams)
    assert report.errors == []


@pytest.mark.parametrize("tp", [3, 5, 6, 16, 0])
def test_invalid_tensor_parallel_rejected(tp):
    report = Report()
    memory = {str(i): "80GiB" for i in range(max(tp, 1))}
    check_tensor_parallel("s", {"tensor_parallel_size": tp, "max_memory": memory}, report)
    assert report.errors


def test_tp3_has_explicit_message():
    report = Report()
    check_tensor_parallel("s", {"tensor_parallel_size": 3, "max_memory": {}}, report)
    assert "tensor_parallel_size=3 is invalid" in report.errors[0]


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_valid_tensor_parallel_accepted(tp):
    report = Report()
    memory = {i: "80GiB" for i in range(tp)}
    check_tensor_parallel("s", {"tensor_parallel_size": tp, "max_memory": memory}, report)
    assert report.errors == []


def test_max_memory_must_match_tp():
    report = Report()
    check_tensor_parallel(
        "s",
        {"tensor_parallel_size": 2, "max_memory": {"0": "1GiB", "1": "1GiB", "2": "1GiB"}},
        report,
    )
    assert "do not match" in report.errors[0]


def test_warmup_must_be_below_trials():
    report = Report()
    check_trials(
        "s", {"num_trials": 20, "num_warmup_trials": 20, "sampler_seed": 42}, 9, 4.4, report
    )
    assert report.errors


def test_low_warmup_per_dim_warns_but_passes():
    report = Report()
    check_trials(
        "s", {"num_trials": 50, "num_warmup_trials": 20, "sampler_seed": 42}, 9, 4.4, report
    )
    assert report.errors == []
    assert "2.22 per dim" in report.warnings[0]


def test_warmup_threshold_is_configurable():
    report = Report()
    check_trials(
        "s", {"num_trials": 50, "num_warmup_trials": 20, "sampler_seed": 42}, 4, 4.4, report
    )
    assert report.warnings == []


def test_key_subset_invariant_holds_for_repo_configs(configs, upstreams):
    for stage in STAGES:
        added = set(flatten_keys(configs[stage])) - set(flatten_keys(upstreams[stage]))
        assert added == {"optimization.sampler_seed"}


def test_unknown_key_rejected(configs, upstreams):
    cfg = copy.deepcopy(configs["stage1"])
    cfg["optimization"]["num_trails"] = 10
    report = Report()
    check_key_subset("stage1", cfg, upstreams["stage1"], report)
    assert "optimization.num_trails" in report.errors[0]


def test_abliterix_schema_catches_silently_ignored_key(configs, upstreams):
    cfg = copy.deepcopy(configs["stage1"])
    upstream = copy.deepcopy(upstreams["stage1"])
    cfg["steering"]["not_a_field"] = 1
    upstream["steering"]["not_a_field"] = 1  # isolate the schema check from the subset check
    report = check_stage("stage1", cfg, upstream)
    assert any("not an abliterix config field" in e for e in report.errors)


def test_upstream_v7_config_is_rejected_by_abliterix(upstreams):
    cfg = copy.deepcopy(upstreams["stage1"])
    cfg["model"]["model_id"] = "out/gpt-oss-120b-bf16"
    cfg["optimization"]["sampler_seed"] = 42
    report = check_stage("stage1", cfg, upstreams["stage1"])
    assert any("weight_normalization='full'" in e for e in report.errors)


def test_stage2_must_point_at_stage1_export(configs, upstreams):
    cfg = copy.deepcopy(configs["stage2"])
    cfg["model"]["model_id"] = "out/elsewhere"
    report = check_stage("stage2", cfg, upstreams["stage2"])
    assert any("expected 'out/stage1'" in e for e in report.errors)


def test_judge_variables_required_on_pod(configs, upstreams, monkeypatch):
    for var in ("LLM_JUDGE_MODEL", "LLM_JUDGE_BASE_URL", "LLM_JUDGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    report = check_stage("stage1", configs["stage1"], upstreams["stage1"], pod=True)
    assert sum("LLM judge" in e for e in report.errors) == 3
    local = check_stage("stage1", configs["stage1"], upstreams["stage1"], pod=False)
    assert not any("LLM judge" in e for e in local.errors)


def test_judge_variables_accepted_when_set(configs, upstreams, monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_MODEL", "judge/x")
    monkeypatch.setenv("LLM_JUDGE_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "key")
    report = check_stage("stage1", configs["stage1"], upstreams["stage1"], pod=True)
    assert not any("LLM judge" in e for e in report.errors)
