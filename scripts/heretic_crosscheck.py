"""Optional cross-check: run Heretic's evaluate mode and record its KL and refusal counts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from layout import RUNS_DIR

# The log format was checked against 1.4.0 only.
VERIFIED_HERETIC_VERSION = "1.4.0"
DETECTOR = "substring refusal_markers (Heretic config.default.toml); empty response = refusal"
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
PATTERNS = {
    "heretic_version": re.compile(r"█.*\bv(\d+\.\d+\.\d+\S*)"),
    "good_dataset": re.compile(r"Loading good evaluation prompts from (.+?)\.\.\.$", re.M),
    "bad_dataset": re.compile(r"Loading bad evaluation prompts from (.+?)\.\.\.$", re.M),
    "base_refusals": re.compile(r"\* Initial refusals: (\d+)/(\d+)"),
    "kl_divergence": re.compile(r"\* KL divergence: ([0-9.eE+-]+)"),
    "refusals": re.compile(r"^\s*\* Refusals: (\d+)/(\d+)", re.M),
}


class HereticOutputError(RuntimeError):
    pass


def build_command(heretic_bin: str, base: str, modified: str) -> list[str]:
    return [heretic_bin, "--model", base, "--evaluate-model", modified]


def resolve_model_arg(value: str) -> str:
    # Heretic runs inside the output directory, so local paths must be absolute.
    path = Path(value)
    return str(path.resolve()) if path.exists() else value


def _single(name: str, text: str) -> re.Match[str]:
    matches = list(PATTERNS[name].finditer(text))
    if len(matches) != 1:
        raise HereticOutputError(f"expected exactly one '{name}' line, found {len(matches)}")
    return matches[0]


def parse_output(log: str) -> dict:
    text = ANSI.sub("", log)
    version = _single("heretic_version", text).group(1)
    if version != VERIFIED_HERETIC_VERSION:
        raise HereticOutputError(
            f"Heretic {version} output format is unverified (verified: {VERIFIED_HERETIC_VERSION})"
        )
    base_k, base_n = map(int, _single("base_refusals", text).groups())
    k, n = map(int, _single("refusals", text).groups())
    if n != base_n:
        raise HereticOutputError(f"prompt counts differ: base {base_n}, modified {n}")
    return {
        "heretic_version": version,
        "kl_divergence": float(_single("kl_divergence", text).group(1)),
        "refusals": k,
        "total_prompts": n,
        "base_refusals": base_k,
        "detector": DETECTOR,
        "datasets": {
            "good_evaluation_prompts": _single("good_dataset", text).group(1),
            "bad_evaluation_prompts": _single("bad_dataset", text).group(1),
        },
    }


def run_heretic(cmd: list[str], out_dir: Path, log_path: Path) -> int:
    env = dict(
        os.environ, NO_COLOR="1", COLUMNS="1000", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"
    )
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=out_dir,  # Heretic reads config.toml from its working directory
            env=env,
            stdin=subprocess.DEVNULL,  # fail instead of hanging on an interactive question
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            sys.stdout.write(line)
        return proc.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="openai/gpt-oss-120b")
    parser.add_argument("--modified", default="out/stage2")
    parser.add_argument("--heretic-bin", help="path to the isolated heretic executable")
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR / "heretic_crosscheck")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = parser.parse_args(argv)

    heretic_bin = args.heretic_bin or shutil.which("heretic")
    base, modified = resolve_model_arg(args.base), resolve_model_arg(args.modified)
    cmd = build_command(heretic_bin or "heretic", base, modified)
    if args.dry_run:
        print(subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd))
        return 0
    if heretic_bin is None:
        print(
            "SKIPPED: Heretic is not installed (optional step; see README 'Heretic cross-check')."
        )
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "heretic.log"
    code = run_heretic(cmd, args.out_dir, log_path)
    if code != 0:
        print(f"Heretic exited with code {code}; raw log: {log_path}", file=sys.stderr)
        return code
    parsed = parse_output(log_path.read_text(encoding="utf-8"))
    summary = {"command": cmd, "base_model": args.base, "modified_model": args.modified, **parsed}
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
