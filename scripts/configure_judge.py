"""Ask for the LLM judge (name, base URL, API key) and store it in a gitignored env file.

The judge is any OpenAI-compatible /chat/completions endpoint. abliterix reads the
model and base URL from CLI flags and the key from LLM_JUDGE_API_KEY (its default
env var when a base URL is set); evaluate.py reads all three from the environment.
Load the file before running anything with `set -a; . ./.env.judge; set +a`.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlparse

from layout import REPO_ROOT

JUDGE_ENV_FILE = REPO_ROOT / ".env.judge"
MODEL_VAR = "LLM_JUDGE_MODEL"
BASE_URL_VAR = "LLM_JUDGE_BASE_URL"
API_KEY_VAR = "LLM_JUDGE_API_KEY"
JUDGE_VARS = (MODEL_VAR, BASE_URL_VAR, API_KEY_VAR)


def base_url_problem(url: str) -> str | None:
    """Why `url` is not a usable OpenAI-compatible base URL, or None if it is."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return f"{BASE_URL_VAR} must be an http(s) URL such as https://host/v1"
    if url.rstrip("/").endswith("/chat/completions"):
        return f"{BASE_URL_VAR} must be the base URL; /chat/completions is appended for you"
    return None


def judge_env_problems(environ: Mapping[str, str]) -> list[str]:
    problems = [f"{var} is not set" for var in JUDGE_VARS if not environ.get(var, "").strip()]
    url = environ.get(BASE_URL_VAR, "").strip()
    if url and (problem := base_url_problem(url)):
        problems.append(problem)
    return problems


def ask_judge(ask: Callable[[str], str], ask_secret: Callable[[str], str]) -> dict[str, str]:
    values = {
        MODEL_VAR: ask("LLM name (model id the endpoint expects): ").strip(),
        BASE_URL_VAR: ask("LLM base URL (OpenAI-compatible, e.g. https://host/v1): ").strip(),
        API_KEY_VAR: "".join(ask_secret("LLM API key (input hidden): ").split()),
    }
    problems = judge_env_problems(values)
    if problems:
        raise ValueError("; ".join(problems))
    return values


def write_env_file(path: Path, values: Mapping[str, str]) -> None:
    lines = [f"{var}={shlex.quote(values[var])}" for var in JUDGE_VARS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=JUDGE_ENV_FILE)
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.force:
        print(f"{args.out} already exists; pass --force to replace it", file=sys.stderr)
        return 1
    try:
        values = ask_judge(input, getpass.getpass)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    write_env_file(args.out, values)
    print(f"wrote {args.out} (judge {values[MODEL_VAR]} at {values[BASE_URL_VAR]}; key not shown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
