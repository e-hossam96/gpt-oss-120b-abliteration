from __future__ import annotations

import json
from pathlib import Path

SUBSETS = ("good_500", "harmful_500", "good_1000", "harmful_1000")


def read_records(path: Path) -> list[dict]:
    # Accepts a JSON array or JSON Lines.
    text = path.read_text(encoding="utf-8")
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path.name}: expected a non-empty list of records")
    for i, rec in enumerate(records):
        if not isinstance(rec, dict) or not isinstance(rec.get("prompt"), str):
            raise ValueError(f"{path.name}: record {i} has no string 'prompt' field")
    return records


def subset_file(data_dir: Path, subset: str) -> Path:
    files = sorted((data_dir / subset).glob("*.json*"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected exactly one JSON file in {data_dir / subset}, found {len(files)} "
            "(run scripts/setup_data.py)"
        )
    return files[0]


def load_prompts(data_dir: Path, subset: str) -> list[str]:
    return [rec["prompt"] for rec in read_records(subset_file(data_dir, subset))]


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def held_out(full: list[str], used: list[str]) -> list[str]:
    # Prompts in `full` but not in `used`, ignoring case and whitespace, without duplicates.
    seen = {normalize(p) for p in used}
    result = []
    for prompt in full:
        key = normalize(prompt)
        if key not in seen:
            seen.add(key)
            result.append(prompt)
    return result


def is_superset(full: list[str], subset: list[str]) -> bool:
    return set(subset) <= set(full)
