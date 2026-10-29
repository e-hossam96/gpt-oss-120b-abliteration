import json
from pathlib import Path

import pytest


def write_subset(data_dir: Path, subset: str, prompts: list[str]) -> Path:
    target = data_dir / subset
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{subset}.json"
    records = [{"id": i, "prompt": p, "language": "en"} for i, p in enumerate(prompts)]
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    small = [f"placeholder question {i}" for i in range(5)]
    write_subset(data, "harmful_500", small)
    # Contains all of `small`, case/whitespace variants of two rows, and 3 new rows.
    write_subset(
        data,
        "harmful_1000",
        small
        + ["PLACEHOLDER  question 0", "placeholder question 1 "]
        + [f"held out {i}" for i in range(3)],
    )
    return data
