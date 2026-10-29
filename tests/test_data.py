import hashlib

import pytest
import setup_data
from conftest import write_subset
from prompt_data import SUBSETS, held_out, is_superset, load_prompts


def make_snapshot(root, sizes):
    for subset in SUBSETS:
        write_subset(root, subset, [f"{subset} item {i}" for i in range(sizes[subset])])


def test_held_out_normalizes_and_dedupes():
    assert held_out(["A b", "a  B", "c", "C "], ["a b"]) == ["c"]


def test_is_superset_is_exact_text():
    assert is_superset(["x", "y"], ["x"])
    assert not is_superset(["X", "y"], ["x"])


def test_materialize_copies_and_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_data, "PINNED_SHA256", {})
    snap = tmp_path / "snap"
    make_snapshot(snap, dict.fromkeys(SUBSETS, 3))
    manifest = setup_data.materialize(snap, tmp_path / "data")
    assert {s: m["rows"] for s, m in manifest.items()} == dict.fromkeys(SUBSETS, 3)
    assert load_prompts(tmp_path / "data", "good_500") == [f"good_500 item {i}" for i in range(3)]


def test_materialize_rejects_pin_mismatch(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    make_snapshot(snap, dict.fromkeys(SUBSETS, 2))
    monkeypatch.setattr(setup_data, "PINNED_SHA256", {"good_500/good_500.json": "0" * 64})
    with pytest.raises(RuntimeError, match="does not match upstream pin"):
        setup_data.materialize(snap, tmp_path / "data")


def test_materialize_accepts_matching_pin(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    make_snapshot(snap, dict.fromkeys(SUBSETS, 2))
    digest = hashlib.sha256((snap / "good_500" / "good_500.json").read_bytes()).hexdigest()
    monkeypatch.setattr(setup_data, "PINNED_SHA256", {"good_500/good_500.json": digest})
    assert setup_data.materialize(snap, tmp_path / "data")["good_500"]["pinned"]


def test_overlap_flags(tmp_path, synthetic_data):
    for kind in ("good",):
        write_subset(synthetic_data, f"{kind}_500", ["p"])
        write_subset(synthetic_data, f"{kind}_1000", ["p", "q"])
    flags = setup_data.overlap_flags(synthetic_data)
    assert flags["harmful"]["1000_is_superset_of_500"] is True
    assert flags["harmful"]["rows_in_1000_not_in_500"] == 5  # exact-text count
    assert flags["good"] == {"1000_is_superset_of_500": True, "rows_in_1000_not_in_500": 1}


def test_materialized_layout_loads_through_abliterix(tmp_path, monkeypatch):
    """The config's `train[400:]` slice works on a directory holding one JSON file."""
    import datasets.config
    from abliterix.data import load_prompt_dataset
    from abliterix.types import PromptSource
    from check_configs import abliterix_config

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    write_subset(tmp_path / "data", "good_500", [f"placeholder {i}" for i in range(500)])
    config = abliterix_config({"model": {"model_id": "unused"}})
    source = PromptSource(
        dataset=str(tmp_path / "data" / "good_500"), split="train[400:]", column="prompt"
    )
    messages = load_prompt_dataset(config, source)
    assert len(messages) == 100
    assert messages[0].user == "placeholder 400"
