"""Download the prompt datasets into data/<subset>/ and print row counts and overlaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from layout import DATA_DIR
from prompt_data import SUBSETS, is_superset, load_prompts, read_records

DATASET_REPO = "wangzhang/abliterix-datasets"
# From upstream benchmarks/SPEC.md.
PINNED_SHA256 = {
    "harmful_500/harmful_prompts_500.json": (
        "249e48f403a8a015a808384ef567c7fa6462aaf7c51d4a78c2caeb0509bd769d"
    ),
    "good_500/good_prompts_500.json": (
        "dfffd37646ddb279429323b360a8190e6d971d0d009a6b5f0b877d472cc0a8a7"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(revision: str | None) -> tuple[Path, str]:
    from huggingface_hub import HfApi, snapshot_download

    commit = HfApi().dataset_info(DATASET_REPO, revision=revision).sha
    snapshot = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=commit,
        allow_patterns=[f"{subset}/*.json*" for subset in SUBSETS],
    )
    return Path(snapshot), commit


def materialize(snapshot: Path, data_dir: Path) -> dict[str, dict]:
    manifest = {}
    for subset in SUBSETS:
        sources = sorted((snapshot / subset).glob("*.json*"))
        if len(sources) != 1:
            names = sorted(p.name for p in (snapshot / subset).glob("*"))
            raise RuntimeError(f"{subset}: expected one JSON file in the snapshot, found {names}")
        source = sources[0]
        rel = f"{subset}/{source.name}"
        digest = sha256_file(source)
        if rel in PINNED_SHA256 and digest != PINNED_SHA256[rel]:
            raise RuntimeError(f"{rel}: SHA256 {digest} does not match upstream pin")
        rows = len(read_records(source))
        target_dir = data_dir / subset
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)
        shutil.copyfile(source, target_dir / source.name)
        manifest[subset] = {
            "file": rel,
            "sha256": digest,
            "rows": rows,
            "pinned": rel in PINNED_SHA256,
        }
    return manifest


def overlap_flags(data_dir: Path) -> dict[str, dict]:
    flags = {}
    for kind in ("good", "harmful"):
        small = load_prompts(data_dir, f"{kind}_500")
        full = load_prompts(data_dir, f"{kind}_1000")
        flags[kind] = {
            "1000_is_superset_of_500": is_superset(full, small),
            "rows_in_1000_not_in_500": len(set(full) - set(small)),
        }
    return flags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", help="dataset repo revision (default: main, resolved to a commit)"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)

    snapshot, commit = download(args.revision)
    files = materialize(snapshot, args.data_dir)
    overlap = overlap_flags(args.data_dir)
    manifest = {"repo": DATASET_REPO, "commit": commit, "files": files, "overlap": overlap}
    (args.data_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"{DATASET_REPO} @ {commit}")
    for subset, info in files.items():
        pin = "sha256 pinned+verified" if info["pinned"] else "not pinned upstream"
        print(f"  {subset}: {info['rows']} rows ({pin})")
    for kind, info in overlap.items():
        print(
            f"  {kind}_1000 superset of {kind}_500: {info['1000_is_superset_of_500']}; "
            f"rows only in 1000: {info['rows_in_1000_not_in_500']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
