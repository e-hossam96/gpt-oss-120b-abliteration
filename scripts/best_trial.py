"""Print the trial number to export from an abliterix Optuna journal.

Selection rule is abliterix 1.12.2's own non-interactive export rule (cli.py): among
COMPLETE trials, minimize (refusals, kl_divergence). The printed number is the trial's
`index` user attribute, which upstream scripts/export_model.py --trial matches first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
from abliterix.util import slugify_model_name
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import FrozenTrial, TrialState


def journal_path(checkpoint_dir: Path, model_id: str) -> Path:
    return checkpoint_dir / f"{slugify_model_name(model_id)}.jsonl"


def open_journal(path: Path) -> JournalStorage:
    # Open-file lock, as abliterix uses; Optuna's default symlink lock fails on Windows.
    lock = JournalFileOpenLock(str(path))
    return JournalStorage(JournalFileBackend(str(path), lock_obj=lock))


def select_best(trials: list[FrozenTrial]) -> FrozenTrial:
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("no COMPLETE trial in the journal")
    return min(complete, key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="model_id exactly as in the stage config")
    args = parser.parse_args(argv)

    path = journal_path(args.checkpoint_dir, args.model)
    if not path.is_file():
        print(f"journal not found: {path}", file=sys.stderr)
        return 1
    study = optuna.load_study(study_name="abliterix", storage=open_journal(path))
    best = select_best(study.trials)
    print(
        f"best trial index={best.user_attrs.get('index')} number={best.number} "
        f"refusals={best.user_attrs['refusals']} kl={best.user_attrs['kl_divergence']}",
        file=sys.stderr,
    )
    print(best.user_attrs.get("index", best.number))
    return 0


if __name__ == "__main__":
    sys.exit(main())
