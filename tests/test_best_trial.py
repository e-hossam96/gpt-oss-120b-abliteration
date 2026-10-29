import best_trial
import optuna
import pytest


def make_journal(path, attrs):
    study = optuna.create_study(study_name="abliterix", storage=best_trial.open_journal(path))
    for i, (refusals, kl) in enumerate(attrs):
        trial = study.ask()
        trial.set_user_attr("index", i + 1)
        trial.set_user_attr("refusals", refusals)
        trial.set_user_attr("kl_divergence", kl)
        study.tell(trial, 0.0)
    study.ask()  # one RUNNING trial that must be ignored


def test_selects_min_refusals_then_kl(tmp_path, capsys):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    make_journal(best_trial.journal_path(ckpt, "out/stage1"), [(10, 0.01), (4, 0.05), (4, 0.02)])
    assert best_trial.main(["--checkpoint-dir", str(ckpt), "--model", "out/stage1"]) == 0
    assert capsys.readouterr().out.strip() == "3"


def test_missing_journal(tmp_path):
    assert best_trial.main(["--checkpoint-dir", str(tmp_path), "--model", "x"]) == 1


def test_no_complete_trials():
    with pytest.raises(RuntimeError):
        best_trial.select_best([])
