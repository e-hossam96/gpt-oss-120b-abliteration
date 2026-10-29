import json

import heretic_crosscheck as hc
import pytest

# Synthetic log in the Heretic 1.4.0 line format. No real model output.
SYNTHETIC_LOG = """\
\x1b[36m█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀\x1b[0m  v1.4.0
█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░  https://heretic-project.org

Loading model /models/base...
* Chosen batch size: 8

Loading good evaluation prompts from mlabonne/harmless_alpaca...
* 100 prompts loaded
* Obtaining first-token probability distributions...

Loading bad evaluation prompts from mlabonne/harmful_behaviors...
* 100 prompts loaded
* Counting model refusals...
* Initial refusals: 97/100

Loading model /models/modified...
* Evaluating...
  * Obtaining first-token probability distributions...
  * KL divergence: 0.1234
  * Counting model refusals...
  * Refusals: 4/100
"""


def test_parse_synthetic_log():
    parsed = hc.parse_output(SYNTHETIC_LOG)
    assert parsed["heretic_version"] == "1.4.0"
    assert parsed["kl_divergence"] == 0.1234
    assert (parsed["refusals"], parsed["total_prompts"], parsed["base_refusals"]) == (4, 100, 97)
    assert parsed["datasets"] == {
        "good_evaluation_prompts": "mlabonne/harmless_alpaca",
        "bad_evaluation_prompts": "mlabonne/harmful_behaviors",
    }


def test_parser_fails_loudly_on_missing_metric():
    log = SYNTHETIC_LOG.replace("  * KL divergence: 0.1234\n", "")
    with pytest.raises(hc.HereticOutputError, match="kl_divergence"):
        hc.parse_output(log)


def test_parser_refuses_unverified_version():
    with pytest.raises(hc.HereticOutputError, match="unverified"):
        hc.parse_output(SYNTHETIC_LOG.replace("v1.4.0", "v1.5.0"))


def test_dry_run_prints_exact_command(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(hc.shutil, "which", lambda name: None)
    assert hc.main(["--base", "org/base", "--modified", "org/mod", "--dry-run"]) == 0
    assert capsys.readouterr().out.strip() == "heretic --model org/base --evaluate-model org/mod"


def test_dry_run_resolves_local_paths(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(hc.shutil, "which", lambda name: None)
    local = tmp_path / "stage2"
    local.mkdir()
    hc.main(["--modified", str(local), "--dry-run"])
    out = capsys.readouterr().out
    assert str(local.resolve()) in out
    assert "--model openai/gpt-oss-120b" in out


def test_skips_cleanly_when_not_installed(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(hc.shutil, "which", lambda name: None)
    assert hc.main(["--out-dir", str(tmp_path / "out")]) == 0
    assert "SKIPPED: Heretic is not installed" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_run_writes_summary(monkeypatch, tmp_path):
    def fake_run(cmd, out_dir, log_path):
        log_path.write_text(SYNTHETIC_LOG, encoding="utf-8")
        return 0

    monkeypatch.setattr(hc, "run_heretic", fake_run)
    out = tmp_path / "out"
    code = hc.main(
        ["--heretic-bin", "heretic", "--base", "b", "--modified", "m", "--out-dir", str(out)]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["command"] == ["heretic", "--model", "b", "--evaluate-model", "m"]
    assert summary["refusals"] == 4 and summary["total_prompts"] == 100
    assert "substring" in summary["detector"]


def test_nonzero_exit_keeps_raw_log_and_skips_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(hc, "run_heretic", lambda cmd, out_dir, log_path: 3)
    out = tmp_path / "out"
    assert hc.main(["--heretic-bin", "heretic", "--out-dir", str(out)]) == 3
    assert not (out / "summary.json").exists()
