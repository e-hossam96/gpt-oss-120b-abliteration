import io
import json

import pytest
from conftest import write_subset
from evaluate import JUDGE_MAX_TOKENS, LLMJudge, evaluate_model, parse_args, run


class FakeJudge:
    model = "judge/x"
    url = "https://judge.example/v1/chat/completions"

    def __init__(self):
        self.calls = []

    def is_refusal(self, question, response):
        self.calls.append((question, response))
        return response.startswith("I can't")


def fake_generate(responses_by_model):
    def generate(model, prompts, max_new_tokens):
        assert max_new_tokens >= 256
        return [responses_by_model[model](i) for i in range(len(prompts))]

    return generate


def make_args(tmp_path, data, *extra):
    return parse_args(
        [
            "--model",
            "m",
            "--data-dir",
            str(data),
            "--out-dir",
            str(tmp_path / "out"),
            "--heretic-summary",
            str(tmp_path / "none.json"),
            *extra,
        ]
    )


def test_evaluate_model_counts_and_degenerate_skip_judge():
    judge = FakeJudge()
    responses = ["I can't help with that.", "Here is a long, specific answer about it.", ""]
    summary, records = evaluate_model(
        "x", "m", ["q1", "q2", "q3"], lambda m, p, n: responses, judge, 256
    )
    assert summary["refusals"] == 2
    assert summary["degenerate"] == 1
    assert summary["judge_refusals"] == 1
    assert len(judge.calls) == 2  # the empty response never reaches the judge
    assert records[2]["judge_refusal"] is None


def test_evaluate_model_rejects_short_generation():
    with pytest.raises(ValueError):
        evaluate_model("x", "m", ["q"], lambda m, p, n: ["a"], FakeJudge(), 100)


def test_run_uses_only_held_out_prompts(tmp_path, synthetic_data):
    judge = FakeJudge()
    seen = {}

    def generate(model, prompts, max_new_tokens):
        seen[model] = prompts
        return ["A specific and coherent answer to the question asked."] * len(prompts)

    args = make_args(tmp_path, synthetic_data, "--baseline", "base")
    assert run(args, generate, judge) == 0
    assert seen["m"] == ["held out 0", "held out 1", "held out 2"]
    assert seen["base"] == seen["m"]
    results = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert [r["label"] for r in results["results"]] == ["abliterated", "baseline"]
    assert results["meta"]["n_prompts"] == 3
    assert results["meta"]["judge_base_url"] == "https://judge.example/v1"
    assert results["heretic_crosscheck"] is None
    assert "Heretic" not in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")


def test_run_stops_loudly_without_held_out_rows(tmp_path, capsys):
    data = tmp_path / "data"
    write_subset(data, "harmful_500", ["a", "b"])
    write_subset(data, "harmful_1000", ["b", "A "])
    assert run(make_args(tmp_path, data), fake_generate({}), FakeJudge()) == 2
    assert "HELD-OUT EVALUATION NOT POSSIBLE" in capsys.readouterr().err


def test_run_stops_loudly_without_data(tmp_path, capsys):
    assert run(make_args(tmp_path, tmp_path / "missing"), fake_generate({}), FakeJudge()) == 2
    assert "HELD-OUT EVALUATION NOT POSSIBLE" in capsys.readouterr().err


def test_heretic_section_is_separate(tmp_path, synthetic_data):
    heretic = {
        "heretic_version": "1.4.0",
        "base_model": "b",
        "modified_model": "m",
        "kl_divergence": 0.12,
        "refusals": 3,
        "total_prompts": 100,
        "base_refusals": 90,
        "detector": "substring",
        "datasets": {},
    }
    path = tmp_path / "heretic.json"
    path.write_text(json.dumps(heretic), encoding="utf-8")
    args = make_args(tmp_path, synthetic_data, "--heretic-summary", str(path))
    run(args, lambda m, p, n: ["fine answer with content"] * len(p), FakeJudge())
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "## Heretic cross-check (separate yardstick)" in summary
    assert "never merged" in summary
    results = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert results["heretic_crosscheck"]["kl_divergence"] == 0.12


def test_parse_args_enforces_min_tokens():
    with pytest.raises(SystemExit):
        parse_args(["--model", "m", "--max-new-tokens", "128"])


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_judge_request_shape():
    sent = []

    def opener(request, timeout):
        sent.append(request)
        content = json.dumps({"labels": ["R"]})
        return FakeResponse(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    judge = LLMJudge("my/judge", "https://judge.example/v1/", "key", opener=opener)
    assert judge.is_refusal("q", "r" * 5000) is True
    assert sent[0].full_url == "https://judge.example/v1/chat/completions"
    assert sent[0].get_header("Authorization") == "Bearer key"
    body = json.loads(sent[0].data)
    assert body["model"] == "my/judge"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == JUDGE_MAX_TOKENS == 343
    assert body["messages"][0]["content"].endswith("1. Question: q\n   Response: " + "r" * 2000)
