import json
import shlex

import pytest
from check_configs import abliterix_config, load_toml
from configure_judge import ask_judge, base_url_problem, judge_env_problems, write_env_file
from evaluate import LLMJudge
from layout import STAGES
from test_evaluate import FakeResponse

VALUES = {
    "LLM_JUDGE_MODEL": "my/judge",
    "LLM_JUDGE_BASE_URL": "https://judge.example/v1",
    "LLM_JUDGE_API_KEY": "sk-$ecret 'quoted'",
}


def answers(*replies):
    it = iter(replies)
    return lambda prompt: next(it)


def test_ask_judge_collects_three_values():
    values = ask_judge(answers(" my/judge ", "https://judge.example/v1"), answers(" k e y "))
    assert values == {
        "LLM_JUDGE_MODEL": "my/judge",
        "LLM_JUDGE_BASE_URL": "https://judge.example/v1",
        "LLM_JUDGE_API_KEY": "key",
    }


@pytest.mark.parametrize(
    "replies",
    [("", "https://judge.example/v1", "k"), ("m", "judge.example", "k"), ("m", "https://h/v1", "")],
)
def test_ask_judge_rejects_bad_input(replies):
    with pytest.raises(ValueError):
        ask_judge(answers(*replies[:2]), answers(replies[2]))


def test_base_url_must_not_include_endpoint_path():
    assert base_url_problem("https://h/v1/chat/completions")
    assert base_url_problem("http://localhost:8000/v1") is None


def test_env_file_round_trips_through_shell_quoting(tmp_path):
    path = tmp_path / ".env.judge"
    write_env_file(path, VALUES)
    parsed = dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())
    assert {k: shlex.split(v)[0] for k, v in parsed.items()} == VALUES
    assert judge_env_problems(VALUES) == []


def test_judge_request_matches_abliterix(tmp_path, monkeypatch):
    """Our request body equals abliterix 1.12.2's own for the same judge and item."""
    from abliterix.eval import detector

    sent = []

    def opener(request, timeout):
        sent.append(json.loads(request.data))
        content = json.dumps({"labels": ["C"]})
        return FakeResponse(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    cfg = load_toml(STAGES["stage2"][0])
    cfg["optimization"]["checkpoint_dir"] = str(tmp_path)
    cfg["detection"]["llm_judge_model"] = VALUES["LLM_JUDGE_MODEL"]
    cfg["detection"]["llm_judge_base_url"] = VALUES["LLM_JUDGE_BASE_URL"]
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "key")
    monkeypatch.setattr(detector.urllib.request, "urlopen", opener)
    item = ("placeholder question", "placeholder response " * 200)
    assert detector.RefusalDetector(abliterix_config(cfg))._query_judge_api([item]) == [False]

    judge = LLMJudge(VALUES["LLM_JUDGE_MODEL"], VALUES["LLM_JUDGE_BASE_URL"], "key", opener)
    assert judge.is_refusal(*item) is False
    assert sent[1] == sent[0]
