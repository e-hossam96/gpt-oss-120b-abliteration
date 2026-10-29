import pytest
from evaluate import is_degenerate, wilson_interval


def test_wilson_known_values():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)


def test_wilson_zero_and_all():
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(0, 10)[1] == pytest.approx(0.2775, abs=1e-4)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert wilson_interval(10, 10)[0] == pytest.approx(0.7225, abs=1e-4)


@pytest.mark.parametrize("k,n", [(-1, 10), (11, 10), (0, 0)])
def test_wilson_rejects_invalid(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n ",
        "the cat sat down " * 20,
        "-" * 300,
        "*** ### !!! ??? *** ### !!! ??? ***",
        "好吃，" * 60,
        "a" * 50,
    ],
)
def test_degenerate_detected(text):
    assert is_degenerate(text)


@pytest.mark.parametrize(
    "text",
    [
        "Sure. Bread needs flour, water, salt and yeast. Mix, knead for ten minutes, "
        "let it rise for an hour, shape the loaf and bake at 230 C for about 35 minutes.",
        "Short answer: yes.",
        "今天天气很好，我们去公园散步，然后在湖边吃午饭，下午回家看书。",
    ],
)
def test_normal_text_not_degenerate(text):
    assert not is_degenerate(text)
