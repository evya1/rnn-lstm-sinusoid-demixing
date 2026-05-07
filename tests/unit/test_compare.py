"""Unit tests for evaluation/compare.py."""

from rnn_lstm_sinusoid_demixing.evaluation.compare import compare_models


class TestCompareModels:
    def test_sorted_ascending(self) -> None:
        results = {"rnn": 0.05, "fc": 0.02, "lstm": 0.03}
        out = compare_models(results)
        values = list(out.values())
        assert values == sorted(values)

    def test_keys_preserved(self) -> None:
        results = {"rnn": 0.05, "fc": 0.02, "lstm": 0.03}
        out = compare_models(results)
        assert set(out.keys()) == {"rnn", "fc", "lstm"}

    def test_first_key_is_best(self) -> None:
        results = {"rnn": 0.05, "fc": 0.02, "lstm": 0.03}
        out = compare_models(results)
        assert list(out.keys())[0] == "fc"

    def test_single_model(self) -> None:
        out = compare_models({"fc": 0.1})
        assert out == {"fc": 0.1}

    def test_returns_dict(self) -> None:
        out = compare_models({"a": 1.0, "b": 0.5})
        assert isinstance(out, dict)

    def test_equal_mse_stable(self) -> None:
        results = {"a": 0.1, "b": 0.1, "c": 0.1}
        out = compare_models(results)
        assert list(out.values()) == [0.1, 0.1, 0.1]
