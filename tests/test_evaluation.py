import pytest

from ragbench.evaluation.retrieval import aggregate_metrics, recall_at_k, reciprocal_rank_at_k


def test_recall_handles_multiple_gold_and_k_larger_than_results() -> None:
    assert recall_at_k(["a", "x"], ["a", "b"], 10) == 0.5
    assert recall_at_k(["x"], ["a"], 5) == 0.0


def test_reciprocal_rank_edges() -> None:
    assert reciprocal_rank_at_k(["a", "x"], ["a"], 2) == 1.0
    assert reciprocal_rank_at_k(["x", "a"], ["a"], 2) == 0.5
    assert reciprocal_rank_at_k(["x", "a"], ["a"], 1) == 0.0


@pytest.mark.parametrize("gold", [[], [""]])
def test_metrics_reject_invalid_gold(gold: list[str]) -> None:
    with pytest.raises(ValueError):
        recall_at_k([], gold, 1)


def test_macro_average() -> None:
    assert aggregate_metrics(
        [
            {"recall_at_k": 1.0, "reciprocal_rank_at_k": 0.5},
            {"recall_at_k": 0.0, "reciprocal_rank_at_k": 0.0},
        ]
    ) == {"recall_at_k": 0.5, "mrr_at_k": 0.25}
