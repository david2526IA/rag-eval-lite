import pytest

from rag_eval_lite.metrics import evaluate


EXPECTED = [{"query_id": "q1", "relevant_ids": ["d1"]}, {"query_id": "q2", "relevant_ids": ["d2", "d3"]}]


def test_metrics_are_reproducible_and_explainable() -> None:
    report = evaluate(EXPECTED, [{"query_id": "q1", "document_ids": ["d4", "d1"]}, {"query_id": "q2", "document_ids": ["d3"]}], k=2)
    assert report["recall_at_k"] == 1.0
    assert report["mrr_at_k"] == 0.75
    assert report["per_query"][0]["first_relevant_rank"] == 2


def test_missing_results_count_as_zero_and_are_reported() -> None:
    report = evaluate(EXPECTED, [{"query_id": "q1", "document_ids": ["d1"]}], k=1)
    assert report["recall_at_k"] == 0.5
    assert report["missing_results"] == ["q2"]


def test_invalid_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        evaluate(EXPECTED, [], k=0)
