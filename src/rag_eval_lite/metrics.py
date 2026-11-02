"""Metrics for ranked document retrieval, with explicit missing-query handling."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def evaluate(expected: Iterable[dict[str, Any]], retrieved: Iterable[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    """Calculate Recall@k and MRR@k from simple local evaluation records.

    Expected rows: ``{"query_id": "q1", "relevant_ids": ["d1"]}``.
    Retrieved rows: ``{"query_id": "q1", "document_ids": ["d2", "d1"]}``.
    Queries without retrieved results score zero; unexpected retrieved queries are
    reported rather than silently affecting the denominator.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    expected_map = {row["query_id"]: set(row["relevant_ids"]) for row in expected}
    if not expected_map:
        raise ValueError("at least one expected query is required")
    retrieved_map = {row["query_id"]: row.get("document_ids", [])[:k] for row in retrieved}
    recall_hits = 0
    reciprocal_ranks: list[float] = []
    per_query: list[dict[str, Any]] = []
    for query_id, relevant in expected_map.items():
        ranking = retrieved_map.get(query_id, [])
        rank = next((index for index, document_id in enumerate(ranking, start=1) if document_id in relevant), None)
        recall_hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        per_query.append({"query_id": query_id, "hit": rank is not None, "first_relevant_rank": rank})
    total = len(expected_map)
    return {
        "k": k,
        "queries": total,
        "recall_at_k": round(recall_hits / total, 4),
        "mrr_at_k": round(sum(reciprocal_ranks) / total, 4),
        "missing_results": sorted(set(expected_map) - set(retrieved_map)),
        "unexpected_results": sorted(set(retrieved_map) - set(expected_map)),
        "per_query": per_query,
    }
