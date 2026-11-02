# RAG Eval Lite

A small, dependency-free evaluator for the retrieval step of a RAG system. It calculates Recall@k and MRR@k from local JSONL files and reports missing or unexpected query results explicitly.

This helps compare changes to chunking, embeddings, ranking or filters without confusing retrieval quality with generation quality.

## Input

`expected.jsonl`:

```json
{"query_id":"q1","relevant_ids":["manual-12"]}
```

`retrieved.jsonl`:

```json
{"query_id":"q1","document_ids":["manual-7","manual-12"]}
```

## Run

```bash
pip install .
rag-eval expected.jsonl retrieved.jsonl --k 5 --output rag_metrics.json
```

The output includes aggregate metrics and per-query first-relevant rank, allowing failures to be investigated rather than hidden by a single score.

## Scope

This evaluates ranked retrieval only. It does not evaluate answer factuality, safety, citations or user experience; those require additional task-specific evaluation.

## Development

```bash
pip install ".[dev]"
pytest -q
```

## License

MIT
