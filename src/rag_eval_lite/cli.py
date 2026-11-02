from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import evaluate


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval rankings.")
    parser.add_argument("expected", type=Path, help="JSONL expected query/relevance rows")
    parser.add_argument("retrieved", type=Path, help="JSONL retrieved ranking rows")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("rag_metrics.json"))
    args = parser.parse_args()
    report = evaluate(_jsonl(args.expected), _jsonl(args.retrieved), args.k)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("queries", "recall_at_k", "mrr_at_k")}))
