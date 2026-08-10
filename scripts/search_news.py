"""Run bounded semantic news retrieval without generating an LLM answer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.repositories import StockRepository
from app.services import SemanticNewsSearchService
from pipelines.embeddings import QueryEmbeddingService


PREVIEW_CHARACTERS = 240


def build_search_service() -> SemanticNewsSearchService:
    """Compose lazy query embedding with the existing repository boundary."""

    return SemanticNewsSearchService(
        StockRepository(),
        QueryEmbeddingService(),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search embedded financial-news chunks using cosine similarity."
    )
    parser.add_argument("query", help="Natural-language semantic search query")
    parser.add_argument("--ticker", help="Optional ticker filter")
    parser.add_argument("--top-k", type=int, default=5, help="Result count (max 20)")
    return parser.parse_args(argv)


def _safe_text(value: Any, *, maximum: int) -> str:
    return " ".join(str(value or "").split())[:maximum]


def print_results(results: list[dict[str, Any]]) -> None:
    """Print concise citation-ready result metadata and chunk previews."""

    if not results:
        print("No semantically similar news chunks were found.")
        return
    for rank, result in enumerate(results, start=1):
        similarity = float(result["similarity"])
        print(
            f"{rank}. {_safe_text(result.get('title'), maximum=160)} | "
            f"published {_safe_text(result.get('published_at'), maximum=40)} | "
            f"similarity {similarity:.4f}"
        )
        print(
            "   "
            + _safe_text(result.get("chunk_text"), maximum=PREVIEW_CHARACTERS)
        )


def main(
    argv: list[str] | None = None,
    *,
    service: SemanticNewsSearchService | None = None,
) -> int:
    args = _parse_args(argv)
    try:
        active_service = service or build_search_service()
        results = active_service.semantic_news_search(
            args.query,
            ticker=args.ticker,
            top_k=args.top_k,
        )
    except Exception:
        print("Semantic news search failed safely.", file=sys.stderr)
        return 1

    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
