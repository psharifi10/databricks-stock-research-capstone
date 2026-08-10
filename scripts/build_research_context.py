"""Build bounded grounded research context without calling an LLM."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.repositories import StockRepository
from app.services import ResearchContextService, SemanticNewsSearchService
from pipelines.embeddings import QueryEmbeddingService


def build_context_service() -> ResearchContextService:
    """Compose Lakebase reads with the existing semantic-search boundary."""

    repository = StockRepository()
    semantic_search = SemanticNewsSearchService(
        repository,
        QueryEmbeddingService(),
    )
    return ResearchContextService(repository, semantic_search)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic grounded stock-research context."
    )
    parser.add_argument("ticker", help="Stock ticker")
    parser.add_argument("question", help="Research question")
    parser.add_argument("--semantic-top-k", type=int, default=8)
    parser.add_argument("--recent-news-limit", type=int, default=5)
    parser.add_argument("--price-history-limit", type=int, default=30)
    return parser.parse_args(argv)


def _safe_text(value: Any, *, maximum: int) -> str:
    return " ".join(str(value or "").split())[:maximum]


def print_context_preview(context: Mapping[str, Any]) -> None:
    """Print counts and citation metadata without vectors or full chunks."""

    company = context.get("company")
    prices = list(context.get("prices") or [])
    recent_news = list(context.get("recent_news") or [])
    semantic_evidence = list(context.get("semantic_evidence") or [])
    print(f"Research context: {_safe_text(context.get('ticker'), maximum=16)}")
    print(f"Company records: {1 if company else 0}")
    print(f"Price observations: {len(prices)}")
    print(f"Recent news articles: {len(recent_news)}")
    print(f"Semantic evidence chunks: {len(semantic_evidence)}")
    for rank, evidence in enumerate(semantic_evidence, start=1):
        similarity = float(evidence.get("similarity") or 0.0)
        print(
            f"{rank}. {_safe_text(evidence.get('title'), maximum=160)} | "
            f"similarity {similarity:.4f}"
        )


def main(
    argv: list[str] | None = None,
    *,
    service: ResearchContextService | None = None,
) -> int:
    args = _parse_args(argv)
    try:
        active_service = service or build_context_service()
        context = active_service.build_research_context(
            args.ticker,
            args.question,
            semantic_top_k=args.semantic_top_k,
            recent_news_limit=args.recent_news_limit,
            price_history_limit=args.price_history_limit,
        )
    except Exception:
        print("Research context assembly failed safely.", file=sys.stderr)
        return 1

    print_context_preview(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
