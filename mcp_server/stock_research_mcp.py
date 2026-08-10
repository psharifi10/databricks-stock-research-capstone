"""Thin read-only FastMCP tools over persisted stock-research services."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import FastMCP

from app.repositories import StockRepository
from app.services import ResearchContextService, SemanticNewsSearchService
from app.validation import (
    ValidationError,
    normalize_bounded_limit,
    normalize_ticker,
)
from pipelines.embeddings import (
    EmbeddingError,
    QueryEmbeddingService,
    normalize_query_text,
)


mcp = FastMCP("Stock Research MCP")


@dataclass(frozen=True)
class ResearchServices:
    """Reusable read-only service composition for all MCP tool calls."""

    repository: StockRepository
    semantic_search: SemanticNewsSearchService
    research_context: ResearchContextService


@lru_cache(maxsize=1)
def get_services() -> ResearchServices:
    """Lazily compose services without connecting to Lakebase at import time."""

    repository = StockRepository()
    semantic_search = SemanticNewsSearchService(
        repository,
        QueryEmbeddingService(),
    )
    return ResearchServices(
        repository=repository,
        semantic_search=semantic_search,
        research_context=ResearchContextService(repository, semantic_search),
    )


class _ToolNotFoundError(LookupError):
    """Internal signal for a safe MCP not-found response."""


def _success(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": _json_compatible(data)}


def _error(error_type: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def _execute_tool(operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        return _success(operation())
    except (ValidationError, EmbeddingError) as error:
        return _error("validation_error", str(error))
    except _ToolNotFoundError as error:
        return _error("not_found", str(error))
    except Exception:
        return _error(
            "service_error",
            "The stock research service could not complete the request.",
        )


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
            if str(key).lower() not in {"embedding", "vector", "query_embedding"}
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return str(value)


def _citation_ready_news(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "article_id": row.get("article_id"),
            "chunk_index": row.get("chunk_index"),
            "title": row.get("title"),
            "article_url": row.get("article_url"),
            "publisher": row.get("publisher_name"),
            "published_at": row.get("published_at"),
            "sentiment": row.get("sentiment"),
            "sentiment_reasoning": row.get("sentiment_reasoning"),
            "similarity": row.get("similarity"),
            "chunk_text": row.get("chunk_text"),
        }
        for row in rows
    ]


@mcp.tool
def get_company(ticker: str) -> dict[str, Any]:
    """Retrieve persisted company metadata when an agent needs company facts."""

    def operation() -> dict[str, Any]:
        symbol = normalize_ticker(ticker)
        company = get_services().repository.get_company(symbol)
        if company is None:
            raise _ToolNotFoundError(
                f"No persisted company data was found for {symbol}."
            )
        return company

    return _execute_tool(operation)


@mcp.tool
def get_price_history(ticker: str, limit: int = 30) -> dict[str, Any]:
    """Retrieve persisted daily prices when an agent needs recent history."""

    def operation() -> list[dict[str, Any]]:
        symbol = normalize_ticker(ticker)
        bounded_limit = normalize_bounded_limit(
            limit,
            field_name="Price history limit",
            maximum=90,
        )
        return get_services().repository.list_recent_prices(
            symbol,
            bounded_limit,
        )

    return _execute_tool(operation)


@mcp.tool
def search_financial_news(
    ticker: str,
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve persisted semantic news evidence for a ticker research query."""

    def operation() -> list[dict[str, Any]]:
        symbol = normalize_ticker(ticker)
        normalized_query = normalize_query_text(query)
        bounded_top_k = normalize_bounded_limit(
            top_k,
            field_name="top_k",
            maximum=10,
        )
        rows = get_services().semantic_search.semantic_news_search(
            normalized_query,
            ticker=symbol,
            top_k=bounded_top_k,
        )
        return _citation_ready_news(rows)

    return _execute_tool(operation)


@mcp.tool
def build_research_context(ticker: str, question: str) -> dict[str, Any]:
    """Retrieve a grounded persisted research package for later agent analysis."""

    return _execute_tool(
        lambda: get_services().research_context.build_research_context(
            ticker,
            question,
        )
    )


@mcp.tool
def health() -> dict[str, Any]:
    """Return non-secret MCP process status for lightweight health checks."""

    return _success(
        {
            "status": "ok",
            "service": "stock-research-mcp",
        }
    )


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
        stateless_http=True,
    )
