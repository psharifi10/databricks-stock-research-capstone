"""Thin FastMCP tools over persisted stock-research services."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from app.agent_client import AgentServiceError, SupervisorAgentClient
from app.repositories import StockRepository
from app.services import (
    ResearchActionService,
    ResearchContextService,
    SemanticNewsSearchService,
)
from app.validation import (
    ValidationError,
    normalize_bounded_text,
    normalize_bounded_limit,
    normalize_ticker,
)
from pipelines.embeddings import (
    EmbeddingError,
    QueryEmbeddingService,
    normalize_query_text,
)
from mcp_server.frontend import load_static_asset


mcp = FastMCP("Stock Research MCP")


@dataclass(frozen=True)
class ResearchServices:
    """Reusable service composition for all MCP tool calls."""

    repository: StockRepository
    semantic_search: SemanticNewsSearchService
    research_context: ResearchContextService
    actions: ResearchActionService


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
        actions=ResearchActionService(repository),
    )


@lru_cache(maxsize=1)
def get_agent_client() -> SupervisorAgentClient:
    """Lazily construct the unified-auth Supervisor client on first use."""

    return SupervisorAgentClient()


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


@mcp.custom_route("/", methods=["GET"], include_in_schema=False)
async def frontend_home(_request: Request) -> Response:
    """Serve the stock-research dashboard from the existing MCP application."""

    content, _media_type = load_static_asset("index.html")
    return HTMLResponse(content)


@mcp.custom_route(
    "/static/styles.css",
    methods=["GET"],
    include_in_schema=False,
)
async def frontend_styles(_request: Request) -> Response:
    """Serve the dashboard stylesheet as a fixed local asset."""

    content, media_type = load_static_asset("styles.css")
    return Response(content, media_type=media_type)


@mcp.custom_route(
    "/static/app.js",
    methods=["GET"],
    include_in_schema=False,
)
async def frontend_script(_request: Request) -> Response:
    """Serve the dashboard JavaScript as a fixed local asset."""

    content, media_type = load_static_asset("app.js")
    return Response(content, media_type=media_type)


@mcp.custom_route("/api/research", methods=["POST"])
async def research_api(request: Request) -> Response:
    """Return existing grounded research context through a safe JSON boundary."""

    try:
        payload = await request.json()
    except (JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            _error("validation_error", "A valid JSON object is required."),
            status_code=400,
        )

    try:
        if not isinstance(payload, Mapping):
            raise ValidationError("A JSON object is required.")
        context = get_services().research_context.build_research_context(
            payload.get("ticker"),
            payload.get("question"),
        )
        return JSONResponse(_success(context))
    except (ValidationError, EmbeddingError) as error:
        return JSONResponse(
            _error("validation_error", str(error)),
            status_code=400,
        )
    except Exception:
        return JSONResponse(
            _error(
                "service_error",
                "The stock research service could not complete the request.",
            ),
            status_code=500,
        )


@mcp.custom_route("/api/agent", methods=["POST"])
async def agent_api(request: Request) -> Response:
    """Return a grounded synthesis from the configured Supervisor Agent."""

    try:
        payload = await request.json()
    except (JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            _error("validation_error", "A valid JSON object is required."),
            status_code=400,
        )

    try:
        if not isinstance(payload, Mapping):
            raise ValidationError("A JSON object is required.")
        symbol = normalize_ticker(payload.get("ticker"))
        question = normalize_bounded_text(
            normalize_query_text(payload.get("question")),
            field_name="Research question",
            maximum=5000,
        )
        prompt = (
            f"Research ticker {symbol}.\n\n"
            f"User question:\n{question}\n\n"
            "Use the available stock research MCP tools when evidence is needed.\n"
            "Base the answer only on available evidence."
        )
        answer = get_agent_client().generate_response(prompt)
        return JSONResponse(
            _success(
                {
                    "ticker": symbol,
                    "answer": answer,
                }
            )
        )
    except (ValidationError, EmbeddingError) as error:
        return JSONResponse(
            _error("validation_error", str(error)),
            status_code=400,
        )
    except AgentServiceError:
        return JSONResponse(
            _error(
                "agent_unavailable",
                "The AI research summary is temporarily unavailable.",
            ),
            status_code=503,
        )
    except Exception:
        return JSONResponse(
            _error(
                "service_error",
                "The AI research service could not complete the request.",
            ),
            status_code=500,
        )


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
def add_to_watchlist(user_email: str, ticker: str) -> dict[str, Any]:
    """Use when the user explicitly asks to add a ticker to their stock research watchlist."""

    return _execute_tool(
        lambda: get_services().actions.add_to_watchlist(user_email, ticker)
    )


@mcp.tool
def remove_from_watchlist(user_email: str, ticker: str) -> dict[str, Any]:
    """Use when the user explicitly asks to remove a ticker from their stock research watchlist."""

    return _execute_tool(
        lambda: get_services().actions.remove_from_watchlist(
            user_email,
            ticker,
        )
    )


@mcp.tool
def save_research_note(
    user_email: str,
    ticker: str,
    note_text: str,
) -> dict[str, Any]:
    """Use when the user explicitly wants to save a personal research note about a ticker."""

    return _execute_tool(
        lambda: get_services().actions.save_research_note(
            user_email,
            ticker,
            note_text,
        )
    )


@mcp.tool
def save_analysis_report(
    user_email: str,
    ticker: str,
    title: str,
    report_text: str,
) -> dict[str, Any]:
    """Use when the user explicitly wants to persist a completed analysis or report."""

    return _execute_tool(
        lambda: get_services().actions.save_analysis_report(
            user_email,
            ticker,
            title,
            report_text,
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
