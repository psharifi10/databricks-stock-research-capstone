"""Framework-independent orchestration for stock research operations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from app.massive_client import MassiveClient
from app.repositories import StockRepository
from app.validation import (
    MAX_SEMANTIC_SEARCH_RESULTS,
    normalize_bounded_text,
    normalize_bounded_limit,
    normalize_email,
    normalize_ticker,
    normalize_top_k,
)
from pipelines.embeddings import (
    QueryEmbeddingService,
    normalize_query_text,
    validate_embedding,
)


class StockResearchService:
    """Compose Massive reads with idempotent Lakebase persistence."""

    def __init__(
        self,
        massive_client: MassiveClient,
        repository: StockRepository,
    ) -> None:
        self._massive_client = massive_client
        self._repository = repository

    def refresh_company(self, ticker: str) -> dict[str, Any]:
        symbol = normalize_ticker(ticker)
        company = self._massive_client.get_company_overview(symbol)
        self._repository.upsert_company(company)
        return company

    def refresh_price_history(
        self,
        ticker: str,
        start_date: date | str,
        end_date: date | str,
        *,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        symbol = normalize_ticker(ticker)
        rows = self._massive_client.get_historical_prices(
            symbol,
            start_date,
            end_date,
            max_pages=max_pages,
        )
        self._repository.upsert_price_snapshots(symbol, rows)
        return rows

    def refresh_news(
        self,
        ticker: str,
        *,
        limit: int = 25,
        published_after: date | datetime | str | None = None,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        symbol = normalize_ticker(ticker)
        articles = self._massive_client.get_news(
            symbol,
            limit=limit,
            published_after=published_after,
            max_pages=max_pages,
        )
        self._repository.upsert_news_articles(articles)
        return articles


class SemanticNewsSearchService:
    """Embed a query and delegate bounded cosine retrieval to the repository."""

    def __init__(
        self,
        repository: StockRepository,
        embedding_service: QueryEmbeddingService,
    ) -> None:
        self._repository = repository
        self._embedding_service = embedding_service

    def semantic_news_search(
        self,
        query: str,
        *,
        ticker: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        normalized_query = normalize_query_text(query)
        symbol = normalize_ticker(ticker) if ticker is not None else None
        limit = normalize_top_k(top_k)
        query_embedding = validate_embedding(
            self._embedding_service.embed_query(normalized_query)
        )
        return self._repository.search_news_chunks(
            query_embedding,
            ticker=symbol,
            top_k=limit,
        )


class ResearchActionService:
    """Validate explicit user actions and delegate Lakebase mutations."""

    def __init__(self, repository: StockRepository) -> None:
        self._repository = repository

    def add_to_watchlist(
        self,
        user_email: str,
        ticker: str,
    ) -> dict[str, Any]:
        email = normalize_email(user_email)
        symbol = normalize_ticker(ticker)
        user_id = self._resolve_user_id(email)
        added = self._repository.add_watchlist_ticker(user_id, symbol)
        return {
            "user_email": email,
            "ticker": symbol,
            "added": added,
            "already_present": not added,
        }

    def remove_from_watchlist(
        self,
        user_email: str,
        ticker: str,
    ) -> dict[str, Any]:
        email = normalize_email(user_email)
        symbol = normalize_ticker(ticker)
        user_id = self._resolve_user_id(email)
        removed = self._repository.remove_watchlist_ticker(user_id, symbol)
        return {
            "user_email": email,
            "ticker": symbol,
            "removed": removed,
        }

    def save_research_note(
        self,
        user_email: str,
        ticker: str,
        note_text: str,
    ) -> dict[str, Any]:
        email = normalize_email(user_email)
        symbol = normalize_ticker(ticker)
        note = normalize_bounded_text(
            note_text,
            field_name="Research note",
            maximum=5000,
        )
        user_id = self._resolve_user_id(email)
        saved = self._repository.save_research_note(user_id, symbol, note)
        return {
            "note_id": saved["id"],
            "user_email": email,
            "ticker": symbol,
            "created_at": saved.get("created_at"),
            "saved": True,
        }

    def save_analysis_report(
        self,
        user_email: str,
        ticker: str,
        title: str,
        report_text: str,
    ) -> dict[str, Any]:
        email = normalize_email(user_email)
        symbol = normalize_ticker(ticker)
        normalized_title = normalize_bounded_text(
            title,
            field_name="Analysis report title",
            maximum=200,
        )
        report = normalize_bounded_text(
            report_text,
            field_name="Analysis report body",
            maximum=20000,
        )
        user_id = self._resolve_user_id(email)
        saved = self._repository.save_analysis_report(
            user_id,
            symbol,
            normalized_title,
            report,
        )
        return {
            "report_id": saved["id"],
            "user_email": email,
            "ticker": symbol,
            "title": normalized_title,
            "created_at": saved.get("created_at"),
            "saved": True,
        }

    def _resolve_user_id(self, email: str) -> int:
        user = self._repository.get_or_create_user(email)
        return int(user["id"])


class ResearchContextService:
    """Assemble deterministic, citation-ready Lakebase research evidence."""

    def __init__(
        self,
        repository: StockRepository,
        semantic_search_service: SemanticNewsSearchService,
    ) -> None:
        self._repository = repository
        self._semantic_search_service = semantic_search_service

    def build_research_context(
        self,
        ticker: str,
        question: str,
        *,
        semantic_top_k: int = 8,
        recent_news_limit: int = 5,
        price_history_limit: int = 30,
    ) -> dict[str, Any]:
        symbol = normalize_ticker(ticker)
        normalized_question = normalize_query_text(question)
        evidence_limit = normalize_top_k(semantic_top_k)
        news_limit = normalize_bounded_limit(
            recent_news_limit,
            field_name="Recent news limit",
            maximum=100,
        )
        price_limit = normalize_bounded_limit(
            price_history_limit,
            field_name="Price history limit",
            maximum=100,
        )
        candidate_limit = min(
            MAX_SEMANTIC_SEARCH_RESULTS,
            evidence_limit * 2,
        )

        company = self._repository.get_company(symbol)
        prices = self._repository.list_recent_prices(symbol, price_limit)
        recent_news = self._repository.list_recent_news(symbol, news_limit)
        semantic_candidates = self._semantic_search_service.semantic_news_search(
            normalized_question,
            ticker=symbol,
            top_k=candidate_limit,
        )

        context = {
            "ticker": symbol,
            "question": normalized_question,
            "company": _company_context(company),
            "prices": [_price_context(row) for row in prices],
            "recent_news": [_recent_news_context(row) for row in recent_news],
            "semantic_evidence": _diversify_semantic_evidence(
                semantic_candidates,
                ticker=symbol,
                limit=evidence_limit,
            ),
        }
        return _json_safe(context)


def _company_context(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "ticker": row.get("ticker"),
        "name": row.get("name"),
        "description": row.get("description"),
        "industry": row.get("industry"),
        "market_cap": row.get("market_cap"),
        "exchange": row.get("exchange"),
    }


def _price_context(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ticker": row.get("ticker"),
        "price_date": row.get("price_date"),
        "open": row.get("open"),
        "high": row.get("high"),
        "low": row.get("low"),
        "close": row.get("close"),
        "volume": row.get("volume"),
        "vwap": row.get("vwap"),
    }


def _recent_news_context(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "article_id": row.get("id"),
        "title": row.get("title"),
        "publisher": row.get("publisher"),
        "published_at": row.get("published_at"),
        "sentiment": row.get("sentiment"),
        "sentiment_reasoning": row.get("sentiment_reasoning"),
        "article_url": row.get("article_url"),
    }


def _diversify_semantic_evidence(
    candidates: list[dict[str, Any]],
    *,
    ticker: str,
    limit: int,
) -> list[dict[str, Any]]:
    diversified: list[dict[str, Any]] = []
    article_counts: dict[str, int] = {}
    for candidate in candidates:
        article_id = str(candidate.get("article_id") or "").strip()
        if not article_id or article_counts.get(article_id, 0) >= 2:
            continue
        diversified.append(
            {
                "article_id": article_id,
                "chunk_index": candidate.get("chunk_index"),
                "chunk_text": candidate.get("chunk_text"),
                "title": candidate.get("title"),
                "publisher": candidate.get("publisher_name"),
                "published_at": candidate.get("published_at"),
                "article_url": candidate.get("article_url"),
                "ticker": candidate.get("ticker") or ticker,
                "sentiment": candidate.get("sentiment"),
                "sentiment_reasoning": candidate.get("sentiment_reasoning"),
                "similarity": candidate.get("similarity"),
            }
        )
        article_counts[article_id] = article_counts.get(article_id, 0) + 1
        if len(diversified) == limit:
            break
    return diversified


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
