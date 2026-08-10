"""Framework-independent orchestration for stock research operations."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from app.massive_client import MassiveClient
from app.repositories import StockRepository
from app.validation import normalize_ticker, normalize_top_k
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
