"""Plain-SQL Lakebase repositories for stock research primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import date
import json
from typing import Any, ContextManager

import psycopg

from app.db import database_connection
from app.validation import (
    ValidationError,
    normalize_bounded_text,
    normalize_bounded_limit,
    normalize_email,
    normalize_ticker,
    normalize_top_k,
)
from pipelines.embeddings import serialize_embedding


DEFAULT_WATCHLIST_NAME = "My Watchlist"
ConnectionFactory = Callable[[], ContextManager[Any]]


class RepositoryError(RuntimeError):
    """A safe error raised when a persistence operation fails."""


class StockRepository:
    """Primitive, transaction-scoped operations over the Phase 1 schema."""

    def __init__(
        self,
        connection_factory: ConnectionFactory = database_connection,
    ) -> None:
        self._connection_factory = connection_factory

    def get_company(self, ticker: str) -> dict[str, Any] | None:
        symbol = normalize_ticker(ticker)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT ticker, name, description, industry,
                           market_cap, exchange
                    FROM companies
                    WHERE ticker = %s
                    """,
                    (symbol,),
                )
                row = cursor.fetchone()
                return dict(row) if row is not None else None

    def get_or_create_user(self, email: str) -> dict[str, Any]:
        normalized_email = normalize_email(email)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (email)
                    VALUES (%s)
                    ON CONFLICT (email) DO UPDATE
                    SET email = EXCLUDED.email
                    RETURNING id, email, created_at, updated_at
                    """,
                    (normalized_email,),
                )
                return _required_row(cursor.fetchone(), "user")

    def get_or_create_default_watchlist(self, user_id: int) -> dict[str, Any]:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                return self._get_or_create_default_watchlist(cursor, user_id)

    def list_watchlist_tickers(self, user_id: int) -> list[str]:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT wt.ticker
                    FROM watchlist_tickers AS wt
                    JOIN watchlists AS w ON w.id = wt.watchlist_id
                    WHERE w.user_id = %s AND w.name = %s
                    ORDER BY wt.ticker ASC
                    """,
                    (user_id, DEFAULT_WATCHLIST_NAME),
                )
                return [str(row["ticker"]) for row in cursor.fetchall()]

    def add_watchlist_ticker(self, user_id: int, ticker: str) -> bool:
        symbol = normalize_ticker(ticker)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                watchlist = self._get_or_create_default_watchlist(cursor, user_id)
                cursor.execute(
                    """
                    INSERT INTO watchlist_tickers (watchlist_id, ticker)
                    VALUES (%s, %s)
                    ON CONFLICT (watchlist_id, ticker) DO NOTHING
                    RETURNING ticker
                    """,
                    (watchlist["id"], symbol),
                )
                return cursor.fetchone() is not None

    def remove_watchlist_ticker(self, user_id: int, ticker: str) -> bool:
        symbol = normalize_ticker(ticker)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM watchlist_tickers AS wt
                    USING watchlists AS w
                    WHERE wt.watchlist_id = w.id
                      AND w.user_id = %s
                      AND w.name = %s
                      AND wt.ticker = %s
                    RETURNING wt.ticker
                    """,
                    (user_id, DEFAULT_WATCHLIST_NAME, symbol),
                )
                return cursor.fetchone() is not None

    def save_research_note(
        self,
        user_id: int,
        ticker: str,
        note_text: str,
    ) -> dict[str, Any]:
        symbol = normalize_ticker(ticker)
        normalized_note = normalize_bounded_text(
            note_text,
            field_name="Research note",
            maximum=5000,
        )
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO research_notes (user_id, ticker, note_text)
                    VALUES (%s, %s, %s)
                    RETURNING id, user_id, ticker, created_at, updated_at
                    """,
                    (user_id, symbol, normalized_note),
                )
                return _required_row(cursor.fetchone(), "research note")

    def save_analysis_report(
        self,
        user_id: int,
        ticker: str,
        title: str,
        report_text: str,
    ) -> dict[str, Any]:
        symbol = normalize_ticker(ticker)
        normalized_title = normalize_bounded_text(
            title,
            field_name="Analysis report title",
            maximum=200,
        )
        normalized_report = normalize_bounded_text(
            report_text,
            field_name="Analysis report body",
            maximum=20000,
        )
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO analysis_reports (
                        user_id, ticker, title, report_text
                    )
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, user_id, ticker, title,
                              created_at, updated_at
                    """,
                    (
                        user_id,
                        symbol,
                        normalized_title,
                        normalized_report,
                    ),
                )
                return _required_row(cursor.fetchone(), "analysis report")

    def upsert_company(self, company: Mapping[str, Any]) -> None:
        ticker = normalize_ticker(str(company.get("ticker") or ""))
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO companies (
                        ticker, name, legal_name, description, market_cap,
                        market, exchange, security_type, active, list_date,
                        sic_code, sic_description, industry, homepage_url,
                        currency_name, locale, raw_source_payload
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (ticker) DO UPDATE SET
                        name = EXCLUDED.name,
                        legal_name = EXCLUDED.legal_name,
                        description = EXCLUDED.description,
                        market_cap = EXCLUDED.market_cap,
                        market = EXCLUDED.market,
                        exchange = EXCLUDED.exchange,
                        security_type = EXCLUDED.security_type,
                        active = EXCLUDED.active,
                        list_date = EXCLUDED.list_date,
                        sic_code = EXCLUDED.sic_code,
                        sic_description = EXCLUDED.sic_description,
                        industry = EXCLUDED.industry,
                        homepage_url = EXCLUDED.homepage_url,
                        currency_name = EXCLUDED.currency_name,
                        locale = EXCLUDED.locale,
                        raw_source_payload = EXCLUDED.raw_source_payload,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        ticker,
                        company.get("name"),
                        company.get("legal_name"),
                        company.get("description"),
                        company.get("market_cap"),
                        company.get("market"),
                        company.get("exchange"),
                        company.get("security_type"),
                        company.get("active"),
                        company.get("list_date"),
                        company.get("sic_code"),
                        company.get("sic_description"),
                        company.get("industry"),
                        company.get("homepage_url"),
                        company.get("currency_name"),
                        company.get("locale"),
                        _json_payload(company.get("raw_source_payload")),
                    ),
                )

    def upsert_price_snapshots(
        self,
        ticker: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> int:
        symbol = normalize_ticker(ticker)
        if not rows:
            return 0

        parameters = [
            (
                symbol,
                row.get("price_date"),
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("volume"),
                row.get("vwap"),
            )
            for row in rows
        ]
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO price_snapshots (
                        ticker, price_date, open, high, low, close, volume, vwap
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, price_date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        vwap = EXCLUDED.vwap,
                        fetched_at = CURRENT_TIMESTAMP
                    """,
                    parameters,
                )
        return len(parameters)

    def get_price_history(
        self,
        ticker: str,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> list[dict[str, Any]]:
        symbol = normalize_ticker(ticker)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT ticker, price_date, open, high, low, close,
                           volume, vwap, fetched_at
                    FROM price_snapshots
                    WHERE ticker = %s
                      AND (%s::date IS NULL OR price_date >= %s::date)
                      AND (%s::date IS NULL OR price_date <= %s::date)
                    ORDER BY price_date ASC
                    """,
                    (symbol, start_date, start_date, end_date, end_date),
                )
                return [dict(row) for row in cursor.fetchall()]

    def list_recent_prices(
        self,
        ticker: str,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Return the latest bounded price window in chronological order."""

        symbol = normalize_ticker(ticker)
        bounded_limit = normalize_bounded_limit(
            limit,
            field_name="Price history limit",
            maximum=100,
        )
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT ticker, price_date, open, high, low, close,
                           volume, vwap
                    FROM (
                        SELECT ticker, price_date, open, high, low, close,
                               volume, vwap
                        FROM price_snapshots
                        WHERE ticker = %s
                        ORDER BY price_date DESC
                        LIMIT %s
                    ) AS recent_prices
                    ORDER BY price_date ASC
                    """,
                    (symbol, bounded_limit),
                )
                return [dict(row) for row in cursor.fetchall()]

    def upsert_news_articles(
        self,
        articles: Sequence[Mapping[str, Any]],
    ) -> int:
        if not articles:
            return 0

        with self._connection() as connection:
            with connection.cursor() as cursor:
                for article in articles:
                    article_id = str(article.get("id") or "").strip()
                    if not article_id:
                        raise RepositoryError("A news article ID is required.")
                    cursor.execute(
                        """
                        INSERT INTO news_articles (
                            id, title, description, author, publisher,
                            article_url, published_at, keywords,
                            raw_source_payload
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            author = EXCLUDED.author,
                            publisher = EXCLUDED.publisher,
                            article_url = EXCLUDED.article_url,
                            published_at = EXCLUDED.published_at,
                            keywords = EXCLUDED.keywords,
                            raw_source_payload = EXCLUDED.raw_source_payload,
                            synced_at = CURRENT_TIMESTAMP
                        """,
                        (
                            article_id,
                            article.get("title"),
                            article.get("description"),
                            article.get("author"),
                            article.get("publisher"),
                            article.get("article_url"),
                            article.get("published_at"),
                            list(article.get("keywords") or []),
                            _json_payload(article.get("raw_payload")),
                        ),
                    )
                    for relationship in _ticker_insights(article):
                        cursor.execute(
                            """
                            INSERT INTO news_article_tickers (
                                article_id, ticker, sentiment,
                                sentiment_reasoning
                            )
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (article_id, ticker) DO UPDATE SET
                                sentiment = EXCLUDED.sentiment,
                                sentiment_reasoning = EXCLUDED.sentiment_reasoning
                            """,
                            (
                                article_id,
                                relationship["ticker"],
                                relationship["sentiment"],
                                relationship["sentiment_reasoning"],
                            ),
                        )
        return len(articles)

    def list_recent_news(
        self,
        ticker: str,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        symbol = normalize_ticker(ticker)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValidationError("News limit must be an integer between 1 and 100.")

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        na.id, na.title, na.description, na.author,
                        na.publisher, na.article_url, na.published_at,
                        na.keywords, requested.ticker AS ticker,
                        requested.sentiment,
                        requested.sentiment_reasoning,
                        ARRAY(
                            SELECT related.ticker
                            FROM news_article_tickers AS related
                            WHERE related.article_id = na.id
                            ORDER BY related.ticker
                        ) AS tickers,
                        na.synced_at
                    FROM news_articles AS na
                    JOIN news_article_tickers AS requested
                      ON requested.article_id = na.id
                    WHERE requested.ticker = %s
                    ORDER BY na.published_at DESC NULLS LAST
                    LIMIT %s
                    """,
                    (symbol, limit),
                )
                return [dict(row) for row in cursor.fetchall()]

    def search_news_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        ticker: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return nearest embedded news chunks using cosine distance."""

        serialized_embedding = serialize_embedding(query_embedding)
        limit = normalize_top_k(top_k)
        symbol = normalize_ticker(ticker) if ticker is not None else None

        if symbol is None:
            statement = """
                WITH query_vector AS (
                    SELECT %s::VECTOR(384) AS embedding
                )
                SELECT
                    chunk.article_id,
                    chunk.chunk_index,
                    chunk.chunk_text,
                    chunk.embedding_model,
                    article.title,
                    article.article_url,
                    article.publisher AS publisher_name,
                    article.published_at,
                    NULL::TEXT AS ticker,
                    NULL::TEXT AS sentiment,
                    NULL::TEXT AS sentiment_reasoning,
                    ARRAY(
                        SELECT related.ticker
                        FROM news_article_tickers AS related
                        WHERE related.article_id = article.id
                        ORDER BY related.ticker
                    ) AS tickers,
                    chunk.embedding <=> query_vector.embedding AS distance,
                    1 - (chunk.embedding <=> query_vector.embedding) AS similarity
                FROM news_article_chunks AS chunk
                JOIN news_articles AS article ON article.id = chunk.article_id
                CROSS JOIN query_vector
                WHERE chunk.embedding IS NOT NULL
                ORDER BY chunk.embedding <=> query_vector.embedding ASC,
                         chunk.article_id ASC,
                         chunk.chunk_index ASC
                LIMIT %s
            """
            parameters = (serialized_embedding, limit)
        else:
            statement = """
                WITH query_vector AS (
                    SELECT %s::VECTOR(384) AS embedding
                )
                SELECT
                    chunk.article_id,
                    chunk.chunk_index,
                    chunk.chunk_text,
                    chunk.embedding_model,
                    article.title,
                    article.article_url,
                    article.publisher AS publisher_name,
                    article.published_at,
                    requested.ticker,
                    requested.sentiment,
                    requested.sentiment_reasoning,
                    ARRAY(
                        SELECT related.ticker
                        FROM news_article_tickers AS related
                        WHERE related.article_id = article.id
                        ORDER BY related.ticker
                    ) AS tickers,
                    chunk.embedding <=> query_vector.embedding AS distance,
                    1 - (chunk.embedding <=> query_vector.embedding) AS similarity
                FROM news_article_chunks AS chunk
                JOIN news_articles AS article ON article.id = chunk.article_id
                JOIN news_article_tickers AS requested
                  ON requested.article_id = article.id
                CROSS JOIN query_vector
                WHERE chunk.embedding IS NOT NULL
                  AND requested.ticker = %s
                ORDER BY chunk.embedding <=> query_vector.embedding ASC,
                         chunk.article_id ASC,
                         chunk.chunk_index ASC
                LIMIT %s
            """
            parameters = (serialized_embedding, symbol, limit)

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(statement, parameters)
                return [dict(row) for row in cursor.fetchall()]

    def replace_news_article_chunks(
        self,
        article_id: str,
        chunks: Sequence[str],
    ) -> int:
        """Atomically delete stale chunks and insert deterministic replacements."""

        normalized_article_id = str(article_id or "").strip()
        if not normalized_article_id:
            raise RepositoryError("A news article ID is required.")

        replacements: list[tuple[str, int, str]] = []
        for chunk_index, chunk in enumerate(chunks):
            if not isinstance(chunk, str) or not chunk.strip():
                raise RepositoryError("News article chunks cannot be blank.")
            replacements.append(
                (normalized_article_id, chunk_index, chunk.strip())
            )

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM news_article_chunks WHERE article_id = %s",
                    (normalized_article_id,),
                )
                if replacements:
                    cursor.executemany(
                        """
                        INSERT INTO news_article_chunks (
                            article_id, chunk_index, chunk_text
                        )
                        VALUES (%s, %s, %s)
                        """,
                        replacements,
                    )
        return len(replacements)

    def _get_or_create_default_watchlist(
        self,
        cursor: Any,
        user_id: int,
    ) -> dict[str, Any]:
        cursor.execute(
            """
            INSERT INTO watchlists (user_id, name)
            VALUES (%s, %s)
            ON CONFLICT (user_id, name) DO UPDATE
            SET name = EXCLUDED.name
            RETURNING id, user_id, name, created_at, updated_at
            """,
            (user_id, DEFAULT_WATCHLIST_NAME),
        )
        return _required_row(cursor.fetchone(), "watchlist")

    @contextmanager
    def _connection(self):
        try:
            with self._connection_factory() as connection:
                yield connection
        except psycopg.Error as error:
            raise RepositoryError("The database operation failed.") from error


def _required_row(row: Any, object_name: str) -> dict[str, Any]:
    if row is None:
        raise RepositoryError(f"The database did not return the expected {object_name}.")
    return dict(row)


def _json_payload(value: Any) -> str:
    if not isinstance(value, Mapping):
        value = {}
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _ticker_insights(article: Mapping[str, Any]) -> list[dict[str, Any]]:
    relationships: dict[str, dict[str, Any]] = {}
    for candidate in article.get("ticker_insights") or []:
        if not isinstance(candidate, Mapping):
            continue
        ticker = normalize_ticker(str(candidate.get("ticker") or ""))
        relationships[ticker] = {
            "ticker": ticker,
            "sentiment": _nullable_text(candidate.get("sentiment")),
            "sentiment_reasoning": _nullable_text(
                candidate.get("sentiment_reasoning")
            ),
        }
    return list(relationships.values())


def _nullable_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
