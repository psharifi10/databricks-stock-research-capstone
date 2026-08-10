"""Offline SQL-boundary tests for stock repositories."""

from datetime import date
import json
import unittest
from unittest.mock import MagicMock

from app.repositories import DEFAULT_WATCHLIST_NAME, StockRepository
from app.validation import ValidationError
from pipelines.embeddings import EMBEDDING_DIMENSION


def _compact_sql(value: str) -> str:
    return " ".join(value.split())


class StockRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cursor = MagicMock()
        cursor_context = MagicMock()
        cursor_context.__enter__.return_value = self.cursor
        self.connection = MagicMock()
        self.connection.cursor.return_value = cursor_context
        connection_context = MagicMock()
        connection_context.__enter__.return_value = self.connection
        self.connection_factory = MagicMock(return_value=connection_context)
        self.repository = StockRepository(self.connection_factory)

    def test_get_or_create_user_uses_normalized_email_and_upsert(self) -> None:
        self.cursor.fetchone.return_value = {
            "id": 7,
            "email": "user@example.com",
        }

        user = self.repository.get_or_create_user(" User@Example.com ")

        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("ON CONFLICT (email) DO UPDATE", _compact_sql(sql))
        self.assertEqual(parameters, ("user@example.com",))
        self.assertEqual(user["id"], 7)

    def test_get_or_create_default_watchlist_is_deterministic(self) -> None:
        self.cursor.fetchone.return_value = {
            "id": 11,
            "user_id": 7,
            "name": DEFAULT_WATCHLIST_NAME,
        }

        watchlist = self.repository.get_or_create_default_watchlist(7)

        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("ON CONFLICT (user_id, name) DO UPDATE", _compact_sql(sql))
        self.assertEqual(parameters, (7, DEFAULT_WATCHLIST_NAME))
        self.assertEqual(watchlist["name"], DEFAULT_WATCHLIST_NAME)

    def test_add_existing_watchlist_ticker_is_idempotent(self) -> None:
        self.cursor.fetchone.side_effect = [
            {"id": 11, "user_id": 7, "name": DEFAULT_WATCHLIST_NAME},
            None,
        ]

        inserted = self.repository.add_watchlist_ticker(7, " aapl ")

        self.assertFalse(inserted)
        sql, parameters = self.cursor.execute.call_args_list[1].args
        self.assertIn(
            "ON CONFLICT (watchlist_id, ticker) DO NOTHING",
            _compact_sql(sql),
        )
        self.assertEqual(parameters, (11, "AAPL"))

    def test_remove_missing_watchlist_ticker_returns_false(self) -> None:
        self.cursor.fetchone.return_value = None

        removed = self.repository.remove_watchlist_ticker(7, "aapl")

        self.assertFalse(removed)
        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("DELETE FROM watchlist_tickers", _compact_sql(sql))
        self.assertEqual(parameters, (7, DEFAULT_WATCHLIST_NAME, "AAPL"))

    def test_company_upsert_is_parameterized_and_preserves_raw_payload(self) -> None:
        company = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "legal_name": "Apple Inc.",
            "description": "Technology company",
            "market_cap": 3000000000000,
            "exchange": "XNAS",
            "raw_source_payload": {"ticker": "AAPL", "name": "Apple Inc."},
        }

        self.repository.upsert_company(company)

        sql, parameters = self.cursor.execute.call_args.args
        compact_sql = _compact_sql(sql)
        self.assertIn("INSERT INTO companies", compact_sql)
        self.assertIn("ON CONFLICT (ticker) DO UPDATE", compact_sql)
        self.assertEqual(parameters[0], "AAPL")
        self.assertEqual(json.loads(parameters[-1]), company["raw_source_payload"])
        self.assertNotIn("Apple Inc.", sql)

    def test_get_company_uses_parameterized_ticker_lookup(self) -> None:
        self.cursor.fetchone.return_value = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "description": "Technology company",
            "industry": "Technology Hardware",
            "market_cap": 3000000000000,
            "exchange": "XNAS",
        }

        company = self.repository.get_company(" aapl ")

        sql, parameters = self.cursor.execute.call_args.args
        compact_sql = _compact_sql(sql)
        self.assertIn("FROM companies", compact_sql)
        self.assertIn("WHERE ticker = %s", compact_sql)
        self.assertEqual(parameters, ("AAPL",))
        self.assertEqual(company["name"], "Apple Inc.")

    def test_get_company_returns_none_when_missing(self) -> None:
        self.cursor.fetchone.return_value = None

        company = self.repository.get_company("AAPL")

        self.assertIsNone(company)

    def test_price_upsert_uses_composite_conflict_key(self) -> None:
        rows = [
            {
                "price_date": date(2026, 8, 8),
                "open": 100,
                "high": 110,
                "low": 99,
                "close": 108,
                "volume": 1000,
                "vwap": 105,
            }
        ]

        count = self.repository.upsert_price_snapshots("aapl", rows)

        sql, parameters = self.cursor.executemany.call_args.args
        self.assertIn(
            "ON CONFLICT (ticker, price_date) DO UPDATE",
            _compact_sql(sql),
        )
        self.assertEqual(parameters[0][0], "AAPL")
        self.assertEqual(parameters[0][1], date(2026, 8, 8))
        self.assertEqual(count, 1)

    def test_recent_prices_select_latest_window_then_return_chronologically(self) -> None:
        self.cursor.fetchall.return_value = [
            {"ticker": "AAPL", "price_date": date(2026, 8, 7)},
            {"ticker": "AAPL", "price_date": date(2026, 8, 8)},
        ]

        rows = self.repository.list_recent_prices(" aapl ", 30)

        sql, parameters = self.cursor.execute.call_args.args
        compact_sql = _compact_sql(sql)
        self.assertIn("FROM price_snapshots", compact_sql)
        self.assertIn("ORDER BY price_date DESC LIMIT %s", compact_sql)
        self.assertTrue(compact_sql.endswith("ORDER BY price_date ASC"))
        self.assertEqual(parameters, ("AAPL", 30))
        self.assertEqual(rows[0]["price_date"], date(2026, 8, 7))

    def test_news_upsert_associates_tickers_with_individual_sentiment(self) -> None:
        article = {
            "id": "article-1",
            "title": "Two-company story",
            "description": None,
            "author": "Reporter",
            "publisher": "Example News",
            "article_url": "https://news.example.invalid/one",
            "published_at": None,
            "keywords": ["technology"],
            "ticker_insights": [
                {
                    "ticker": "AAPL",
                    "sentiment": "positive",
                    "sentiment_reasoning": "Strong demand.",
                },
                {
                    "ticker": "MSFT",
                    "sentiment": "negative",
                    "sentiment_reasoning": "Demand weakened.",
                },
                {
                    "ticker": "NVDA",
                    "sentiment": None,
                    "sentiment_reasoning": None,
                },
            ],
            "raw_payload": {"id": "article-1"},
        }

        count = self.repository.upsert_news_articles([article])

        self.assertEqual(count, 1)
        calls = self.cursor.execute.call_args_list
        article_sql = _compact_sql(calls[0].args[0])
        self.assertIn("INSERT INTO news_articles", article_sql)
        self.assertIn("ON CONFLICT (id) DO UPDATE", article_sql)
        self.assertNotIn("sentiment", article_sql)
        association_parameters = [call.args[1] for call in calls[1:]]
        self.assertEqual(
            association_parameters,
            [
                ("article-1", "AAPL", "positive", "Strong demand."),
                ("article-1", "MSFT", "negative", "Demand weakened."),
                ("article-1", "NVDA", None, None),
            ],
        )
        for call in calls[1:]:
            association_sql = _compact_sql(call.args[0])
            self.assertIn("ON CONFLICT (article_id, ticker) DO UPDATE", association_sql)
            self.assertIn("sentiment = EXCLUDED.sentiment", association_sql)

    def test_duplicate_article_and_updated_sentiment_use_upserts(self) -> None:
        article = {
            "id": "article-1",
            "title": "Reusable article",
            "keywords": [],
            "ticker_insights": [
                {
                    "ticker": "MSFT",
                    "sentiment": "negative",
                    "sentiment_reasoning": "Initial view.",
                }
            ],
            "raw_payload": {"id": "article-1"},
        }
        updated = {
            **article,
            "ticker_insights": [
                {
                    "ticker": "MSFT",
                    "sentiment": "neutral",
                    "sentiment_reasoning": "Updated view.",
                }
            ],
        }

        self.repository.upsert_news_articles([article])
        self.repository.upsert_news_articles([updated])

        calls = self.cursor.execute.call_args_list
        article_calls = [
            call for call in calls if "INSERT INTO news_articles" in call.args[0]
        ]
        self.assertEqual(len(article_calls), 2)
        self.assertTrue(
            all("ON CONFLICT (id) DO UPDATE" in call.args[0] for call in article_calls)
        )
        self.assertEqual(
            calls[-1].args[1],
            ("article-1", "MSFT", "neutral", "Updated view."),
        )
        self.assertIn(
            "sentiment_reasoning = EXCLUDED.sentiment_reasoning",
            _compact_sql(calls[-1].args[0]),
        )

    def test_list_recent_news_selects_requested_ticker_sentiment(self) -> None:
        self.cursor.fetchall.return_value = [
            {
                "id": "article-1",
                "ticker": "MSFT",
                "sentiment": "negative",
                "sentiment_reasoning": "Demand weakened.",
            }
        ]

        rows = self.repository.list_recent_news("msft", 5)

        sql, parameters = self.cursor.execute.call_args.args
        compact_sql = _compact_sql(sql)
        self.assertIn("requested.sentiment", compact_sql)
        self.assertIn("requested.sentiment_reasoning", compact_sql)
        self.assertNotIn("na.sentiment", compact_sql)
        self.assertEqual(parameters, ("MSFT", 5))
        self.assertEqual(rows[0]["sentiment"], "negative")

    def test_replace_news_chunks_deletes_stale_rows_before_inserting(self) -> None:
        count = self.repository.replace_news_article_chunks(
            " article-1 ",
            ["First chunk", "Second chunk"],
        )

        delete_sql, delete_parameters = self.cursor.execute.call_args.args
        self.assertEqual(
            _compact_sql(delete_sql),
            "DELETE FROM news_article_chunks WHERE article_id = %s",
        )
        self.assertEqual(delete_parameters, ("article-1",))
        insert_sql, insert_parameters = self.cursor.executemany.call_args.args
        self.assertIn("INSERT INTO news_article_chunks", _compact_sql(insert_sql))
        self.assertEqual(
            [call[0] for call in self.cursor.method_calls],
            ["execute", "executemany"],
        )
        self.assertEqual(
            insert_parameters,
            [
                ("article-1", 0, "First chunk"),
                ("article-1", 1, "Second chunk"),
            ],
        )
        self.assertEqual(count, 2)

    def test_empty_chunk_replacement_only_deletes_stale_rows(self) -> None:
        count = self.repository.replace_news_article_chunks("article-1", [])

        delete_sql, delete_parameters = self.cursor.execute.call_args.args
        self.assertIn("DELETE FROM news_article_chunks", _compact_sql(delete_sql))
        self.assertEqual(delete_parameters, ("article-1",))
        self.cursor.executemany.assert_not_called()
        self.assertEqual(count, 0)

    def test_semantic_search_uses_parameterized_cosine_query(self) -> None:
        self.cursor.fetchall.return_value = [
            {
                "article_id": "article-1",
                "chunk_index": 0,
                "similarity": 0.88,
            }
        ]
        query_vector = [0.125] * EMBEDDING_DIMENSION

        rows = self.repository.search_news_chunks(query_vector, top_k=100)

        sql, parameters = self.cursor.execute.call_args.args
        compact_sql = _compact_sql(sql)
        self.assertIn("embedding <=> query_vector.embedding", compact_sql)
        self.assertIn("1 - (chunk.embedding <=> query_vector.embedding)", compact_sql)
        self.assertIn("chunk.embedding IS NOT NULL", compact_sql)
        self.assertIn("JOIN news_articles", compact_sql)
        self.assertNotIn(parameters[0], sql)
        self.assertEqual(parameters[1], 20)
        self.assertEqual(rows[0]["similarity"], 0.88)

    def test_ticker_semantic_search_joins_normalized_association(self) -> None:
        self.cursor.fetchall.return_value = []

        self.repository.search_news_chunks(
            [0.0] * EMBEDDING_DIMENSION,
            ticker=" aapl ",
            top_k=3,
        )

        sql, parameters = self.cursor.execute.call_args.args
        compact_sql = _compact_sql(sql)
        self.assertIn("JOIN news_article_tickers AS requested", compact_sql)
        self.assertIn("requested.sentiment", compact_sql)
        self.assertIn("requested.ticker = %s", compact_sql)
        self.assertEqual(parameters[1:], ("AAPL", 3))
        self.assertNotIn("AAPL", sql)

    def test_semantic_search_rejects_invalid_top_k(self) -> None:
        with self.assertRaises(ValidationError):
            self.repository.search_news_chunks(
                [0.0] * EMBEDDING_DIMENSION,
                top_k=0,
            )

        self.connection_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
