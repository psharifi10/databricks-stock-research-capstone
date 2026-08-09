"""Offline SQL-boundary tests for stock repositories."""

from datetime import date
import json
import unittest
from unittest.mock import MagicMock

from app.repositories import DEFAULT_WATCHLIST_NAME, StockRepository


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


if __name__ == "__main__":
    unittest.main()
