"""Offline tests for the bounded Phase 3C validation scripts."""

from datetime import date, datetime, timezone
import unittest
from unittest.mock import MagicMock

from scripts.refresh_stock_data import refresh_ticker, recent_completed_price_range
from scripts.verify_stock_data import get_stock_summary, validation_errors


class RefreshStockDataTests(unittest.TestCase):
    def test_recent_range_ends_on_last_completed_weekday(self) -> None:
        start_date, end_date = recent_completed_price_range(date(2026, 8, 9))

        self.assertEqual(end_date, date(2026, 8, 7))
        self.assertEqual(start_date, date(2026, 7, 28))

    def test_refresh_delegates_exactly_three_bounded_operations(self) -> None:
        service = MagicMock()
        service.refresh_company.return_value = {
            "ticker": "AAPL",
            "name": "Example Company",
        }
        service.refresh_price_history.return_value = [{"price_date": "2026-08-07"}]
        service.refresh_news.return_value = [{"id": "article-1"}]

        result = refresh_ticker(service, " aapl ", as_of=date(2026, 8, 9))

        service.refresh_company.assert_called_once_with("AAPL")
        service.refresh_price_history.assert_called_once_with(
            "AAPL",
            date(2026, 7, 28),
            date(2026, 8, 7),
            max_pages=1,
        )
        service.refresh_news.assert_called_once_with(
            "AAPL",
            limit=5,
            published_after=date(2026, 7, 28),
            max_pages=1,
        )
        self.assertEqual(result["ticker"], "AAPL")


class VerifyStockDataTests(unittest.TestCase):
    def test_summary_uses_parameterized_queries_and_preserves_relationships(self) -> None:
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.fetchall.side_effect = [
            [
                {
                    "ticker": "AAPL",
                    "name": "Example Company",
                    "exchange": "XNAS",
                    "updated_at": datetime(2026, 8, 9, tzinfo=timezone.utc),
                }
            ],
            [
                {
                    "title": "Example article",
                    "published_at": datetime(2026, 8, 8, tzinfo=timezone.utc),
                    "sentiment": "positive",
                    "related_tickers": ["AAPL", "MSFT"],
                }
            ],
        ]
        cursor.fetchone.side_effect = [
            {
                "row_count": 4,
                "minimum_price_date": date(2026, 8, 4),
                "maximum_price_date": date(2026, 8, 7),
            },
            {
                "association_count": 1,
                "distinct_article_count": 1,
                "newest_published_at": datetime(2026, 8, 8, tzinfo=timezone.utc),
            },
        ]
        connection = MagicMock()
        connection.cursor.return_value = cursor

        summary = get_stock_summary(connection, " aapl ")

        self.assertEqual(summary["company_count"], 1)
        self.assertEqual(summary["price_count"], 4)
        self.assertEqual(summary["distinct_news_articles"], 1)
        self.assertEqual(summary["news_associations"], 1)
        self.assertEqual(summary["articles"][0]["related_tickers"], ["AAPL", "MSFT"])
        self.assertEqual(validation_errors(summary), [])
        self.assertEqual(cursor.execute.call_count, 4)
        for call in cursor.execute.call_args_list[:3]:
            self.assertEqual(call.args[1], ("AAPL",))
        self.assertEqual(cursor.execute.call_args_list[3].args[1], ("AAPL", 3))

    def test_validation_reports_missing_required_data(self) -> None:
        summary = {
            "company_count": 0,
            "price_count": 0,
            "distinct_news_articles": 0,
            "news_associations": 0,
        }

        errors = validation_errors(summary)

        self.assertEqual(len(errors), 3)


if __name__ == "__main__":
    unittest.main()
