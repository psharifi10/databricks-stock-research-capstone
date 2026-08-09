"""Tests for framework-independent service orchestration."""

from datetime import date
import unittest
from unittest.mock import MagicMock

from app.services import StockResearchService


class StockResearchServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        self.repository = MagicMock()
        self.service = StockResearchService(self.client, self.repository)

    def test_refresh_company_fetches_persists_and_returns(self) -> None:
        company = {"ticker": "AAPL", "name": "Apple Inc."}
        self.client.get_company_overview.return_value = company

        result = self.service.refresh_company(" aapl ")

        self.client.get_company_overview.assert_called_once_with("AAPL")
        self.repository.upsert_company.assert_called_once_with(company)
        self.assertIs(result, company)

    def test_refresh_price_history_fetches_persists_and_returns(self) -> None:
        rows = [{"ticker": "AAPL", "price_date": date(2026, 8, 8)}]
        self.client.get_historical_prices.return_value = rows

        result = self.service.refresh_price_history(
            "aapl",
            "2026-08-01",
            "2026-08-08",
        )

        self.client.get_historical_prices.assert_called_once_with(
            "AAPL",
            "2026-08-01",
            "2026-08-08",
        )
        self.repository.upsert_price_snapshots.assert_called_once_with(
            "AAPL",
            rows,
        )
        self.assertIs(result, rows)

    def test_refresh_news_persists_articles_and_ticker_relationships(self) -> None:
        articles = [
            {
                "id": "article-1",
                "ticker_insights": [
                    {"ticker": "AAPL", "sentiment": None},
                    {"ticker": "MSFT", "sentiment": "neutral"},
                ],
            }
        ]
        self.client.get_news.return_value = articles

        result = self.service.refresh_news("aapl", limit=5)

        self.client.get_news.assert_called_once_with(
            "AAPL",
            limit=5,
            published_after=None,
        )
        self.repository.upsert_news_articles.assert_called_once_with(articles)
        self.assertIs(result, articles)


if __name__ == "__main__":
    unittest.main()
