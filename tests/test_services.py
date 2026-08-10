"""Tests for framework-independent service orchestration."""

from datetime import date
import unittest
from unittest.mock import MagicMock

from app.services import SemanticNewsSearchService, StockResearchService
from pipelines.embeddings import EMBEDDING_DIMENSION, EmbeddingError


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
            max_pages=None,
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
            max_pages=None,
        )
        self.repository.upsert_news_articles.assert_called_once_with(articles)
        self.assertIs(result, articles)


class SemanticNewsSearchServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = MagicMock()
        self.embedding_service = MagicMock()
        self.embedding_service.embed_query.return_value = (
            [0.25] * EMBEDDING_DIMENSION
        )
        self.repository.search_news_chunks.return_value = [
            {"article_id": "article-1", "similarity": 0.9}
        ]
        self.service = SemanticNewsSearchService(
            self.repository,
            self.embedding_service,
        )

    def test_query_embedding_is_passed_to_bounded_ticker_search(self) -> None:
        results = self.service.semantic_news_search(
            "  Apple CEO succession  ",
            ticker=" aapl ",
            top_k=100,
        )

        self.embedding_service.embed_query.assert_called_once_with(
            "Apple CEO succession"
        )
        args, kwargs = self.repository.search_news_chunks.call_args
        self.assertEqual(len(args[0]), EMBEDDING_DIMENSION)
        self.assertEqual(kwargs, {"ticker": "AAPL", "top_k": 20})
        self.assertEqual(results[0]["similarity"], 0.9)

    def test_blank_query_is_rejected_before_embedding(self) -> None:
        with self.assertRaises(EmbeddingError):
            self.service.semantic_news_search("   ")

        self.embedding_service.embed_query.assert_not_called()
        self.repository.search_news_chunks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
