"""Tests for framework-independent service orchestration."""

from datetime import date, datetime, timezone
from decimal import Decimal
import inspect
import json
import unittest
from unittest.mock import MagicMock

from app.services import (
    ResearchActionService,
    ResearchContextService,
    SemanticNewsSearchService,
    StockResearchService,
)
from app.validation import ValidationError
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


class ResearchActionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = MagicMock()
        self.repository.get_or_create_user.return_value = {
            "id": 7,
            "email": "user@example.com",
        }
        self.service = ResearchActionService(self.repository)

    def test_watchlist_add_resolves_user_and_reports_idempotency(self) -> None:
        self.repository.add_watchlist_ticker.side_effect = [True, False]

        added = self.service.add_to_watchlist(" User@Example.com ", " aapl ")
        existing = self.service.add_to_watchlist("user@example.com", "AAPL")

        self.assertEqual(
            added,
            {
                "user_email": "user@example.com",
                "ticker": "AAPL",
                "added": True,
                "already_present": False,
            },
        )
        self.assertFalse(existing["added"])
        self.assertTrue(existing["already_present"])
        self.assertEqual(self.repository.get_or_create_user.call_count, 2)
        self.repository.add_watchlist_ticker.assert_called_with(7, "AAPL")

    def test_watchlist_remove_resolves_user_and_delegates(self) -> None:
        self.repository.remove_watchlist_ticker.return_value = True

        result = self.service.remove_from_watchlist(
            " User@Example.com ",
            " msft ",
        )

        self.repository.get_or_create_user.assert_called_once_with(
            "user@example.com"
        )
        self.repository.remove_watchlist_ticker.assert_called_once_with(
            7,
            "MSFT",
        )
        self.assertTrue(result["removed"])

    def test_note_validation_precedes_user_resolution_and_save_delegates(self) -> None:
        with self.assertRaises(ValidationError):
            self.service.save_research_note(
                "user@example.com",
                "AAPL",
                "   ",
            )
        with self.assertRaises(ValidationError):
            self.service.save_research_note(
                "user@example.com",
                "AAPL",
                "x" * 5001,
            )
        self.repository.get_or_create_user.assert_not_called()

        self.repository.save_research_note.return_value = {
            "id": 31,
            "created_at": "created",
        }
        result = self.service.save_research_note(
            " User@Example.com ",
            " aapl ",
            "  Evidence-based note.  ",
        )

        self.repository.save_research_note.assert_called_once_with(
            7,
            "AAPL",
            "Evidence-based note.",
        )
        self.assertEqual(result["note_id"], 31)
        self.assertTrue(result["saved"])
        self.assertNotIn("note_text", result)

    def test_report_validation_precedes_user_resolution_and_save_delegates(self) -> None:
        invalid_values = (
            ("", "Report body"),
            ("x" * 201, "Report body"),
            ("Title", "   "),
            ("Title", "x" * 20001),
        )
        for title, body in invalid_values:
            with self.subTest(title_length=len(title), body_length=len(body)):
                with self.assertRaises(ValidationError):
                    self.service.save_analysis_report(
                        "user@example.com",
                        "AAPL",
                        title,
                        body,
                    )
        self.repository.get_or_create_user.assert_not_called()

        self.repository.save_analysis_report.return_value = {
            "id": 41,
            "created_at": "created",
        }
        result = self.service.save_analysis_report(
            " User@Example.com ",
            " aapl ",
            "  Quarterly outlook  ",
            "  Grounded report body.  ",
        )

        self.repository.save_analysis_report.assert_called_once_with(
            7,
            "AAPL",
            "Quarterly outlook",
            "Grounded report body.",
        )
        self.assertEqual(result["report_id"], 41)
        self.assertEqual(result["title"], "Quarterly outlook")
        self.assertNotIn("report_text", result)


class ResearchContextServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = MagicMock()
        self.semantic_search = MagicMock()
        self.repository.get_company.return_value = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "description": "Technology company",
            "industry": "Technology Hardware",
            "market_cap": Decimal("3000000000000.25"),
            "exchange": "XNAS",
        }
        self.repository.list_recent_prices.return_value = [
            {
                "ticker": "AAPL",
                "price_date": date(2026, 8, 7),
                "close": Decimal("220.25"),
            }
        ]
        self.repository.list_recent_news.return_value = [
            {
                "id": "recent-1",
                "title": "Recent development",
                "publisher": "Example News",
                "published_at": datetime(
                    2026,
                    8,
                    8,
                    12,
                    0,
                    tzinfo=timezone.utc,
                ),
                "sentiment": "positive",
                "sentiment_reasoning": "Demand improved.",
                "article_url": "https://news.example.invalid/recent-1",
            }
        ]
        self.semantic_search.semantic_news_search.return_value = []
        self.service = ResearchContextService(
            self.repository,
            self.semantic_search,
        )

    def test_ticker_and_question_are_validated_before_reads(self) -> None:
        with self.assertRaises(ValidationError):
            self.service.build_research_context("../AAPL", "Outlook?")
        with self.assertRaises(EmbeddingError):
            self.service.build_research_context("AAPL", "   ")

        self.repository.get_company.assert_not_called()
        self.semantic_search.semantic_news_search.assert_not_called()

    def test_context_delegates_bounded_reads_and_semantic_candidate_search(self) -> None:
        context = self.service.build_research_context(
            " aapl ",
            "  What changed?  ",
            semantic_top_k=8,
            recent_news_limit=5,
            price_history_limit=30,
        )

        self.repository.get_company.assert_called_once_with("AAPL")
        self.repository.list_recent_prices.assert_called_once_with("AAPL", 30)
        self.repository.list_recent_news.assert_called_once_with("AAPL", 5)
        self.semantic_search.semantic_news_search.assert_called_once_with(
            "What changed?",
            ticker="AAPL",
            top_k=16,
        )
        self.assertEqual(context["ticker"], "AAPL")
        self.assertEqual(context["question"], "What changed?")
        self.assertEqual(context["company"]["exchange"], "XNAS")
        self.assertEqual(context["recent_news"][0]["article_id"], "recent-1")

    def test_context_limits_are_validated_before_reads(self) -> None:
        invalid_options = (
            {"semantic_top_k": 0},
            {"recent_news_limit": 0},
            {"recent_news_limit": 101},
            {"price_history_limit": False},
            {"price_history_limit": 101},
        )

        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValidationError):
                    self.service.build_research_context(
                        "AAPL",
                        "What changed?",
                        **options,
                    )

        self.repository.get_company.assert_not_called()

    def test_semantic_evidence_is_diversified_without_reordering(self) -> None:
        self.semantic_search.semantic_news_search.return_value = [
            _semantic_row("article-a", 0, 0.99),
            _semantic_row("article-a", 1, 0.98),
            _semantic_row("article-a", 2, 0.97),
            _semantic_row("article-b", 0, 0.96),
            _semantic_row("article-c", 0, 0.95),
        ]

        context = self.service.build_research_context(
            "AAPL",
            "What changed?",
            semantic_top_k=4,
        )

        evidence = context["semantic_evidence"]
        self.assertEqual(
            [(row["article_id"], row["chunk_index"]) for row in evidence],
            [
                ("article-a", 0),
                ("article-a", 1),
                ("article-b", 0),
                ("article-c", 0),
            ],
        )
        self.assertEqual(
            [row["similarity"] for row in evidence],
            [0.99, 0.98, 0.96, 0.95],
        )
        self.assertEqual(
            set(evidence[0]),
            {
                "article_id",
                "chunk_index",
                "chunk_text",
                "title",
                "publisher",
                "published_at",
                "article_url",
                "ticker",
                "sentiment",
                "sentiment_reasoning",
                "similarity",
            },
        )
        self.semantic_search.semantic_news_search.assert_called_once_with(
            "What changed?",
            ticker="AAPL",
            top_k=8,
        )

    def test_missing_company_and_json_serialization_are_safe(self) -> None:
        self.repository.get_company.return_value = None
        self.semantic_search.semantic_news_search.return_value = [
            {
                **_semantic_row("article-a", 0, Decimal("0.875")),
                "embedding": [0.1, 0.2, 0.3],
            }
        ]

        context = self.service.build_research_context("AAPL", "Outlook?")

        self.assertIsNone(context["company"])
        self.assertEqual(context["prices"][0]["price_date"], "2026-08-07")
        self.assertEqual(context["prices"][0]["close"], "220.25")
        self.assertEqual(
            context["recent_news"][0]["published_at"],
            "2026-08-08T12:00:00+00:00",
        )
        self.assertNotIn("embedding", context["semantic_evidence"][0])
        json.dumps(context)

    def test_context_has_no_massive_or_llm_dependency(self) -> None:
        source = inspect.getsource(ResearchContextService)

        self.assertNotIn("_massive_client", source)
        self.assertNotIn("OpenAI", source)
        self.assertNotIn("chat.completions", source)


def _semantic_row(
    article_id: str,
    chunk_index: int,
    similarity: float | Decimal,
) -> dict[str, object]:
    return {
        "article_id": article_id,
        "chunk_index": chunk_index,
        "chunk_text": f"Evidence for {article_id} chunk {chunk_index}",
        "title": f"Title for {article_id}",
        "publisher_name": "Example News",
        "published_at": datetime(2026, 8, 8, tzinfo=timezone.utc),
        "article_url": f"https://news.example.invalid/{article_id}",
        "ticker": "AAPL",
        "sentiment": "neutral",
        "sentiment_reasoning": "Mixed evidence.",
        "similarity": similarity,
    }


if __name__ == "__main__":
    unittest.main()
