"""Offline contracts for the Phase 6A FastMCP server."""

from datetime import date
from decimal import Decimal
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import yaml

import mcp_server.stock_research_mcp as server


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = PROJECT_ROOT / "mcp_server" / "stock_research_mcp.py"


class McpRegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_read_and_write_tools_have_meaningful_docstrings(self) -> None:
        tools = await server.mcp.list_tools()

        self.assertEqual(
            {tool.name for tool in tools},
            {
                "get_company",
                "get_price_history",
                "search_financial_news",
                "build_research_context",
                "health",
                "add_to_watchlist",
                "remove_from_watchlist",
                "save_research_note",
                "save_analysis_report",
            },
        )
        for tool in tools:
            with self.subTest(tool=tool.name):
                self.assertGreaterEqual(len((tool.description or "").split()), 8)
                if tool.name in {
                    "add_to_watchlist",
                    "remove_from_watchlist",
                    "save_research_note",
                    "save_analysis_report",
                }:
                    self.assertIn("explicitly", tool.description.lower())


class McpToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = MagicMock()
        self.semantic_search = MagicMock()
        self.research_context = MagicMock()
        self.actions = MagicMock()
        self.services = server.ResearchServices(
            repository=self.repository,
            semantic_search=self.semantic_search,
            research_context=self.research_context,
            actions=self.actions,
        )
        self.services_patcher = patch.object(
            server,
            "get_services",
            return_value=self.services,
        )
        self.get_services = self.services_patcher.start()

    def tearDown(self) -> None:
        self.services_patcher.stop()

    def test_health_returns_status_without_composing_services(self) -> None:
        result = server.health()

        self.assertEqual(
            result,
            {
                "ok": True,
                "data": {
                    "status": "ok",
                    "service": "stock-research-mcp",
                },
            },
        )
        self.get_services.assert_not_called()

    def test_company_tool_delegates_and_handles_missing_data(self) -> None:
        self.repository.get_company.return_value = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "market_cap": Decimal("3000000000000.25"),
        }

        result = server.get_company(" aapl ")

        self.repository.get_company.assert_called_once_with("AAPL")
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["market_cap"], "3000000000000.25")

        self.repository.get_company.reset_mock()
        self.repository.get_company.return_value = None
        missing = server.get_company("MSFT")

        self.assertFalse(missing["ok"])
        self.assertEqual(missing["error"]["type"], "not_found")
        self.assertIn("MSFT", missing["error"]["message"])

    def test_price_tool_delegates_chronological_rows_and_enforces_bounds(self) -> None:
        self.repository.list_recent_prices.return_value = [
            {"price_date": date(2026, 8, 7), "close": Decimal("220.25")},
            {"price_date": date(2026, 8, 8), "close": Decimal("221.10")},
        ]

        result = server.get_price_history("aapl", 30)

        self.repository.list_recent_prices.assert_called_once_with("AAPL", 30)
        self.assertEqual(
            [row["price_date"] for row in result["data"]],
            ["2026-08-07", "2026-08-08"],
        )

        for invalid_limit in (0, 91, True):
            with self.subTest(limit=invalid_limit):
                invalid = server.get_price_history("AAPL", invalid_limit)
                self.assertFalse(invalid["ok"])
                self.assertEqual(invalid["error"]["type"], "validation_error")

    def test_semantic_tool_delegates_and_never_returns_vectors(self) -> None:
        self.semantic_search.semantic_news_search.return_value = [
            {
                "article_id": "article-1",
                "chunk_index": 0,
                "title": "Leadership update",
                "article_url": "https://news.example.invalid/article-1",
                "publisher_name": "Example News",
                "published_at": "2026-08-08T12:00:00+00:00",
                "sentiment": "neutral",
                "sentiment_reasoning": "Mixed evidence.",
                "similarity": 0.91,
                "chunk_text": "Grounded evidence.",
                "embedding": [0.1, 0.2, 0.3],
            }
        ]

        result = server.search_financial_news(
            " aapl ",
            "  leadership outlook  ",
            5,
        )

        self.semantic_search.semantic_news_search.assert_called_once_with(
            "leadership outlook",
            ticker="AAPL",
            top_k=5,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"][0]["publisher"], "Example News")
        self.assertNotIn("embedding", result["data"][0])
        self.assertNotIn("vector", str(result).lower())

        invalid = server.search_financial_news("AAPL", "outlook", 11)
        self.assertFalse(invalid["ok"])
        self.assertEqual(invalid["error"]["type"], "validation_error")

    def test_research_context_delegates_defaults_and_strips_vector_keys(self) -> None:
        self.research_context.build_research_context.return_value = {
            "ticker": "AAPL",
            "question": "What changed?",
            "company": {"ticker": "AAPL"},
            "prices": [],
            "recent_news": [],
            "semantic_evidence": [
                {
                    "article_id": "article-1",
                    "embedding": [0.1, 0.2],
                    "chunk_text": "Evidence.",
                }
            ],
        }

        result = server.build_research_context("AAPL", "What changed?")

        self.research_context.build_research_context.assert_called_once_with(
            "AAPL",
            "What changed?",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            set(result["data"]),
            {
                "ticker",
                "question",
                "company",
                "prices",
                "recent_news",
                "semantic_evidence",
            },
        )
        self.assertNotIn("embedding", result["data"]["semantic_evidence"][0])

    def test_validation_and_unexpected_errors_are_safe(self) -> None:
        validation = server.get_company("../AAPL")

        self.assertFalse(validation["ok"])
        self.assertEqual(validation["error"]["type"], "validation_error")

        self.repository.get_company.side_effect = RuntimeError(
            "password=secret database details"
        )
        failure = server.get_company("AAPL")

        self.assertEqual(
            failure,
            {
                "ok": False,
                "error": {
                    "type": "service_error",
                    "message": (
                        "The stock research service could not complete the request."
                    ),
                },
            },
        )
        self.assertNotIn("password", str(failure).lower())
        self.assertNotIn("secret", str(failure).lower())

    def test_watchlist_write_tools_delegate_without_retrieval(self) -> None:
        self.actions.add_to_watchlist.return_value = {
            "user_email": "user@example.com",
            "ticker": "AAPL",
            "added": False,
            "already_present": True,
        }
        self.actions.remove_from_watchlist.return_value = {
            "user_email": "user@example.com",
            "ticker": "AAPL",
            "removed": True,
        }

        added = server.add_to_watchlist("user@example.com", "AAPL")
        removed = server.remove_from_watchlist("user@example.com", "AAPL")

        self.actions.add_to_watchlist.assert_called_once_with(
            "user@example.com",
            "AAPL",
        )
        self.actions.remove_from_watchlist.assert_called_once_with(
            "user@example.com",
            "AAPL",
        )
        self.assertTrue(added["data"]["already_present"])
        self.assertTrue(removed["data"]["removed"])
        self.repository.get_company.assert_not_called()
        self.semantic_search.semantic_news_search.assert_not_called()

    def test_note_and_report_write_tools_delegate(self) -> None:
        self.actions.save_research_note.return_value = {
            "note_id": 31,
            "ticker": "AAPL",
            "saved": True,
        }
        self.actions.save_analysis_report.return_value = {
            "report_id": 41,
            "ticker": "AAPL",
            "title": "Outlook",
            "saved": True,
        }

        note = server.save_research_note(
            "user@example.com",
            "AAPL",
            "Research note",
        )
        report = server.save_analysis_report(
            "user@example.com",
            "AAPL",
            "Outlook",
            "Completed report",
        )

        self.actions.save_research_note.assert_called_once_with(
            "user@example.com",
            "AAPL",
            "Research note",
        )
        self.actions.save_analysis_report.assert_called_once_with(
            "user@example.com",
            "AAPL",
            "Outlook",
            "Completed report",
        )
        self.assertEqual(note["data"]["note_id"], 31)
        self.assertEqual(report["data"]["report_id"], 41)

    def test_write_tool_validation_and_unexpected_errors_are_safe(self) -> None:
        self.actions.save_research_note.side_effect = server.ValidationError(
            "Research note cannot be blank."
        )

        invalid = server.save_research_note(
            "user@example.com",
            "AAPL",
            "",
        )

        self.assertFalse(invalid["ok"])
        self.assertEqual(invalid["error"]["type"], "validation_error")

        self.actions.save_research_note.side_effect = RuntimeError(
            "private internal database detail"
        )
        failure = server.save_research_note(
            "user@example.com",
            "AAPL",
            "Note",
        )

        self.assertEqual(failure["error"]["type"], "service_error")
        self.assertNotIn("private", str(failure).lower())
        self.assertNotIn("database detail", str(failure).lower())


class McpSourceAndDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SERVER_PATH.read_text(encoding="utf-8")

    def test_import_composes_no_services_or_database_connection(self) -> None:
        server.get_services.cache_clear()
        server.get_agent_client.cache_clear()

        self.assertEqual(server.get_services.cache_info().currsize, 0)
        self.assertEqual(server.get_agent_client.cache_info().currsize, 0)
        self.assertNotIn("database_connection(", self.source)

    def test_server_has_no_massive_llm_or_trading_behavior(self) -> None:
        for forbidden in (
            "MassiveClient",
            "OpenAI",
            "Anthropic",
            "chat.completions",
            "buy_stock",
            "sell_stock",
            "execute_trade",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_http_run_configuration_and_app_entrypoint(self) -> None:
        self.assertIn('transport="http"', self.source)
        self.assertIn('host="0.0.0.0"', self.source)
        self.assertIn("port=8000", self.source)
        self.assertIn('path="/mcp"', self.source)
        self.assertIn("stateless_http=True", self.source)
        self.assertNotIn("from flask", self.source.lower())
        self.assertNotIn("from fastapi", self.source.lower())

        app_config = yaml.safe_load(
            (PROJECT_ROOT / "app.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            app_config,
            {
                "command": [
                    "python",
                    "mcp_server/stock_research_mcp.py",
                ],
                "env": [
                    {
                        "name": "ENDPOINT_NAME",
                        "valueFrom": "postgres",
                    },
                    {
                        "name": "SUPERVISOR_ENDPOINT",
                        "valueFrom": "supervisor-agent",
                    },
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
