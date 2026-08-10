"""Offline contracts for the FastMCP-hosted stock research frontend."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import warnings

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=StarletteDeprecationWarning,
)
from starlette.testclient import TestClient

import mcp_server.stock_research_mcp as server
from app.validation import ValidationError
from pipelines.embeddings import EmbeddingError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "mcp_server" / "static"


class FrontendRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.research_context = MagicMock()
        self.services = MagicMock()
        self.services.research_context = self.research_context
        self.services_patcher = patch.object(
            server,
            "get_services",
            return_value=self.services,
        )
        self.get_services = self.services_patcher.start()
        app = server.mcp.http_app(
            path="/mcp",
            stateless_http=True,
            host_origin_protection=False,
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.services_patcher.stop()

    def test_homepage_and_static_assets_are_served_without_services(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("AI Stock Research Assistant", response.text)
        self.assertIn('id="research-form"', response.text)
        self.assertIn('value="AAPL"', response.text)
        self.assertIn("What developments could affect", response.text)
        self.assertEqual(self.client.get("/static/styles.css").status_code, 200)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.get_services.assert_not_called()

    def test_frontend_api_and_mcp_routes_coexist(self) -> None:
        routes = {
            route.path: set(route.methods or [])
            for route in self.client.app.routes
        }

        self.assertIn("GET", routes["/"])
        self.assertEqual(routes["/api/research"], {"POST"})
        self.assertIn("POST", routes["/mcp"])

    def test_research_api_delegates_and_strips_vectors(self) -> None:
        self.research_context.build_research_context.return_value = {
            "ticker": "AAPL",
            "question": "What changed?",
            "company": {"ticker": "AAPL", "name": "Apple Inc."},
            "prices": [],
            "recent_news": [],
            "semantic_evidence": [
                {
                    "article_id": "article-1",
                    "chunk_text": "Grounded evidence.",
                    "embedding": [0.1, 0.2],
                    "vector": [0.1, 0.2],
                }
            ],
        }

        response = self.client.post(
            "/api/research",
            json={"ticker": " aapl ", "question": "  What changed?  "},
        )

        self.assertEqual(response.status_code, 200)
        self.research_context.build_research_context.assert_called_once_with(
            " aapl ",
            "  What changed?  ",
        )
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["ticker"], "AAPL")
        self.assertNotIn(
            "embedding",
            payload["data"]["semantic_evidence"][0],
        )
        self.assertNotIn("vector", payload["data"]["semantic_evidence"][0])

    def test_ticker_and_question_validation_return_400(self) -> None:
        failures = (
            ValidationError("Ticker must be a string."),
            EmbeddingError("Research question cannot be blank."),
        )
        requests = (
            {"question": "What changed?"},
            {"ticker": "AAPL", "question": "   "},
        )

        for error, body in zip(failures, requests, strict=True):
            with self.subTest(error=type(error).__name__):
                self.research_context.build_research_context.side_effect = error
                response = self.client.post("/api/research", json=body)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(
                    response.json()["error"]["type"],
                    "validation_error",
                )

    def test_invalid_json_and_non_object_body_return_400(self) -> None:
        malformed = self.client.post(
            "/api/research",
            content="{not-json",
            headers={"Content-Type": "application/json"},
        )
        non_object = self.client.post("/api/research", json=["AAPL"])

        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(non_object.status_code, 400)
        self.get_services.assert_not_called()

    def test_unexpected_failure_is_sanitized(self) -> None:
        self.research_context.build_research_context.side_effect = RuntimeError(
            "private internal database detail"
        )

        response = self.client.post(
            "/api/research",
            json={"ticker": "AAPL", "question": "What changed?"},
        )

        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["error"]["type"], "service_error")
        self.assertNotIn("private", str(payload).lower())
        self.assertNotIn("database detail", str(payload).lower())


class FrontendAssetSafetyTests(unittest.TestCase):
    def test_frontend_uses_safe_dom_and_external_link_attributes(self) -> None:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        combined = f"{html}\n{script}".lower()

        self.assertIn("textcontent", combined)
        self.assertNotIn("innerhtml", combined)
        self.assertIn('link.target = "_blank"', script)
        self.assertIn('link.rel = "noopener noreferrer"', script)
        self.assertIn('["http:", "https:"]', script)

    def test_frontend_assets_contain_no_credentials_or_vectors(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(STATIC_DIR.iterdir())
            if path.is_file()
        ).lower()

        for forbidden in (
            "pgpassword",
            "lakebase_url",
            "bearer credential",
            "client_secret",
            "api_key",
            "oauth token",
            "query_embedding",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
