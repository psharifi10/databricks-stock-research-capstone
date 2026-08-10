"""Offline contracts for the FastMCP-hosted stock research frontend."""

from pathlib import Path
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
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
        self.agent_client = MagicMock()
        self.services = MagicMock()
        self.services.research_context = self.research_context
        self.services_patcher = patch.object(
            server,
            "get_services",
            return_value=self.services,
        )
        self.get_services = self.services_patcher.start()
        self.agent_patcher = patch.object(
            server,
            "get_agent_client",
            return_value=self.agent_client,
        )
        self.get_agent_client = self.agent_patcher.start()
        app = server.mcp.http_app(
            path="/mcp",
            stateless_http=True,
            host_origin_protection=False,
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.agent_patcher.stop()
        self.services_patcher.stop()

    def test_homepage_and_static_assets_are_served_without_services(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("AI Stock Research Assistant", response.text)
        self.assertIn('id="research-form"', response.text)
        self.assertIn('value="AAPL"', response.text)
        self.assertIn("What developments could affect", response.text)
        self.assertIn("AI Research Summary", response.text)
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
        self.assertEqual(routes["/api/agent"], {"POST"})
        self.assertIn("POST", routes["/mcp"])

    def test_agent_api_validates_and_delegates_without_stock_facts(self) -> None:
        self.agent_client.generate_response.return_value = (
            "Supervisor-generated grounded answer."
        )

        to_thread = AsyncMock(
            side_effect=lambda operation, *args: operation(*args)
        )
        with patch.object(server.asyncio, "to_thread", new=to_thread):
            response = self.client.post(
                "/api/agent",
                json={
                    "ticker": " aapl ",
                    "question": "  What changed in the outlook?  ",
                },
            )

        self.assertEqual(response.status_code, 200)
        to_thread.assert_awaited_once()
        self.assertIs(
            to_thread.await_args.args[0],
            self.agent_client.generate_response,
        )
        prompt = self.agent_client.generate_response.call_args.args[0]
        self.assertIn("Research ticker AAPL.", prompt)
        self.assertIn("User question:\nWhat changed in the outlook?", prompt)
        self.assertIn("Use the available stock research MCP tools", prompt)
        self.assertNotIn("price", prompt.lower())
        self.assertEqual(
            response.json(),
            {
                "ok": True,
                "data": {
                    "ticker": "AAPL",
                    "answer": "Supervisor-generated grounded answer.",
                },
            },
        )

    def test_agent_api_validation_errors_return_400(self) -> None:
        for body in (
            {"ticker": "../AAPL", "question": "What changed?"},
            {"ticker": "AAPL", "question": "   "},
            ["AAPL"],
        ):
            with self.subTest(body=body):
                response = self.client.post("/api/agent", json=body)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(
                    response.json()["error"]["type"],
                    "validation_error",
                )
        self.agent_client.generate_response.assert_not_called()

    def test_agent_service_failure_returns_safe_503(self) -> None:
        self.agent_client.generate_response.side_effect = (
            server.AgentServiceError("private endpoint detail")
        )

        response = self.client.post(
            "/api/agent",
            json={"ticker": "AAPL", "question": "What changed?"},
        )

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error"]["type"], "agent_unavailable")
        self.assertNotIn("private", str(payload).lower())
        self.assertNotIn("endpoint", str(payload).lower())

    def test_agent_timeout_returns_safe_503(self) -> None:
        self.agent_client.generate_response.return_value = "Too late"

        with patch.object(server, "AGENT_TIMEOUT_SECONDS", 0):
            response = self.client.post(
                "/api/agent",
                json={"ticker": "AAPL", "question": "What changed?"},
            )

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error"]["type"], "agent_unavailable")
        self.assertEqual(
            payload["error"]["message"],
            "The AI research summary is temporarily unavailable.",
        )
        self.assertNotIn("timeout", str(payload).lower())

    def test_unexpected_agent_failure_returns_sanitized_500(self) -> None:
        self.agent_client.generate_response.side_effect = RuntimeError(
            "private workspace and credential detail"
        )

        response = self.client.post(
            "/api/agent",
            json={"ticker": "AAPL", "question": "What changed?"},
        )

        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["error"]["type"], "service_error")
        self.assertNotIn("workspace", str(payload).lower())
        self.assertNotIn("credential", str(payload).lower())

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

        to_thread = AsyncMock(
            side_effect=lambda operation, *args: operation(*args)
        )
        with patch.object(server.asyncio, "to_thread", new=to_thread):
            response = self.client.post(
                "/api/research",
                json={"ticker": " aapl ", "question": "  What changed?  "},
            )

        self.assertEqual(response.status_code, 200)
        to_thread.assert_awaited_once()
        self.assertIs(
            to_thread.await_args.args[0],
            self.research_context.build_research_context,
        )
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

    def test_agent_failure_keeps_deterministic_research_path(self) -> None:
        script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

        self.assertIn('requestResearch("/api/agent"', script)
        self.assertIn('requestResearch("/api/research"', script)
        self.assertIn("Promise.allSettled", script)
        self.assertIn("AI summary is temporarily unavailable", script)
        self.assertIn("renderResearch(researchOutcome.value)", script)
        self.assertIn("agentAnswer.textContent", script)

    def test_html_gateway_errors_are_sanitized_without_reading_body(self) -> None:
        script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        message = "The research service is temporarily unavailable."

        self.assertIn('response.headers.get("Content-Type")', script)
        self.assertIn('includes("application/json")', script)
        self.assertIn(message, script)
        self.assertNotIn("response.text()", script)
        self.assertLess(
            script.index('includes("application/json")'),
            script.index("response.json()"),
        )
        for status in (502, 503, 504):
            with self.subTest(status=status):
                self.assertNotIn(f"response.status === {status}", script)

    def test_malformed_json_uses_sanitized_error(self) -> None:
        script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        parse_start = script.index("try {\n    payload = await response.json();")
        parse_end = script.index("\n  }\n  if (!response.ok", parse_start)
        parse_boundary = script[parse_start:parse_end]

        self.assertIn("catch (_error)", parse_boundary)
        self.assertIn("RESEARCH_SERVICE_UNAVAILABLE_MESSAGE", parse_boundary)
        self.assertNotIn("_error.message", parse_boundary)
        self.assertNotIn("Unexpected token", script)

    def test_valid_json_success_and_structured_errors_are_preserved(self) -> None:
        script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

        self.assertIn("if (!response.ok || !payload?.ok)", script)
        self.assertIn("payload?.error?.message", script)
        self.assertIn("return payload.data", script)

    def test_agent_request_has_an_independent_browser_timeout(self) -> None:
        script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

        self.assertIn("const AGENT_REQUEST_TIMEOUT_MS = 65000", script)
        self.assertIn("new AbortController()", script)
        self.assertIn("() => controller.abort()", script)
        self.assertIn(
            'requestResearch("/api/agent", body, controller.signal)',
            script,
        )
        self.assertIn('requestResearch("/api/research", requestBody)', script)
        self.assertNotIn(
            'requestResearch("/api/research", requestBody,',
            script,
        )

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
