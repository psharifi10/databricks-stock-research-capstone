"""Offline tests for the lazy Databricks Supervisor Agent client."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from app.agent_client import AgentServiceError, SupervisorAgentClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = PROJECT_ROOT / "app" / "agent_client.py"


class SupervisorAgentClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.responses = MagicMock()
        self.client = SimpleNamespace(responses=self.responses)
        self.client_factory = MagicMock(return_value=self.client)

    def test_endpoint_is_read_from_environment_and_client_is_lazy(self) -> None:
        self.responses.create.return_value = SimpleNamespace(
            output_text="  Grounded synthesized answer.  "
        )
        agent = SupervisorAgentClient(client_factory=self.client_factory)

        self.client_factory.assert_not_called()
        with patch.dict(
            os.environ,
            {"SUPERVISOR_ENDPOINT": "configured-by-app-resource"},
            clear=False,
        ):
            answer = agent.generate_response("  Research AAPL.  ")

        self.client_factory.assert_called_once_with()
        self.responses.create.assert_called_once_with(
            model="configured-by-app-resource",
            input=[{"role": "user", "content": "Research AAPL."}],
            stream=False,
        )
        self.assertEqual(answer, "Grounded synthesized answer.")

    def test_missing_endpoint_fails_before_client_initialization(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            agent = SupervisorAgentClient(client_factory=self.client_factory)

            with self.assertRaisesRegex(
                AgentServiceError,
                "endpoint is not configured",
            ):
                agent.generate_response("Research AAPL.")

        self.client_factory.assert_not_called()
        self.responses.create.assert_not_called()

    def test_endpoint_failure_and_empty_output_are_safe(self) -> None:
        self.responses.create.side_effect = RuntimeError(
            "private endpoint configuration detail"
        )
        agent = SupervisorAgentClient(
            endpoint_name="injected-test-endpoint",
            client=self.client,
        )

        with self.assertRaises(AgentServiceError) as raised:
            agent.generate_response("Research AAPL.")
        self.assertNotIn("private", str(raised.exception).lower())
        self.assertNotIn("endpoint configuration detail", str(raised.exception))

        self.responses.create.side_effect = None
        self.responses.create.return_value = SimpleNamespace(output_text="   ")
        with self.assertRaisesRegex(
            AgentServiceError,
            "did not return a textual response",
        ):
            agent.generate_response("Research AAPL.")

    def test_supported_client_import_occurs_only_inside_lazy_factory(self) -> None:
        source = CLIENT_PATH.read_text(encoding="utf-8")

        self.assertLess(
            source.index("def _create_databricks_client"),
            source.index("from databricks_openai import DatabricksOpenAI"),
        )
        self.assertNotIn("DATABRICKS_CLIENT_ID", source)
        self.assertNotIn("DATABRICKS_CLIENT_SECRET", source)
        self.assertNotIn("DATABRICKS_TOKEN", source)


if __name__ == "__main__":
    unittest.main()
