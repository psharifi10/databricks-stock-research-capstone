"""Offline tests for the lazy Databricks Supervisor Agent client."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from app.agent_client import (
    AgentServiceError,
    MAX_SUPERVISOR_TURNS,
    READ_ONLY_MCP_TOOLS,
    SupervisorAgentClient,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = PROJECT_ROOT / "app" / "agent_client.py"


def _message_output(text: str, *, item_id: str = "message-1") -> MagicMock:
    output = MagicMock()
    output.type = "message"
    output.content = [SimpleNamespace(type="output_text", text=text)]
    output.model_dump.return_value = {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    return output


def _approval_output(tool_name: str, *, item_id: str = "approval-1") -> MagicMock:
    output = MagicMock()
    output.type = "mcp_approval_request"
    output.name = tool_name
    output.id = item_id
    output.content = []
    output.model_dump.return_value = {
        "id": item_id,
        "type": "mcp_approval_request",
        "name": tool_name,
        "arguments": "[not exposed to browser]",
    }
    return output


class SupervisorAgentClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.responses = MagicMock()
        self.client = SimpleNamespace(responses=self.responses)
        self.client_factory = MagicMock(return_value=self.client)

    def test_endpoint_is_read_from_environment_and_client_is_lazy(self) -> None:
        self.responses.create.return_value = SimpleNamespace(
            output=[
                SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            text="  Grounded synthesized answer.  "
                        )
                    ]
                )
            ],
            output_text="Intermediate convenience text.",
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

    def test_final_output_item_wins_over_intermediate_tool_intent(self) -> None:
        self.responses.create.return_value = SimpleNamespace(
            output=[
                SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            text="I'll search for recent financial news."
                        )
                    ]
                ),
                SimpleNamespace(content=[]),
                SimpleNamespace(
                    content=[
                        SimpleNamespace(text="  Final grounded answer. "),
                        SimpleNamespace(text=" Supporting synthesis.  "),
                    ]
                ),
            ],
            output_text="I'll search for recent financial news.",
        )
        agent = SupervisorAgentClient(
            endpoint_name="injected-test-endpoint",
            client=self.client,
        )

        answer = agent.generate_response("Research AAPL.")

        self.assertEqual(
            answer,
            "Final grounded answer. Supporting synthesis.",
        )
        self.assertNotIn("I'll search", answer)

    def test_empty_response_output_uses_output_text_fallback(self) -> None:
        self.responses.create.return_value = SimpleNamespace(
            output=[],
            output_text="  Safe fallback answer.  ",
        )
        agent = SupervisorAgentClient(
            endpoint_name="injected-test-endpoint",
            client=self.client,
        )

        answer = agent.generate_response("Research AAPL.")

        self.assertEqual(answer, "Safe fallback answer.")

    def test_read_only_approval_continues_to_final_synthesis(self) -> None:
        planning = _message_output(
            "I'll search for recent financial news.",
            item_id="message-planning",
        )
        approval = _approval_output(
            "search_financial_news",
            item_id="approval-search",
        )
        final = _message_output(
            "Final grounded Apple news synthesis.",
            item_id="message-final",
        )
        self.responses.create.side_effect = [
            SimpleNamespace(
                output=[planning, approval],
                output_text="I'll search for recent financial news.",
            ),
            SimpleNamespace(
                output=[final],
                output_text="Final grounded Apple news synthesis.",
            ),
        ]
        agent = SupervisorAgentClient(
            endpoint_name="injected-test-endpoint",
            client=self.client,
        )

        answer = agent.generate_response("Research AAPL.")

        self.assertEqual(answer, "Final grounded Apple news synthesis.")
        self.assertNotIn("I'll search", answer)
        self.assertEqual(self.responses.create.call_count, 2)
        continuation_history = self.responses.create.call_args_list[1].kwargs[
            "input"
        ]
        self.assertEqual(
            continuation_history,
            [
                {"role": "user", "content": "Research AAPL."},
                planning.model_dump.return_value,
                approval.model_dump.return_value,
                {
                    "type": "mcp_approval_response",
                    "id": "approval-search",
                    "approval_request_id": "approval-search",
                    "approve": True,
                },
            ],
        )

    def test_exact_read_only_allowlist_tools_are_approved(self) -> None:
        self.assertIsInstance(READ_ONLY_MCP_TOOLS, frozenset)
        self.assertEqual(
            READ_ONLY_MCP_TOOLS,
            {
                "get_company",
                "get_price_history",
                "search_financial_news",
                "build_research_context",
                "health",
            },
        )
        for tool_name in sorted(READ_ONLY_MCP_TOOLS):
            with self.subTest(tool_name=tool_name):
                responses = MagicMock()
                responses.create.side_effect = [
                    SimpleNamespace(
                        output=[_approval_output(tool_name)],
                        output_text="Planning.",
                    ),
                    SimpleNamespace(
                        output=[_message_output("Final answer.")],
                        output_text="Final answer.",
                    ),
                ]
                agent = SupervisorAgentClient(
                    endpoint_name="injected-test-endpoint",
                    client=SimpleNamespace(responses=responses),
                )

                self.assertEqual(
                    agent.generate_response("Research AAPL."),
                    "Final answer.",
                )
                approval_response = responses.create.call_args_list[1].kwargs[
                    "input"
                ][-1]
                self.assertEqual(
                    approval_response["type"],
                    "mcp_approval_response",
                )
                self.assertIs(approval_response["approve"], True)

    def test_write_and_unknown_tools_are_never_automatically_approved(self) -> None:
        rejected_tools = (
            "add_to_watchlist",
            "remove_from_watchlist",
            "save_research_note",
            "save_analysis_report",
            "unknown_tool",
        )
        for tool_name in rejected_tools:
            with self.subTest(tool_name=tool_name):
                responses = MagicMock()
                responses.create.return_value = SimpleNamespace(
                    output=[_approval_output(tool_name)],
                    output_text="Planning.",
                )
                agent = SupervisorAgentClient(
                    endpoint_name="injected-test-endpoint",
                    client=SimpleNamespace(responses=responses),
                )

                with self.assertRaisesRegex(
                    AgentServiceError,
                    "requires explicit confirmation",
                ):
                    agent.generate_response("Research AAPL.")

                responses.create.assert_called_once()

    def test_approval_continuation_turns_are_bounded(self) -> None:
        self.responses.create.return_value = SimpleNamespace(
            output=[_approval_output("health")],
            output_text="Planning.",
        )
        agent = SupervisorAgentClient(
            endpoint_name="injected-test-endpoint",
            client=self.client,
        )

        with self.assertRaisesRegex(
            AgentServiceError,
            "allowed number of turns",
        ):
            agent.generate_response("Research AAPL.")

        self.assertEqual(self.responses.create.call_count, MAX_SUPERVISOR_TURNS)

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
        self.responses.create.return_value = SimpleNamespace(
            output=[
                SimpleNamespace(
                    content=[
                        SimpleNamespace(text="   "),
                        SimpleNamespace(value="not textual content"),
                    ]
                )
            ],
            output_text="   ",
        )
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
