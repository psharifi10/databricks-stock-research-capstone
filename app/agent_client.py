"""Lazy Databricks Responses API client for the deployed Supervisor Agent."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from typing import Any

from app.validation import normalize_bounded_text


ClientFactory = Callable[[], Any]
READ_ONLY_MCP_TOOLS = frozenset(
    {
        "get_company",
        "get_price_history",
        "search_financial_news",
        "build_research_context",
        "health",
    }
)
MAX_SUPERVISOR_TURNS = 5


class AgentServiceError(RuntimeError):
    """A browser-safe failure raised by the Supervisor Agent boundary."""


class SupervisorAgentClient:
    """Query one configured Supervisor endpoint through unified authentication."""

    def __init__(
        self,
        *,
        endpoint_name: str | None = None,
        client: Any | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._configured_endpoint = endpoint_name
        self._client = client
        self._client_factory = client_factory or _create_databricks_client

    def generate_response(self, prompt: str) -> str:
        """Return only final textual output from a non-streaming agent request."""

        normalized_prompt = normalize_bounded_text(
            prompt,
            field_name="Agent prompt",
            maximum=10000,
        )
        endpoint_name = self._resolve_endpoint_name()
        input_history: list[Any] = [
            {"role": "user", "content": normalized_prompt}
        ]
        for _turn in range(MAX_SUPERVISOR_TURNS):
            try:
                response = self._get_client().responses.create(
                    model=endpoint_name,
                    input=input_history,
                    stream=False,
                )
            except Exception as error:
                raise AgentServiceError(
                    "The Supervisor Agent request could not be completed."
                ) from error

            outputs = _field(response, "output") or []
            approval_requests = [
                output
                for output in outputs
                if _field(output, "type") == "mcp_approval_request"
            ]
            if not approval_requests:
                return _extract_final_text(response)

            for approval_request in approval_requests:
                if _field(approval_request, "name") not in READ_ONLY_MCP_TOOLS:
                    raise AgentServiceError(
                        "This action requires explicit confirmation and cannot "
                        "be performed from the research dashboard."
                    )

            input_history.extend(
                _serialize_output_item(output) for output in outputs
            )
            input_history.extend(
                {
                    "type": "mcp_approval_response",
                    "id": _field(approval_request, "id"),
                    "approval_request_id": _field(approval_request, "id"),
                    "approve": True,
                }
                for approval_request in approval_requests
            )

        raise AgentServiceError(
            "The Supervisor Agent could not complete the request within the "
            "allowed number of turns."
        )

    def _resolve_endpoint_name(self) -> str:
        endpoint_name = self._configured_endpoint
        if endpoint_name is None:
            endpoint_name = os.environ.get("SUPERVISOR_ENDPOINT")
        normalized = str(endpoint_name or "").strip()
        if not normalized:
            raise AgentServiceError(
                "The Supervisor Agent endpoint is not configured."
            )
        return normalized

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client


def _extract_final_text(response: Any) -> str:
    """Return the last structured assistant text after approvals are complete."""

    final_output_text = None
    for output in _field(response, "output") or []:
        text_parts = []
        for content in _field(output, "content") or []:
            text = _field(content, "text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        if text_parts:
            final_output_text = " ".join(text_parts)

    if final_output_text is not None:
        return final_output_text

    output_text = _field(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    raise AgentServiceError(
        "The Supervisor Agent did not return a textual response."
    )


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _serialize_output_item(output: Any) -> dict[str, Any]:
    if isinstance(output, Mapping):
        return dict(output)
    model_dump = getattr(output, "model_dump", None)
    if not callable(model_dump):
        raise AgentServiceError(
            "The Supervisor Agent response could not be continued safely."
        )
    serialized = model_dump()
    if not isinstance(serialized, Mapping):
        raise AgentServiceError(
            "The Supervisor Agent response could not be continued safely."
        )
    return dict(serialized)


def _create_databricks_client() -> Any:
    """Construct the supported client only when the first request needs it."""

    from databricks_openai import DatabricksOpenAI

    return DatabricksOpenAI()
