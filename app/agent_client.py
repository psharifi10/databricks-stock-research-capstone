"""Lazy Databricks Responses API client for the deployed Supervisor Agent."""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

from app.validation import normalize_bounded_text


ClientFactory = Callable[[], Any]


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
        try:
            response = self._get_client().responses.create(
                model=endpoint_name,
                input=[{"role": "user", "content": normalized_prompt}],
                stream=False,
            )
        except Exception as error:
            raise AgentServiceError(
                "The Supervisor Agent request could not be completed."
            ) from error

        final_output_text = None
        for output in getattr(response, "output", None) or []:
            text_parts = []
            for content in getattr(output, "content", None) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
            if text_parts:
                final_output_text = " ".join(text_parts)

        if final_output_text is not None:
            return final_output_text

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        raise AgentServiceError(
            "The Supervisor Agent did not return a textual response."
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


def _create_databricks_client() -> Any:
    """Construct the supported client only when the first request needs it."""

    from databricks_openai import DatabricksOpenAI

    return DatabricksOpenAI()
