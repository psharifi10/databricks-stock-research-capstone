"""Reusable article extraction, fallback, and deterministic chunking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

import requests
import trafilatura


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MINIMUM_ARTICLE_CHARACTERS = 200
ARTICLE_USER_AGENT = (
    "DatabricksStockResearchCapstone/1.0 "
    "(educational article extraction; respectful bounded requests)"
)


@dataclass(frozen=True, slots=True)
class ArticleContent:
    """Usable article text and the source selected for it."""

    text: str
    extraction_source: str


def normalize_text(value: Any) -> str:
    """Collapse all repeated whitespace into deterministic single spaces."""

    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def build_metadata_fallback(
    title: Any,
    description: Any,
) -> str | None:
    """Build legitimate fallback text from already-persisted article metadata."""

    normalized_title = normalize_text(title)
    normalized_description = normalize_text(description)
    sections: list[str] = []
    if normalized_title:
        sections.append(f"Title: {normalized_title}")
    if normalized_description:
        sections.append(f"Description: {normalized_description}")
    return "\n\n".join(sections) or None


def extract_article_text(
    html: Any,
    *,
    article_url: str | None = None,
    minimum_characters: int = DEFAULT_MINIMUM_ARTICLE_CHARACTERS,
    extractor: Callable[..., Any] | None = None,
) -> str | None:
    """Extract normalized main-body text from supplied HTML without fetching it."""

    if (
        isinstance(minimum_characters, bool)
        or not isinstance(minimum_characters, int)
        or minimum_characters < 1
    ):
        raise ValueError("minimum_characters must be a positive integer.")
    if not isinstance(html, str) or not html.strip():
        return None

    extraction_function = extractor or trafilatura.extract
    try:
        extracted = extraction_function(
            html,
            url=article_url,
            output_format="txt",
            include_comments=False,
            include_tables=False,
        )
    except Exception:
        return None

    normalized = normalize_text(extracted)
    if len(normalized) < minimum_characters:
        return None
    return normalized


def create_article_session() -> requests.Session:
    """Create one modest, identifiable HTTP session for a Spark partition."""

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": ARTICLE_USER_AGENT,
        }
    )
    return session


def resolve_article_content(
    session: requests.Session,
    *,
    article_url: Any,
    title: Any,
    description: Any,
    request_timeout: float,
    minimum_characters: int = DEFAULT_MINIMUM_ARTICLE_CHARACTERS,
    extractor: Callable[..., Any] | None = None,
) -> ArticleContent | None:
    """Fetch/extract an article or safely use persisted metadata as fallback."""

    if (
        isinstance(request_timeout, bool)
        or not isinstance(request_timeout, (int, float))
        or request_timeout <= 0
    ):
        raise ValueError("request_timeout must be a positive number.")

    fallback = build_metadata_fallback(title, description)
    normalized_url = normalize_text(article_url)
    parsed_url = urlparse(normalized_url) if normalized_url else None
    usable_url = bool(
        parsed_url
        and parsed_url.scheme in {"http", "https"}
        and parsed_url.hostname
        and parsed_url.username is None
        and parsed_url.password is None
    )

    if usable_url:
        try:
            response = session.get(normalized_url, timeout=request_timeout)
            response.raise_for_status()
            extracted = extract_article_text(
                response.text,
                article_url=normalized_url,
                minimum_characters=minimum_characters,
                extractor=extractor,
            )
            if extracted:
                return ArticleContent(extracted, "article_body")
        except (requests.RequestException, TypeError, ValueError):
            pass

    if fallback:
        return ArticleContent(fallback, "metadata_fallback")
    return None


def chunk_text(
    text: Any,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split normalized text into deterministic, word-aware overlapping chunks."""

    _validate_chunk_settings(chunk_size, chunk_overlap)
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = hard_end
        if hard_end < text_length:
            minimum_break = start + chunk_overlap + 1
            word_break = normalized.rfind(" ", minimum_break, hard_end + 1)
            if word_break > minimum_break:
                end = word_break

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break

        next_start = end - chunk_overlap
        if 0 < next_start < text_length and normalized[next_start - 1] != " ":
            next_space = normalized.find(" ", next_start, end)
            if next_space != -1:
                next_start = next_space + 1
        start = max(start + 1, next_start)

    return chunks


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer.")
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise ValueError("chunk_overlap must be at least 0 and less than chunk_size.")
