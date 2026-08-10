"""Offline tests for article extraction, fallback, and deterministic chunking."""

import unittest
from unittest.mock import MagicMock

import requests

from pipelines.article_processing import (
    ARTICLE_USER_AGENT,
    build_metadata_fallback,
    chunk_text,
    create_article_session,
    extract_article_text,
    normalize_text,
    resolve_article_content,
)


class ArticleTextTests(unittest.TestCase):
    def test_normalize_text_collapses_repeated_whitespace(self) -> None:
        self.assertEqual(normalize_text("  alpha\n\tbeta   gamma "), "alpha beta gamma")
        self.assertEqual(normalize_text(" \n\t "), "")
        self.assertEqual(normalize_text(None), "")

    def test_usable_html_extracts_body_content(self) -> None:
        body = " ".join(["Substantive article paragraph."] * 20)
        html = f"<html><body><nav>Menu</nav><article><p>{body}</p></article></body></html>"

        extracted = extract_article_text(html, minimum_characters=100)

        self.assertIsNotNone(extracted)
        self.assertIn("Substantive article paragraph.", extracted)
        self.assertNotIn("Menu", extracted)

    def test_failed_or_short_extraction_uses_metadata_without_invention(self) -> None:
        session = MagicMock()
        response = MagicMock()
        response.text = "<html><body>short</body></html>"
        session.get.return_value = response

        content = resolve_article_content(
            session,
            article_url="https://news.example.invalid/article",
            title="Known title",
            description="Known description",
            request_timeout=5,
            extractor=lambda *args, **kwargs: None,
        )

        self.assertEqual(content.extraction_source, "metadata_fallback")
        self.assertEqual(
            content.text,
            "Title: Known title\n\nDescription: Known description",
        )

    def test_resolved_usable_html_is_labeled_article_body(self) -> None:
        session = MagicMock()
        response = MagicMock()
        response.text = "<html><article><p>" + ("Body sentence. " * 30) + "</p></article></html>"
        session.get.return_value = response

        content = resolve_article_content(
            session,
            article_url="https://news.example.invalid/article",
            title="Fallback title",
            description="Fallback description",
            request_timeout=5,
            minimum_characters=100,
        )

        self.assertEqual(content.extraction_source, "article_body")
        self.assertIn("Body sentence.", content.text)
        self.assertNotIn("Fallback title", content.text)

    def test_http_timeout_uses_fallback(self) -> None:
        session = MagicMock()
        session.get.side_effect = requests.Timeout("private upstream detail")

        content = resolve_article_content(
            session,
            article_url="https://news.example.invalid/article",
            title="Known title",
            description=None,
            request_timeout=5,
        )

        self.assertEqual(content.text, "Title: Known title")
        self.assertEqual(content.extraction_source, "metadata_fallback")

    def test_blank_metadata_and_failed_fetch_produce_no_content(self) -> None:
        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("unavailable")

        content = resolve_article_content(
            session,
            article_url="https://news.example.invalid/article",
            title=" ",
            description=None,
            request_timeout=5,
        )

        self.assertIsNone(content)

    def test_metadata_fallback_variants(self) -> None:
        self.assertEqual(
            build_metadata_fallback("Title", "Description"),
            "Title: Title\n\nDescription: Description",
        )
        self.assertEqual(build_metadata_fallback("Title", None), "Title: Title")
        self.assertIsNone(build_metadata_fallback(" ", None))

    def test_partition_session_has_educational_user_agent(self) -> None:
        session = create_article_session()
        try:
            self.assertEqual(session.headers["User-Agent"], ARTICLE_USER_AGENT)
        finally:
            session.close()


class ChunkTextTests(unittest.TestCase):
    def test_short_text_produces_one_normalized_chunk(self) -> None:
        self.assertEqual(
            chunk_text(" alpha   beta ", chunk_size=50, chunk_overlap=10),
            ["alpha beta"],
        )

    def test_long_text_is_deterministic_with_overlap_and_no_empty_chunks(self) -> None:
        text = " ".join(f"word{index:03d}" for index in range(80))

        first = chunk_text(text, chunk_size=100, chunk_overlap=24)
        second = chunk_text(text, chunk_size=100, chunk_overlap=24)

        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        self.assertTrue(all(chunk and len(chunk) <= 100 for chunk in first))
        for current, following in zip(first, first[1:]):
            self.assertTrue(set(current.split()) & set(following.split()))

    def test_invalid_chunk_settings_are_rejected(self) -> None:
        invalid_settings = [
            {"chunk_size": 0, "chunk_overlap": 0},
            {"chunk_size": 10, "chunk_overlap": -1},
            {"chunk_size": 10, "chunk_overlap": 10},
            {"chunk_size": 10, "chunk_overlap": 11},
        ]
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    chunk_text("some text", **settings)

    def test_blank_text_produces_no_chunks(self) -> None:
        self.assertEqual(chunk_text(" \n\t "), [])


if __name__ == "__main__":
    unittest.main()
