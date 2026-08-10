"""Offline tests for concise semantic-search CLI output."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest
from unittest.mock import MagicMock

from scripts.search_news import main


class SearchNewsScriptTests(unittest.TestCase):
    def test_script_delegates_and_prints_ranked_preview_without_vectors(self) -> None:
        service = MagicMock()
        service.semantic_news_search.return_value = [
            {
                "title": "Leadership update",
                "published_at": "2026-08-08T12:00:00Z",
                "similarity": 0.87654,
                "chunk_text": "Grounded article excerpt " * 30,
                "embedding": [0.1, 0.2, 0.3],
            }
        ]
        output = StringIO()

        with redirect_stdout(output):
            result = main(
                ["Apple CEO succession", "--ticker", "AAPL", "--top-k", "5"],
                service=service,
            )

        self.assertEqual(result, 0)
        service.semantic_news_search.assert_called_once_with(
            "Apple CEO succession",
            ticker="AAPL",
            top_k=5,
        )
        text = output.getvalue()
        self.assertIn("Leadership update", text)
        self.assertIn("similarity 0.8765", text)
        self.assertLessEqual(len(text.splitlines()[1].strip()), 240)
        self.assertNotIn("[0.1, 0.2, 0.3]", text)

    def test_script_reports_safe_failure(self) -> None:
        service = MagicMock()
        service.semantic_news_search.side_effect = RuntimeError("secret details")
        error_output = StringIO()

        with redirect_stderr(error_output):
            result = main(["query"], service=service)

        self.assertEqual(result, 1)
        self.assertEqual(error_output.getvalue(), "Semantic news search failed safely.\n")


if __name__ == "__main__":
    unittest.main()
