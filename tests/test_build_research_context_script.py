"""Offline tests for the safe research-context preview CLI."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import MagicMock

from scripts.build_research_context import main


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_research_context.py"
)


class BuildResearchContextScriptTests(unittest.TestCase):
    def test_cli_delegates_and_prints_counts_and_evidence_without_vectors(self) -> None:
        service = MagicMock()
        service.build_research_context.return_value = {
            "ticker": "AAPL",
            "question": "What changed?",
            "company": {"ticker": "AAPL", "name": "Apple Inc."},
            "prices": [{"price_date": "2026-08-08", "close": "220.25"}],
            "recent_news": [{"article_id": "article-1"}],
            "semantic_evidence": [
                {
                    "title": "Leadership update",
                    "similarity": 0.87654,
                    "chunk_text": "Do not print this full chunk.",
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
        }
        output = StringIO()

        with redirect_stdout(output):
            result = main(
                [
                    "AAPL",
                    "What changed?",
                    "--semantic-top-k",
                    "4",
                    "--recent-news-limit",
                    "3",
                    "--price-history-limit",
                    "20",
                ],
                service=service,
            )

        self.assertEqual(result, 0)
        service.build_research_context.assert_called_once_with(
            "AAPL",
            "What changed?",
            semantic_top_k=4,
            recent_news_limit=3,
            price_history_limit=20,
        )
        text = output.getvalue()
        self.assertIn("Company records: 1", text)
        self.assertIn("Price observations: 1", text)
        self.assertIn("Recent news articles: 1", text)
        self.assertIn("Semantic evidence chunks: 1", text)
        self.assertIn("Leadership update | similarity 0.8765", text)
        self.assertNotIn("[0.1, 0.2, 0.3]", text)
        self.assertNotIn("Do not print this full chunk.", text)

    def test_cli_source_has_no_massive_llm_or_vector_output_path(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertNotIn("MassiveClient", source)
        self.assertNotIn("openai", source.lower())
        self.assertNotIn("row.embedding", source)
        self.assertNotIn("context['embedding']", source)

    def test_cli_reports_safe_failure(self) -> None:
        service = MagicMock()
        service.build_research_context.side_effect = RuntimeError("secret details")
        error_output = StringIO()

        with redirect_stderr(error_output):
            result = main(["AAPL", "What changed?"], service=service)

        self.assertEqual(result, 1)
        self.assertEqual(
            error_output.getvalue(),
            "Research context assembly failed safely.\n",
        )


if __name__ == "__main__":
    unittest.main()
