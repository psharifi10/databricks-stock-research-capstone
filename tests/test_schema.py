"""Static tests for important relational schema contracts."""

from pathlib import Path
import unittest


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "001_core_schema.sql"


class NewsSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = SCHEMA_PATH.read_text(encoding="utf-8")

    def test_news_articles_contains_only_article_level_sentiment_model(self) -> None:
        table = self.schema.split(
            "CREATE TABLE IF NOT EXISTS news_articles (",
            maxsplit=1,
        )[1].split(");", maxsplit=1)[0]

        self.assertNotIn("sentiment TEXT", table)
        self.assertNotIn("sentiment_reasoning TEXT", table)

    def test_news_article_tickers_owns_nullable_sentiment(self) -> None:
        table = self.schema.split(
            "CREATE TABLE IF NOT EXISTS news_article_tickers (",
            maxsplit=1,
        )[1].split(");", maxsplit=1)[0]

        self.assertIn("sentiment TEXT", table)
        self.assertIn("sentiment_reasoning TEXT", table)
        self.assertIn("PRIMARY KEY (article_id, ticker)", table)


if __name__ == "__main__":
    unittest.main()
