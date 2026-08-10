"""Static offline contracts for the Databricks Phase 4A notebook source."""

from pathlib import Path
import unittest


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "pipelines" / "process_news_content.py"
)


class ProcessNewsContentNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = NOTEBOOK_PATH.read_text(encoding="utf-8")

    def test_notebook_uses_spark_jdbc_anti_join_and_partition_processing(self) -> None:
        self.assertTrue(self.source.startswith("# Databricks notebook source"))
        self.assertIn("spark.read.jdbc", self.source)
        self.assertIn('"left_anti"', self.source)
        self.assertGreaterEqual(self.source.count("mapInPandas"), 2)
        self.assertIn("replace_news_article_chunks", self.source)

    def test_notebook_uses_oauth_without_massive_or_password_widget(self) -> None:
        self.assertIn("generate_database_credential", self.source)
        self.assertNotIn("MassiveClient", self.source)
        self.assertNotIn('dbutils.widgets.text("PGPASSWORD"', self.source)
        self.assertNotIn('dbutils.widgets.text("password"', self.source)


if __name__ == "__main__":
    unittest.main()
