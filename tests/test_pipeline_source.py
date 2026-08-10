"""Static offline contracts for the Databricks Phase 4A notebook source."""

from pathlib import Path
import re
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
        self.assertIn("replace_article_chunks_via_jdbc", self.source)

    def test_notebook_uses_oauth_without_massive_or_password_widget(self) -> None:
        self.assertIn("generate_database_credential", self.source)
        self.assertNotIn("MassiveClient", self.source)
        self.assertNotIn('dbutils.widgets.text("PGPASSWORD"', self.source)
        self.assertNotIn('dbutils.widgets.text("password"', self.source)

    def test_notebook_avoids_native_python_postgres_dependencies(self) -> None:
        self.assertNotIn("psycopg", self.source)
        self.assertNotIn("from app.db import", self.source)
        self.assertNotIn("StockRepository", self.source)

    def test_jdbc_writes_use_prepared_replacement_transactions(self) -> None:
        self.assertIn("java.sql.DriverManager.getConnection", self.source)
        self.assertIn("connection.setAutoCommit(False)", self.source)
        self.assertIn(
            '"DELETE FROM news_article_chunks WHERE article_id = ?"',
            self.source,
        )
        self.assertIn("VALUES (?, ?, ?)", self.source)
        self.assertIn("connection.prepareStatement", self.source)
        self.assertIn("delete_statement.setString(1, article_id)", self.source)
        self.assertIn(
            "insert_statement.setString(3, indexed_chunks[chunk_index])",
            self.source,
        )
        self.assertIn("connection.commit()", self.source)
        self.assertIn("connection.rollback()", self.source)

    def test_oauth_token_is_not_printed_or_displayed(self) -> None:
        self.assertIsNone(
            re.search(r"(?:print|display)\([^\n)]*database_token", self.source)
        )


if __name__ == "__main__":
    unittest.main()
