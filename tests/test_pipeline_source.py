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
        self.assertIn('chunks_to_write_df.write.format("postgresql")', self.source)

    def test_notebook_uses_oauth_without_massive_or_password_widget(self) -> None:
        self.assertIn("generate_database_credential", self.source)
        self.assertNotIn("MassiveClient", self.source)
        self.assertNotIn('dbutils.widgets.text("PGPASSWORD"', self.source)
        self.assertNotIn('dbutils.widgets.text("password"', self.source)

    def test_notebook_avoids_native_python_postgres_dependencies(self) -> None:
        self.assertNotIn("psycopg", self.source)
        self.assertNotIn("from app.db import", self.source)
        self.assertNotIn("StockRepository", self.source)

    def test_notebook_has_no_direct_spark_context_or_jvm_dependency(self) -> None:
        self.assertNotIn("spark._sc", self.source)
        self.assertNotIn("spark.sparkContext", self.source)
        self.assertNotIn("_gateway", self.source)
        self.assertNotIn("DriverManager", self.source)

    def test_serverless_postgresql_write_is_bounded_append_only(self) -> None:
        self.assertIn('.format("postgresql")', self.source)
        self.assertIn('.option("dbtable", "public.news_article_chunks")', self.source)
        self.assertIn('.option("batchsize", "100")', self.source)
        self.assertIn('.option("numPartitions", "1")', self.source)
        self.assertIn('.mode("append")', self.source)
        self.assertNotIn('.mode("overwrite")', self.source)
        self.assertIn("if total_chunks > 0:", self.source)
        self.assertIn("articles_persisted = len(article_ids_to_write)", self.source)

    def test_bounded_materialization_avoids_explicit_spark_storage(self) -> None:
        self.assertIn("extracted_rows = extracted_articles_df.collect()", self.source)
        self.assertIn("chunk_rows = chunks_df.collect()", self.source)
        self.assertNotIn(".cache()", self.source)
        self.assertNotIn(".persist(", self.source)

    def test_oauth_token_is_not_printed_or_displayed(self) -> None:
        self.assertIsNone(
            re.search(r"(?:print|display)\([^\n)]*database_token", self.source)
        )


if __name__ == "__main__":
    unittest.main()
