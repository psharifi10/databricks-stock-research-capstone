"""Static offline contracts for the Phase 4B Serverless notebook."""

from pathlib import Path
import re
import unittest

from pipelines.embeddings import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "pipelines" / "embed_news_chunks.py"


class EmbedNewsChunksNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = NOTEBOOK_PATH.read_text(encoding="utf-8")

    def test_notebook_is_serverless_compatible(self) -> None:
        self.assertTrue(self.source.startswith("# Databricks notebook source"))
        for forbidden in (
            "psycopg",
            "spark._sc",
            "SparkContext",
            "_jvm",
            "_gateway",
            "DriverManager",
            ".cache()",
            ".persist(",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_notebook_reads_only_bounded_unembedded_chunks(self) -> None:
        self.assertIn("spark.read.jdbc", self.source)
        self.assertIn("WHERE embedding IS NULL", self.source)
        self.assertIn('.limit(max_chunks)', self.source)
        self.assertIn('dbutils.widgets.text("max_chunks", "100"', self.source)
        self.assertIn('dbutils.widgets.text("embedding_partitions", "2"', self.source)

    def test_embedding_generation_is_distributed_and_uses_fixed_model(self) -> None:
        self.assertIn("mapInPandas", self.source)
        self.assertIn("SentenceTransformer(EMBEDDING_MODEL_NAME)", self.source)
        self.assertIn("normalize_embeddings=True", self.source)
        self.assertIn("validate_embedding(vector)", self.source)
        self.assertEqual(EMBEDDING_MODEL_NAME, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(EMBEDDING_DIMENSION, 384)

    def test_worker_configures_writable_hugging_face_cache_before_import(self) -> None:
        worker_start = self.source.index("def embed_chunk_partitions")
        cache_start = self.source.index(
            'cache_root = "/tmp/huggingface"',
            worker_start,
        )
        sentence_transformer_import = self.source.index(
            "from sentence_transformers import SentenceTransformer",
            worker_start,
        )

        self.assertLess(cache_start, sentence_transformer_import)
        self.assertIn('os.environ.setdefault("HF_HOME", cache_root)', self.source)
        self.assertIn('os.environ.setdefault("HF_HUB_CACHE"', self.source)
        self.assertIn('os.environ.setdefault("TRANSFORMERS_CACHE"', self.source)
        self.assertIn("os.makedirs(cache_path, exist_ok=True)", self.source)
        self.assertNotIn("/Workspace", self.source)
        self.assertNotIn("/Repos", self.source)

    def test_data_api_rpc_updates_existing_rows_without_exposing_values(self) -> None:
        self.assertIn("workspace_client.config.authenticate()", self.source)
        self.assertIn("/public/rpc/set_news_article_chunk_embedding", self.source)
        self.assertIn('"p_article_id"', self.source)
        self.assertIn('"p_chunk_index"', self.source)
        self.assertIn('"p_embedding"', self.source)
        self.assertNotIn('.mode("overwrite")', self.source)
        self.assertIsNone(
            re.search(
                r"(?:print|display)\([^\n)]*"
                r"(?:database_token|oauth_headers|row\.embedding|payload)",
                self.source,
                flags=re.IGNORECASE,
            )
        )


if __name__ == "__main__":
    unittest.main()
