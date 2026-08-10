"""Static contracts for the idempotent Phase 4B vector migration."""

from pathlib import Path
import unittest


MIGRATION_PATH = (
    Path(__file__).resolve().parents[1] / "sql" / "002_chunk_embeddings.sql"
)


class EmbeddingMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.migration = MIGRATION_PATH.read_text(encoding="utf-8")
        cls.compact = " ".join(cls.migration.split())

    def test_migration_adds_vector_columns_without_recreating_chunks(self) -> None:
        self.assertIn("CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE", self.compact)
        self.assertIn("ADD COLUMN IF NOT EXISTS embedding VECTOR(384)", self.compact)
        self.assertIn("ADD COLUMN IF NOT EXISTS embedding_model TEXT", self.compact)
        self.assertNotIn("DROP TABLE", self.migration.upper())
        self.assertNotIn("TRUNCATE", self.migration.upper())
        self.assertNotIn("CREATE TABLE", self.migration.upper())

    def test_migration_creates_lakebase_cosine_ann_index(self) -> None:
        self.assertIn("news_article_chunks_embedding_ann", self.migration)
        self.assertIn("USING lakebase_ann", self.migration)
        self.assertIn("embedding vector_cosine_ops", self.migration)

    def test_data_api_rpc_updates_existing_composite_key_row(self) -> None:
        self.assertIn("SET embedding = p_embedding::VECTOR(384)", self.compact)
        self.assertIn("WHERE article_id = p_article_id", self.compact)
        self.assertIn("AND chunk_index = p_chunk_index", self.compact)
        self.assertNotIn("INSERT INTO news_article_chunks", self.migration)


if __name__ == "__main__":
    unittest.main()
