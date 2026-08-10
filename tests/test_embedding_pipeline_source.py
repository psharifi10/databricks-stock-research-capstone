"""Static offline contracts for the Phase 4B Serverless notebook."""

import ast
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
        tree = ast.parse(cls.source)
        helper_names = {
            "_safe_postgrest_code",
            "_safe_postgrest_message",
            "_safe_content_type",
            "_safe_json_keys",
            "_data_api_response_summary",
            "_data_api_http_failure",
        }
        helper_nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id == "DATA_API_ERROR_MESSAGE_LIMIT"
                for target in node.targets
            ):
                helper_nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in helper_names:
                helper_nodes.append(node)
        namespace: dict[str, object] = {}
        exec(
            compile(
                ast.Module(body=helper_nodes, type_ignores=[]),
                str(NOTEBOOK_PATH),
                "exec",
            ),
            namespace,
        )
        cls.data_api_http_failure = staticmethod(namespace["_data_api_http_failure"])

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

    def test_http_error_retains_status_and_safe_postgrest_fields(self) -> None:
        response = _FakeResponse(
            403,
            {"code": "42501", "message": "permission denied for function"},
            content_type="application/json",
        )

        error = self.data_api_http_failure(response)

        self.assertEqual(
            str(error),
            "Lakebase embedding persistence failed (HTTP 403, "
            "content-type=application/json, json_keys=code,message, "
            "code=42501: message=permission denied for function)",
        )

    def test_data_api_error_message_is_bounded(self) -> None:
        response = _FakeResponse(
            404,
            {"code": "PGRST202", "message": "x" * 500},
            content_type="application/json; charset=utf-8",
        )

        message = str(self.data_api_http_failure(response))

        self.assertIn("HTTP 404", message)
        self.assertIn("code=PGRST202", message)
        self.assertIn("content-type=application/json; charset=utf-8", message)
        self.assertTrue(message.endswith(f"message={'x' * 200})"))

    def test_data_api_error_never_surfaces_sensitive_or_structured_message(self) -> None:
        sensitive = (
            "Authorization: Bearer oauth-token-secret "
            "p_embedding=[0.1,0.2] payload={private}"
        )
        response = _FakeResponse(
            403,
            {"code": "42501", "message": sensitive},
            content_type="application/json",
        )

        message = str(self.data_api_http_failure(response))

        self.assertEqual(
            message,
            "Lakebase embedding persistence failed (HTTP 403, "
            "content-type=application/json, json_keys=code,message, "
            "code=42501)",
        )
        for forbidden in (
            "Authorization",
            "oauth-token-secret",
            "[0.1,0.2]",
            "payload",
        ):
            self.assertNotIn(forbidden, message)

    def test_data_api_non_json_error_retains_status_without_raw_body(self) -> None:
        response = _FakeResponse(
            502,
            ValueError("not JSON"),
            content_type="text/plain",
        )

        message = str(self.data_api_http_failure(response))

        self.assertEqual(
            message,
            "Lakebase embedding persistence failed "
            "(HTTP 502, content-type=text/plain)",
        )
        self.assertNotIn("response.text", self.source)
        self.assertNotIn("response.content", self.source)
        self.assertIn("except requests.Timeout:", self.source)
        self.assertIn("except requests.ConnectionError:", self.source)

    def test_postgres_type_notation_survives_while_numeric_arrays_are_redacted(self) -> None:
        response = _FakeResponse(
            400,
            {
                "code": "42804",
                "details": (
                    "argument p_embedding has type real[] but numeric value "
                    "[0.1, -2e-3, 3] was rejected"
                ),
                "hint": "Use the declared RPC argument type.",
            },
            content_type="application/json",
        )

        message = str(self.data_api_http_failure(response))

        self.assertIn("json_keys=code,details,hint", message)
        self.assertIn("real[]", message)
        self.assertIn("[redacted]", message)
        self.assertNotIn("[0.1, -2e-3, 3]", message)
        self.assertIn("hint=Use the declared RPC argument type.", message)

    def test_rollback_diagnostic_runs_once_before_normal_persistence(self) -> None:
        diagnostic_start = self.source.index("diagnostic_row = embedded_rows[0]")
        normal_loop = self.source.index("for row in embedded_rows:", diagnostic_start)
        diagnostic_source = self.source[diagnostic_start:normal_loop]

        self.assertEqual(diagnostic_source.count("session.post("), 1)
        self.assertIn('headers={"Prefer": "tx=rollback"}', diagnostic_source)
        self.assertIn(
            '"p_article_id": str(diagnostic_row.article_id)',
            diagnostic_source,
        )
        self.assertIn(
            '"p_chunk_index": int(diagnostic_row.chunk_index)',
            diagnostic_source,
        )
        self.assertIn('"p_embedding": [0.0] * EMBEDDING_DIMENSION', diagnostic_source)
        self.assertIn(
            '"p_embedding_model": "phase4b-data-api-diagnostic"',
            diagnostic_source,
        )
        self.assertIn(
            'raise RuntimeError("Data API diagnostic RPC succeeded.")',
            diagnostic_source,
        )
        self.assertIn("diagnostic_response.raise_for_status()", diagnostic_source)


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        body: object,
        *,
        content_type: str | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = (
            {"Content-Type": content_type}
            if content_type is not None
            else {}
        )

    def json(self) -> object:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


if __name__ == "__main__":
    unittest.main()
