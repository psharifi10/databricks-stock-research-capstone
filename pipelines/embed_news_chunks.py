# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4B: embed persisted financial-news chunks
# MAGIC
# MAGIC This bounded Serverless pipeline reads only unembedded Lakebase chunks,
# MAGIC generates 384-dimensional embeddings through Spark, and updates the
# MAGIC existing chunk rows through the OAuth-authenticated Lakebase Data API.

# COMMAND ----------

# MAGIC %pip install "databricks-sdk>=0.89,<1.0" "sentence-transformers>=5.1,<6.0" "requests>=2.32,<3.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from collections.abc import Iterator
from urllib.parse import urlparse

import pandas as pd
import requests

from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from pipelines.embeddings import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME


DATA_API_CLIENT_ID = "626bf1da-0ff0-4af7-96f1-75ab9ca6aa08"
DATA_API_ERROR_FIELD_LIMIT = 500


def _safe_postgrest_code(value: object) -> str | None:
    import re

    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if not isinstance(value, str):
        return None
    code = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", code):
        return None
    return code


def _sanitize_data_api_field(value: object) -> str | None:
    import re

    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None

    text = re.sub(
        r"(?i)\b(?:https?|postgres(?:ql)?):\/\/[^\s<>()]+",
        "[redacted-url]",
        text,
    )
    text = re.sub(
        r"(?i)\bBearer\s+[^\s,;)\]}]+",
        "Bearer [redacted]",
        text,
    )
    text = re.sub(
        r"(?i)\b(oauth(?:[_ -]?(?:token|secret))?|token|password|secret)"
        r"(\s*(?:[:=]\s*|\s+))([^\s,;)\]}]+)",
        r"\1\2[redacted]",
        text,
    )
    for _ in range(5):
        redacted = re.sub(r"\{[^{}]*\}", "[redacted-object]", text)
        if redacted == text:
            break
        text = redacted
    text = re.sub(
        r"\[(?=[^\]\r\n]*\d)[^\]\r\n]*\]",
        "[redacted-array]",
        text,
    )
    return text[:DATA_API_ERROR_FIELD_LIMIT]


def _safe_content_type(response: object) -> str:
    import re

    headers = getattr(response, "headers", {})
    if not hasattr(headers, "get"):
        return "unknown"
    value = headers.get("Content-Type")
    if not isinstance(value, str):
        return "unknown"
    content_type = " ".join(value.split())[:100]
    if not re.fullmatch(r"[A-Za-z0-9!#$&^_.+*/;= -]+", content_type):
        return "unknown"
    return content_type


def _safe_json_keys(body: object) -> tuple[str, ...]:
    import re

    if not isinstance(body, dict):
        return ()
    keys = (
        key
        for key in body
        if isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", key)
    )
    return tuple(sorted(keys))[:10]


def _data_api_response_summary(response: object) -> str:
    raw_status = getattr(response, "status_code", None)
    status = (
        raw_status
        if isinstance(raw_status, int) and not isinstance(raw_status, bool)
        else "unknown"
    )
    try:
        body = response.json()
    except (TypeError, ValueError):
        body = None

    metadata = [f"HTTP {status}", f"content-type={_safe_content_type(response)}"]
    json_keys = _safe_json_keys(body)
    if json_keys:
        metadata.append(f"json_keys={','.join(json_keys)}")

    safe_fields = []
    if isinstance(body, dict):
        code = _safe_postgrest_code(body.get("code"))
        if code is not None:
            safe_fields.append(f"code={code}")
        for field in ("message", "detail", "details", "hint"):
            value = _sanitize_data_api_field(body.get(field))
            if value is not None:
                safe_fields.append(f"{field}={value}")

    summary = ", ".join(metadata)
    if safe_fields:
        summary += ": " + "; ".join(safe_fields)
    return summary


def _data_api_http_failure(response: object) -> RuntimeError:
    return RuntimeError(
        "Lakebase embedding persistence failed "
        f"({_data_api_response_summary(response)})"
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Non-secret bounded run parameters
# MAGIC
# MAGIC `data_api_url` is the non-secret base URL shown on the Lakebase Data API
# MAGIC page. Database and workspace OAuth credentials remain in memory only.

# COMMAND ----------

dbutils.widgets.text("endpoint_name", "", "Lakebase endpoint resource name")
dbutils.widgets.text("pg_host", "", "Lakebase PostgreSQL host")
dbutils.widgets.text("pg_database", "databricks_postgres", "PostgreSQL database")
dbutils.widgets.text("pg_port", "5432", "PostgreSQL port")
dbutils.widgets.text("pg_sslmode", "require", "PostgreSQL SSL mode")
dbutils.widgets.text("data_api_url", "", "Lakebase Data API base URL")
dbutils.widgets.text("max_chunks", "100", "Maximum chunks in this run")
dbutils.widgets.text("embedding_partitions", "2", "Embedding partitions")


def _required_widget(name: str) -> str:
    value = dbutils.widgets.get(name).strip()
    if not value:
        raise ValueError(f"Widget {name} is required.")
    return value


def _bounded_int(name: str, *, minimum: int, maximum: int) -> int:
    try:
        value = int(dbutils.widgets.get(name))
    except ValueError as error:
        raise ValueError(f"Widget {name} must be an integer.") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"Widget {name} must be between {minimum} and {maximum}.")
    return value


def _required_https_url(name: str) -> str:
    value = _required_widget(name).rstrip("/")
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"Widget {name} must be a credential-free HTTPS URL.")
    return value


endpoint_name = _required_widget("endpoint_name")
pg_host = _required_widget("pg_host")
pg_database = _required_widget("pg_database")
pg_port = _bounded_int("pg_port", minimum=1, maximum=65535)
pg_sslmode = _required_widget("pg_sslmode")
data_api_url = _required_https_url("data_api_url")
max_chunks = _bounded_int("max_chunks", minimum=1, maximum=500)
embedding_partitions = _bounded_int(
    "embedding_partitions",
    minimum=1,
    maximum=8,
)
pg_user = str(spark.sql("SELECT current_user() AS user").first()["user"])


# COMMAND ----------

# MAGIC %md
# MAGIC ## OAuth and unembedded Lakebase chunk read

# COMMAND ----------

workspace_client = WorkspaceClient()
database_credential = workspace_client.postgres.generate_database_credential(
    endpoint=endpoint_name
)
database_token = database_credential.token
if not database_token:
    raise RuntimeError("Databricks returned an unusable Lakebase credential.")

jdbc_url = f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_database}"
jdbc_properties = {
    "user": pg_user,
    "password": database_token,
    "driver": "org.postgresql.Driver",
    "sslmode": pg_sslmode,
}

unembedded_chunks_query = """
(
    SELECT article_id, chunk_index, chunk_text
    FROM news_article_chunks
    WHERE embedding IS NULL
) AS unembedded_news_article_chunks
"""

unembedded_chunks_df = spark.read.jdbc(
    url=jdbc_url,
    table=unembedded_chunks_query,
    properties=jdbc_properties,
)
bounded_chunks_df = (
    unembedded_chunks_df.orderBy("article_id", "chunk_index").limit(max_chunks)
)
eligible_chunks = bounded_chunks_df.count()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed sentence-transformer embedding
# MAGIC
# MAGIC Each Spark partition lazily loads exactly one model and encodes each
# MAGIC pandas batch. Embeddings are normalized for cosine retrieval and are
# MAGIC never printed.

# COMMAND ----------

embedded_schema = StructType(
    [
        StructField("article_id", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
        StructField(
            "embedding",
            ArrayType(FloatType(), containsNull=False),
            False,
        ),
        StructField("embedding_model", StringType(), False),
    ]
)


def embed_chunk_partitions(
    batches: Iterator[pd.DataFrame],
) -> Iterator[pd.DataFrame]:
    import os

    cache_root = "/tmp/huggingface"
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_HUB_CACHE", f"{cache_root}/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{cache_root}/transformers")
    for cache_path in (
        cache_root,
        f"{cache_root}/hub",
        f"{cache_root}/transformers",
    ):
        os.makedirs(cache_path, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    from pipelines.embeddings import validate_embedding

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    for batch in batches:
        texts = [str(value) for value in batch["chunk_text"].tolist()]
        encoded = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        output_rows: list[dict[str, object]] = []
        for row, vector in zip(batch.itertuples(index=False), encoded, strict=True):
            validated = validate_embedding(vector)
            output_rows.append(
                {
                    "article_id": str(row.article_id),
                    "chunk_index": int(row.chunk_index),
                    "embedding": list(validated),
                    "embedding_model": EMBEDDING_MODEL_NAME,
                }
            )
        yield pd.DataFrame(
            output_rows,
            columns=[
                "article_id",
                "chunk_index",
                "embedding",
                "embedding_model",
            ],
        )


partition_count = max(1, min(embedding_partitions, eligible_chunks or 1))
embedded_chunks_df = bounded_chunks_df.repartition(partition_count).mapInPandas(
    embed_chunk_partitions,
    schema=embedded_schema,
)
embedded_rows = embedded_chunks_df.collect()
embedded_chunks = len(embedded_rows)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Bounded Lakebase Data API updates
# MAGIC
# MAGIC The RPC performs a parameterized update of the existing composite-key
# MAGIC row. The OAuth header and embedding payload are never displayed.

# COMMAND ----------

persisted_chunks = 0
if embedded_chunks > 0:
    data_api_client_secret = dbutils.secrets.get(
        scope="stock-research-capstone-auth",
        key="client-secret",
    )
    if not data_api_client_secret:
        raise RuntimeError("Data API OAuth client secret is unavailable.")
    workspace_host = workspace_client.config.host
    if not workspace_host:
        raise RuntimeError("Current Databricks workspace host is unavailable.")
    data_api_client = WorkspaceClient(
        host=workspace_host,
        client_id=DATA_API_CLIENT_ID,
        client_secret=data_api_client_secret,
    )
    oauth_headers = data_api_client.config.authenticate()
    if not any(key.lower() == "authorization" for key in oauth_headers):
        raise RuntimeError("Databricks returned unusable Data API authentication.")

    rpc_url = (
        f"{data_api_url}/public/rpc/set_news_article_chunk_embedding"
    )
    session = requests.Session()
    session.headers.update(
        {
            **oauth_headers,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )
    try:
        for row in embedded_rows:
            payload = {
                "p_article_id": str(row.article_id),
                "p_chunk_index": int(row.chunk_index),
                "p_embedding": [float(value) for value in row.embedding],
                "p_embedding_model": str(row.embedding_model),
            }
            try:
                response = session.post(rpc_url, json=payload, timeout=30)
            except requests.Timeout:
                raise RuntimeError(
                    "Lakebase embedding persistence timed out."
                ) from None
            except requests.ConnectionError:
                raise RuntimeError(
                    "Lakebase embedding persistence connection failed."
                ) from None
            except requests.RequestException:
                raise RuntimeError(
                    "Lakebase embedding persistence request failed."
                ) from None
            try:
                response.raise_for_status()
            except requests.HTTPError:
                raise _data_api_http_failure(response) from None
            persisted_chunks += 1
    finally:
        session.close()


# COMMAND ----------

print(f"Eligible unembedded chunks: {eligible_chunks}")
print(f"Embeddings generated: {embedded_chunks}")
print(f"Embeddings persisted: {persisted_chunks}")
print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimensions: {EMBEDDING_DIMENSION}")
