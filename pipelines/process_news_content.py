# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4A: process persisted financial-news content
# MAGIC
# MAGIC This bounded Spark pipeline reads already-ingested `news_articles` from
# MAGIC Lakebase, skips article IDs that already have chunks, performs distributed
# MAGIC webpage extraction with metadata fallback, creates deterministic chunks,
# MAGIC and persists them through bounded JDBC transactions.
# MAGIC
# MAGIC It does not call Massive and it does not generate embeddings.

# COMMAND ----------

# MAGIC %pip install "databricks-sdk>=0.89,<1.0" "trafilatura>=2.1,<3.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from collections import defaultdict
from collections.abc import Iterator
import pandas as pd

from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Non-secret run parameters
# MAGIC
# MAGIC `PGUSER` is resolved from the active Databricks identity. There is no
# MAGIC password widget: Lakebase credentials are generated through Databricks
# MAGIC OAuth and remain in memory only.

# COMMAND ----------

dbutils.widgets.text("endpoint_name", "", "Lakebase endpoint resource name")
dbutils.widgets.text("pg_host", "", "Lakebase PostgreSQL host")
dbutils.widgets.text("pg_database", "databricks_postgres", "PostgreSQL database")
dbutils.widgets.text("pg_port", "5432", "PostgreSQL port")
dbutils.widgets.text("pg_sslmode", "require", "PostgreSQL SSL mode")
dbutils.widgets.text("max_articles", "25", "Maximum articles in this run")
dbutils.widgets.text("fetch_partitions", "4", "HTTP extraction partitions")
dbutils.widgets.text("chunk_size", "1200", "Chunk size in characters")
dbutils.widgets.text("chunk_overlap", "200", "Chunk overlap in characters")
dbutils.widgets.text("request_timeout", "15", "HTTP request timeout in seconds")


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


endpoint_name = _required_widget("endpoint_name")
pg_host = _required_widget("pg_host")
pg_database = _required_widget("pg_database")
pg_port = _bounded_int("pg_port", minimum=1, maximum=65535)
pg_sslmode = _required_widget("pg_sslmode")
max_articles = _bounded_int("max_articles", minimum=1, maximum=100)
fetch_partitions = _bounded_int("fetch_partitions", minimum=1, maximum=16)
chunk_size = _bounded_int("chunk_size", minimum=100, maximum=10000)
chunk_overlap = _bounded_int(
    "chunk_overlap",
    minimum=0,
    maximum=chunk_size - 1,
)
request_timeout = _bounded_int("request_timeout", minimum=1, maximum=60)

pg_user = str(spark.sql("SELECT current_user() AS user").first()["user"])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Lakebase OAuth and Spark JDBC reads
# MAGIC
# MAGIC The short-lived token is passed only as the JDBC password property. It
# MAGIC is never printed or written to disk.

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

articles_query = """
(
    SELECT id AS article_id, title, description, article_url, published_at
    FROM news_articles
) AS source_news_articles
"""
processed_article_ids_query = """
(
    SELECT DISTINCT article_id
    FROM news_article_chunks
) AS processed_news_article_ids
"""

articles_df = spark.read.jdbc(
    url=jdbc_url,
    table=articles_query,
    properties=jdbc_properties,
)
processed_article_ids_df = spark.read.jdbc(
    url=jdbc_url,
    table=processed_article_ids_query,
    properties=jdbc_properties,
)

has_url = F.length(F.trim(F.coalesce(F.col("article_url"), F.lit("")))) > 0
has_metadata = (
    F.length(F.trim(F.coalesce(F.col("title"), F.lit("")))) > 0
) | (
    F.length(F.trim(F.coalesce(F.col("description"), F.lit("")))) > 0
)

eligible_articles_df = (
    articles_df.join(processed_article_ids_df, "article_id", "left_anti")
    .filter(has_url | has_metadata)
    .orderBy(F.col("published_at").desc_nulls_last())
    .limit(max_articles)
)
eligible_count = eligible_articles_df.count()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed article extraction
# MAGIC
# MAGIC Each Spark partition creates one HTTP session. A blocked, inaccessible,
# MAGIC or insufficient publisher page becomes a metadata fallback; it does not
# MAGIC fail the remaining partition. No body text is printed.

# COMMAND ----------

extracted_schema = StructType(
    [
        StructField("article_id", StringType(), False),
        StructField("title", StringType(), True),
        StructField("article_url", StringType(), True),
        StructField("extraction_source", StringType(), False),
        StructField("extracted_text", StringType(), False),
    ]
)


def extract_article_partitions(
    batches: Iterator[pd.DataFrame],
) -> Iterator[pd.DataFrame]:
    from pipelines.article_processing import (
        create_article_session,
        resolve_article_content,
    )

    session = create_article_session()
    try:
        for batch in batches:
            output_rows: list[dict[str, object]] = []
            for row in batch.itertuples(index=False):
                safe_title = row.title if isinstance(row.title, str) else None
                safe_url = row.article_url if isinstance(row.article_url, str) else None
                content = resolve_article_content(
                    session,
                    article_url=safe_url,
                    title=safe_title,
                    description=row.description,
                    request_timeout=request_timeout,
                )
                if content is None:
                    continue
                output_rows.append(
                    {
                        "article_id": str(row.article_id),
                        "title": safe_title,
                        "article_url": safe_url,
                        "extraction_source": content.extraction_source,
                        "extracted_text": content.text,
                    }
                )
            yield pd.DataFrame(
                output_rows,
                columns=[
                    "article_id",
                    "title",
                    "article_url",
                    "extraction_source",
                    "extracted_text",
                ],
            )
    finally:
        session.close()


partition_count = max(1, min(fetch_partitions, eligible_count or 1))
extracted_articles_df = (
    eligible_articles_df.repartition(partition_count)
    .mapInPandas(extract_article_partitions, schema=extracted_schema)
    .cache()
)

extracted_count = extracted_articles_df.count()
article_body_count = extracted_articles_df.filter(
    F.col("extraction_source") == "article_body"
).count()
metadata_fallback_count = extracted_articles_df.filter(
    F.col("extraction_source") == "metadata_fallback"
).count()
skipped_count = eligible_count - extracted_count


# COMMAND ----------

# MAGIC %md
# MAGIC ## Deterministic Spark chunk DataFrame

# COMMAND ----------

chunk_schema = StructType(
    [
        StructField("article_id", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
        StructField("chunk_text", StringType(), False),
    ]
)


def chunk_article_partitions(
    batches: Iterator[pd.DataFrame],
) -> Iterator[pd.DataFrame]:
    from pipelines.article_processing import chunk_text

    for batch in batches:
        output_rows: list[dict[str, object]] = []
        for row in batch.itertuples(index=False):
            chunks = chunk_text(
                row.extracted_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for chunk_index, text in enumerate(chunks):
                output_rows.append(
                    {
                        "article_id": str(row.article_id),
                        "chunk_index": chunk_index,
                        "chunk_text": text,
                    }
                )
        yield pd.DataFrame(
            output_rows,
            columns=["article_id", "chunk_index", "chunk_text"],
        )


chunks_df = extracted_articles_df.mapInPandas(
    chunk_article_partitions,
    schema=chunk_schema,
).cache()
total_chunks = chunks_df.count()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Bounded driver-side persistence
# MAGIC
# MAGIC Extraction and chunk generation occur through Spark. Because
# MAGIC `max_articles` strictly bounds this educational workload, final chunks
# MAGIC are collected to the driver, grouped by article, and atomically replaced
# MAGIC through the JVM PostgreSQL JDBC driver. This avoids native Python
# MAGIC PostgreSQL dependencies in the Databricks runtime while preventing stale
# MAGIC extra chunks.

# COMMAND ----------

chunks_by_article: dict[str, dict[int, str]] = defaultdict(dict)
for row in chunks_df.collect():
    chunks_by_article[str(row.article_id)][int(row.chunk_index)] = str(row.chunk_text)


def replace_article_chunks_via_jdbc(
    article_id: str,
    indexed_chunks: dict[int, str],
) -> None:
    """Atomically replace one article's chunks with prepared JDBC statements."""

    connection = None
    delete_statement = None
    insert_statement = None
    try:
        jvm = spark._sc._gateway.jvm
        connection_properties = jvm.java.util.Properties()
        connection_properties.setProperty("user", pg_user)
        connection_properties.setProperty("password", database_token)
        connection_properties.setProperty("sslmode", pg_sslmode)
        connection = jvm.java.sql.DriverManager.getConnection(
            jdbc_url,
            connection_properties,
        )
        connection.setAutoCommit(False)

        delete_statement = connection.prepareStatement(
            "DELETE FROM news_article_chunks WHERE article_id = ?"
        )
        delete_statement.setString(1, article_id)
        delete_statement.executeUpdate()

        if indexed_chunks:
            insert_statement = connection.prepareStatement(
                """
                INSERT INTO news_article_chunks
                    (article_id, chunk_index, chunk_text)
                VALUES (?, ?, ?)
                """
            )
            for chunk_index in sorted(indexed_chunks):
                insert_statement.setString(1, article_id)
                insert_statement.setInt(2, chunk_index)
                insert_statement.setString(3, indexed_chunks[chunk_index])
                insert_statement.addBatch()
            insert_statement.executeBatch()

        connection.commit()
    except Exception:
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass
        raise RuntimeError(
            f"Failed to persist chunks for article {article_id}."
        ) from None
    finally:
        for resource in (insert_statement, delete_statement, connection):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass

articles_persisted = 0
for article_id, indexed_chunks in chunks_by_article.items():
    ordered_indexes = sorted(indexed_chunks)
    if ordered_indexes != list(range(len(ordered_indexes))):
        raise RuntimeError("Generated chunk indexes are not contiguous.")
    replace_article_chunks_via_jdbc(article_id, indexed_chunks)
    articles_persisted += 1


# COMMAND ----------

print(f"Eligible articles: {eligible_count}")
print(f"Successfully fetched article bodies: {article_body_count}")
print(f"Metadata fallbacks: {metadata_fallback_count}")
print(f"Skipped unusable articles: {skipped_count}")
print(f"Total chunks generated: {total_chunks}")
print(f"Articles persisted: {articles_persisted}")
