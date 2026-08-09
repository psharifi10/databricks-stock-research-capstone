"""Verify that the public schema contains exactly the ten MVP tables."""

from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg

from app.config import ConfigurationError
from app.db import DatabaseConnectionError, database_connection


EXPECTED_TABLES = frozenset(
    {
        "users",
        "watchlists",
        "watchlist_tickers",
        "companies",
        "price_snapshots",
        "news_articles",
        "news_article_tickers",
        "news_article_chunks",
        "research_notes",
        "analysis_reports",
    }
)


def get_public_tables(connection: Any) -> set[str]:
    """Read public base-table names through a parameterized metadata query."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s AND table_type = %s
            """,
            ("public", "BASE TABLE"),
        )
        return {str(row["table_name"]) for row in cursor.fetchall()}


def schema_differences(table_names: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return sorted missing and unexpected table names."""

    actual = set(table_names)
    return sorted(EXPECTED_TABLES - actual), sorted(actual - EXPECTED_TABLES)


def main() -> int:
    try:
        with database_connection() as connection:
            tables = get_public_tables(connection)
    except ConfigurationError as error:
        print(f"Schema verification failed: {error}", file=sys.stderr)
        return 1
    except (DatabaseConnectionError, psycopg.Error):
        print("Schema verification failed safely.", file=sys.stderr)
        return 1

    missing, unexpected = schema_differences(tables)
    if missing:
        print(f"Missing tables: {', '.join(missing)}", file=sys.stderr)
    if unexpected:
        print(f"Unexpected public tables: {', '.join(unexpected)}", file=sys.stderr)
    if missing or unexpected:
        return 1

    print("Schema verification passed: all 10 expected public tables exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
