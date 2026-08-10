"""Apply the idempotent core schema to the configured Lakebase database."""

from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg

from app.config import ConfigurationError
from app.db import DatabaseConnectionError, database_connection


SCHEMA_PATHS = (
    PROJECT_ROOT / "sql" / "001_core_schema.sql",
    PROJECT_ROOT / "sql" / "002_chunk_embeddings.sql",
)


def read_schema(paths: Sequence[Path] = SCHEMA_PATHS) -> str:
    """Read ordered idempotent migrations without duplicating SQL in Python."""

    return "\n\n".join(path.read_text(encoding="utf-8") for path in paths)


def apply_schema(connection: Any, schema: str) -> None:
    """Execute the schema inside the caller-managed transaction."""

    with connection.cursor() as cursor:
        cursor.execute(schema)


def main() -> int:
    try:
        schema = read_schema()
        with database_connection() as connection:
            apply_schema(connection, schema)
    except OSError:
        print(
            "Schema application failed: unable to read the schema file.",
            file=sys.stderr,
        )
        return 1
    except ConfigurationError as error:
        print(f"Schema application failed: {error}", file=sys.stderr)
        return 1
    except (DatabaseConnectionError, psycopg.Error):
        print("Schema application failed safely.", file=sys.stderr)
        return 1

    print("Applied the ordered SQL migrations successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
