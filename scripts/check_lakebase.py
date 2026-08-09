"""Safely validate the configured Lakebase connection."""

from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg

from app.config import ConfigurationError
from app.db import DatabaseConnectionError, database_connection


def get_connection_details(connection: Any) -> dict[str, str]:
    """Return only the safe identity fields from a harmless database query."""

    with connection.cursor() as cursor:
        cursor.execute("SELECT current_user, current_database(), version();")
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError("The connection check returned no result.")
    return {
        "current_user": str(row["current_user"]),
        "current_database": str(row["current_database"]),
    }


def main() -> int:
    try:
        with database_connection() as connection:
            details = get_connection_details(connection)
    except ConfigurationError as error:
        print(f"Lakebase connection check failed: {error}", file=sys.stderr)
        return 1
    except (DatabaseConnectionError, psycopg.Error, RuntimeError):
        print("Lakebase connection check failed safely.", file=sys.stderr)
        return 1

    print(f"current_user: {details['current_user']}")
    print(f"current_database: {details['current_database']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
