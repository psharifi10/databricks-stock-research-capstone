"""Minimal psycopg 3 connection boundary for Lakebase/PostgreSQL."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from app.config import DatabaseConfig, load_database_config


def open_connection(
    config: DatabaseConfig | None = None,
) -> Connection[dict[str, Any]]:
    """Open one database connection after validating configuration.

    No connection is attempted at module import time. Future callers must use
    psycopg placeholders for values; SQL identifiers must remain static or be
    selected from an explicitly approved set.
    """

    resolved = config or load_database_config()
    connection_url, parameters = resolved.psycopg_connect_parameters()
    if connection_url:
        return psycopg.connect(connection_url, row_factory=dict_row, **parameters)
    return psycopg.connect(row_factory=dict_row, **parameters)


@contextmanager
def database_connection(
    config: DatabaseConfig | None = None,
) -> Iterator[Connection[dict[str, Any]]]:
    """Yield a transaction-scoped connection.

    Successful work is committed, exceptions trigger a rollback, and the
    connection is always closed. Repository and business queries belong in
    later service modules, not in this infrastructure boundary.
    """

    connection = open_connection(config)
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.close()
