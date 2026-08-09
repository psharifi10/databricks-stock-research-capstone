"""Minimal psycopg 3 connection boundary for Lakebase/PostgreSQL."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from app.config import DatabaseConfig, load_database_config


class DatabaseConnectionError(psycopg.Error):
    """A safe connection error that does not expose credential internals."""


WorkspaceClientFactory = Callable[[], Any]
ConnectFactory = Callable[..., Connection[dict[str, Any]]]


def _default_workspace_client_factory() -> Any:
    """Construct the SDK client lazily so imports never require authentication."""

    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def open_connection(
    config: DatabaseConfig | None = None,
    *,
    workspace_client_factory: WorkspaceClientFactory | None = None,
    connect_factory: ConnectFactory | None = None,
) -> Connection[dict[str, Any]]:
    """Open one database connection after validating configuration.

    No connection is attempted at module import time. Future callers must use
    psycopg placeholders for values; SQL identifiers must remain static or be
    selected from an explicitly approved set.
    """

    resolved = config or load_database_config()
    connection_url, parameters = resolved.psycopg_connect_parameters()
    connect = connect_factory or psycopg.connect

    try:
        if connection_url:
            return connect(connection_url, row_factory=dict_row, **parameters)

        client_factory = (
            workspace_client_factory or _default_workspace_client_factory
        )
        workspace_client = client_factory()
        credential = workspace_client.postgres.generate_database_credential(
            endpoint=resolved.endpoint_name
        )
        token = getattr(credential, "token", None)
        if not token:
            raise DatabaseConnectionError(
                "Databricks returned an unusable database credential."
            )
        return connect(row_factory=dict_row, password=token, **parameters)
    except DatabaseConnectionError:
        raise
    except Exception:
        raise DatabaseConnectionError(
            "Unable to establish the Lakebase database connection."
        ) from None


@contextmanager
def database_connection(
    config: DatabaseConfig | None = None,
    *,
    workspace_client_factory: WorkspaceClientFactory | None = None,
    connect_factory: ConnectFactory | None = None,
) -> Iterator[Connection[dict[str, Any]]]:
    """Yield a transaction-scoped connection.

    Successful work is committed, exceptions trigger a rollback, and the
    connection is always closed. Repository and business queries belong in
    later service modules, not in this infrastructure boundary.
    """

    connection = open_connection(
        config,
        workspace_client_factory=workspace_client_factory,
        connect_factory=connect_factory,
    )
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.close()
