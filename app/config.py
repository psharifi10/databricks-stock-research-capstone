"""Environment-driven configuration for the Lakebase connection boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Mapping


DEFAULT_PG_PORT = 5432
DEFAULT_PG_SSLMODE = "require"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_APPLICATION_NAME = "databricks-stock-research-capstone"


class DatabaseConfigurationError(RuntimeError):
    """Raised when database settings are missing or invalid."""


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Database settings resolved without opening a connection.

    Sensitive fields are excluded from the generated representation to reduce
    the chance of credentials being exposed through logs or exceptions.
    """

    lakebase_url: str | None = field(default=None, repr=False)
    host: str | None = None
    database: str | None = None
    user: str | None = None
    password: str | None = field(default=None, repr=False)
    port: int = DEFAULT_PG_PORT
    sslmode: str = DEFAULT_PG_SSLMODE
    connect_timeout_seconds: int = DEFAULT_CONNECT_TIMEOUT_SECONDS
    application_name: str = DEFAULT_APPLICATION_NAME

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DatabaseConfig":
        """Build configuration from standard PostgreSQL environment names."""

        values = os.environ if env is None else env
        return cls(
            lakebase_url=_optional_value(values, "LAKEBASE_URL"),
            host=_optional_value(values, "PGHOST"),
            database=_optional_value(values, "PGDATABASE"),
            user=_optional_value(values, "PGUSER"),
            password=_optional_value(values, "PGPASSWORD"),
            port=_positive_int(
                values,
                "PGPORT",
                DEFAULT_PG_PORT,
                maximum=65535,
            ),
            sslmode=_optional_value(values, "PGSSLMODE") or DEFAULT_PG_SSLMODE,
            connect_timeout_seconds=_positive_int(
                values,
                "DB_CONNECT_TIMEOUT_SECONDS",
                DEFAULT_CONNECT_TIMEOUT_SECONDS,
            ),
            application_name=(
                _optional_value(values, "DB_APPLICATION_NAME")
                or DEFAULT_APPLICATION_NAME
            ),
        )

    @property
    def connection_mode(self) -> str:
        """Describe the selected connection source without exposing secrets."""

        if self.lakebase_url:
            return "lakebase_url"
        if any((self.host, self.database, self.user, self.password)):
            return "pg_environment"
        return "unconfigured"

    def psycopg_connect_parameters(self) -> tuple[str | None, dict[str, object]]:
        """Return validated arguments for ``psycopg.connect``.

        A legacy/local URL takes precedence when provided. Otherwise PGHOST,
        PGDATABASE, and PGUSER are required. PGPASSWORD is optional because the
        production credential mechanism will be finalized for the actual
        Databricks PostgreSQL App resource.
        """

        common: dict[str, object] = {
            "connect_timeout": self.connect_timeout_seconds,
            "application_name": self.application_name,
        }
        if self.lakebase_url:
            return self.lakebase_url, common

        required = {
            "PGHOST": self.host,
            "PGDATABASE": self.database,
            "PGUSER": self.user,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            names = ", ".join(missing)
            raise DatabaseConfigurationError(
                "Database configuration is incomplete. Set LAKEBASE_URL for "
                "local/legacy use, or provide the required PostgreSQL "
                f"environment variables: {names}."
            )

        parameters: dict[str, object] = {
            **common,
            "host": self.host,
            "dbname": self.database,
            "user": self.user,
            "port": self.port,
            "sslmode": self.sslmode,
        }
        if self.password:
            parameters["password"] = self.password
        return None, parameters


def load_database_config() -> DatabaseConfig:
    """Load database configuration from the current process environment."""

    return DatabaseConfig.from_env()


def _optional_value(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _positive_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    maximum: int | None = None,
) -> int:
    raw_value = _optional_value(env, name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise DatabaseConfigurationError(
            f"{name} must be a positive integer; received {raw_value!r}."
        ) from error
    if value <= 0 or (maximum is not None and value > maximum):
        upper_bound = f" no greater than {maximum}" if maximum else ""
        raise DatabaseConfigurationError(
            f"{name} must be a positive integer{upper_bound}; received {value}."
        )
    return value
