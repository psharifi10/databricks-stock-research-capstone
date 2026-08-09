"""Offline tests for database environment configuration."""

import unittest

from app.config import (
    DEFAULT_APPLICATION_NAME,
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_MASSIVE_API_BASE_URL,
    DEFAULT_MASSIVE_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_PG_PORT,
    DEFAULT_PG_SSLMODE,
    DatabaseConfig,
    DatabaseConfigurationError,
    MassiveConfig,
    MassiveConfigurationError,
)


class DatabaseConfigTests(unittest.TestCase):
    def test_non_sensitive_defaults_do_not_require_credentials(self) -> None:
        config = DatabaseConfig.from_env({})

        self.assertEqual(config.port, DEFAULT_PG_PORT)
        self.assertEqual(config.sslmode, DEFAULT_PG_SSLMODE)
        self.assertEqual(
            config.connect_timeout_seconds,
            DEFAULT_CONNECT_TIMEOUT_SECONDS,
        )
        self.assertEqual(config.application_name, DEFAULT_APPLICATION_NAME)
        self.assertEqual(config.connection_mode, "unconfigured")
        self.assertIsNone(config.endpoint_name)
        self.assertIsNone(config.lakebase_url)

    def test_oauth_fields_are_parsed_without_a_password(self) -> None:
        config = DatabaseConfig.from_env(
            {
                "PGHOST": "db.example.invalid",
                "PGDATABASE": "stock_research",
                "PGUSER": "app_user",
                "PGPORT": "6543",
                "PGSSLMODE": "verify-full",
                "ENDPOINT_NAME": (
                    "projects/example/branches/production/endpoints/primary"
                ),
                "DB_CONNECT_TIMEOUT_SECONDS": "7",
                "DB_APPLICATION_NAME": "capstone-test",
            }
        )

        connection_url, parameters = config.psycopg_connect_parameters()

        self.assertEqual(config.connection_mode, "databricks_oauth")
        self.assertIsNone(connection_url)
        self.assertEqual(
            parameters,
            {
                "host": "db.example.invalid",
                "dbname": "stock_research",
                "user": "app_user",
                "port": 6543,
                "sslmode": "verify-full",
                "connect_timeout": 7,
                "application_name": "capstone-test",
            },
        )

    def test_lakebase_url_is_detected_and_takes_precedence(self) -> None:
        placeholder_url = (
            "postgresql://placeholder:placeholder@db.example.invalid/example"
        )
        config = DatabaseConfig.from_env(
            {
                "LAKEBASE_URL": placeholder_url,
                "PGHOST": "ignored.example.invalid",
                "PGDATABASE": "ignored",
                "PGUSER": "ignored",
                "ENDPOINT_NAME": "ignored",
            }
        )

        connection_url, parameters = config.psycopg_connect_parameters()

        self.assertEqual(config.connection_mode, "lakebase_url")
        self.assertEqual(connection_url, placeholder_url)
        self.assertEqual(
            parameters,
            {
                "connect_timeout": DEFAULT_CONNECT_TIMEOUT_SECONDS,
                "application_name": DEFAULT_APPLICATION_NAME,
            },
        )

    def test_missing_database_configuration_has_clear_error(self) -> None:
        config = DatabaseConfig.from_env({})

        with self.assertRaisesRegex(
            DatabaseConfigurationError,
            "PGHOST, PGDATABASE, PGUSER, ENDPOINT_NAME",
        ):
            config.psycopg_connect_parameters()

    def test_partial_pg_configuration_names_only_missing_fields(self) -> None:
        config = DatabaseConfig.from_env(
            {"PGHOST": "db.example.invalid", "PGUSER": "app_user"}
        )

        with self.assertRaises(DatabaseConfigurationError) as raised:
            config.psycopg_connect_parameters()

        self.assertIn("PGDATABASE", str(raised.exception))
        self.assertIn("ENDPOINT_NAME", str(raised.exception))
        self.assertNotIn("PGHOST,", str(raised.exception))

    def test_oauth_configuration_requires_endpoint_name(self) -> None:
        config = DatabaseConfig.from_env(
            {
                "PGHOST": "db.example.invalid",
                "PGDATABASE": "stock_research",
                "PGUSER": "app_user",
            }
        )

        with self.assertRaisesRegex(
            DatabaseConfigurationError,
            "ENDPOINT_NAME",
        ):
            config.psycopg_connect_parameters()

    def test_invalid_pg_port_is_rejected_without_connecting(self) -> None:
        with self.assertRaisesRegex(DatabaseConfigurationError, "PGPORT"):
            DatabaseConfig.from_env({"PGPORT": "not-a-port"})

    def test_sensitive_values_are_not_in_repr(self) -> None:
        config = DatabaseConfig.from_env(
            {
                "LAKEBASE_URL": "postgresql://user:secret@example.invalid/db",
            }
        )

        representation = repr(config)
        self.assertNotIn("secret", representation)


class MassiveConfigTests(unittest.TestCase):
    def test_non_sensitive_massive_defaults(self) -> None:
        config = MassiveConfig.from_env({})

        self.assertIsNone(config.api_key)
        self.assertEqual(config.base_url, DEFAULT_MASSIVE_API_BASE_URL)
        self.assertEqual(
            config.request_timeout_seconds,
            DEFAULT_MASSIVE_REQUEST_TIMEOUT_SECONDS,
        )

    def test_massive_environment_values_are_parsed(self) -> None:
        config = MassiveConfig.from_env(
            {
                "MASSIVE_API_KEY": "placeholder-key",
                "MASSIVE_API_BASE_URL": "https://massive.example.invalid/",
                "MASSIVE_REQUEST_TIMEOUT": "4.5",
            }
        )

        self.assertEqual(config.require_api_key(), "placeholder-key")
        self.assertEqual(config.base_url, "https://massive.example.invalid")
        self.assertEqual(config.request_timeout_seconds, 4.5)
        self.assertNotIn("placeholder-key", repr(config))

    def test_missing_massive_key_has_clear_error(self) -> None:
        with self.assertRaisesRegex(
            MassiveConfigurationError,
            "MASSIVE_API_KEY",
        ):
            MassiveConfig.from_env({}).require_api_key()


if __name__ == "__main__":
    unittest.main()
