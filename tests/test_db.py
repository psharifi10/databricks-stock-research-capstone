"""Offline tests for Lakebase OAuth connection handling."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.config import DatabaseConfig
from app.db import DatabaseConnectionError, open_connection


ENDPOINT = "projects/example/branches/production/endpoints/primary"


def _oauth_config() -> DatabaseConfig:
    return DatabaseConfig(
        host="db.example.invalid",
        database="stock_research",
        user="app_user",
        endpoint_name=ENDPOINT,
    )


class DatabaseConnectionTests(unittest.TestCase):
    def test_generated_credential_is_used_for_new_connection(self) -> None:
        generated_credential = "short-lived-test-credential"
        workspace_client = Mock()
        workspace_client.postgres.generate_database_credential.return_value = (
            SimpleNamespace(token=generated_credential)
        )
        connection = Mock()
        connect = Mock(return_value=connection)

        result = open_connection(
            _oauth_config(),
            workspace_client_factory=lambda: workspace_client,
            connect_factory=connect,
        )

        self.assertIs(result, connection)
        workspace_client.postgres.generate_database_credential.assert_called_once_with(
            endpoint=ENDPOINT
        )
        self.assertEqual(connect.call_args.kwargs["password"], generated_credential)
        self.assertNotIn(generated_credential, repr(_oauth_config()))

    def test_each_new_connection_generates_a_fresh_credential(self) -> None:
        workspace_client = Mock()
        workspace_client.postgres.generate_database_credential.side_effect = [
            SimpleNamespace(token="first-short-lived-credential"),
            SimpleNamespace(token="second-short-lived-credential"),
        ]
        connect = Mock(side_effect=[Mock(), Mock()])
        client_factory = lambda: workspace_client

        open_connection(
            _oauth_config(),
            workspace_client_factory=client_factory,
            connect_factory=connect,
        )
        open_connection(
            _oauth_config(),
            workspace_client_factory=client_factory,
            connect_factory=connect,
        )

        self.assertEqual(
            workspace_client.postgres.generate_database_credential.call_count,
            2,
        )
        self.assertNotEqual(
            connect.call_args_list[0].kwargs["password"],
            connect.call_args_list[1].kwargs["password"],
        )

    def test_credential_generation_failure_has_safe_error(self) -> None:
        sensitive_detail = "short-lived-sensitive-detail"
        workspace_client = Mock()
        workspace_client.postgres.generate_database_credential.side_effect = (
            RuntimeError(sensitive_detail)
        )
        connect = Mock()

        with self.assertRaises(DatabaseConnectionError) as raised:
            open_connection(
                _oauth_config(),
                workspace_client_factory=lambda: workspace_client,
                connect_factory=connect,
            )

        self.assertNotIn(sensitive_detail, str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)
        connect.assert_not_called()

    def test_connection_failure_does_not_echo_generated_credential(self) -> None:
        generated_credential = "short-lived-sensitive-detail"
        workspace_client = Mock()
        workspace_client.postgres.generate_database_credential.return_value = (
            SimpleNamespace(token=generated_credential)
        )

        def fail_connect(**kwargs):
            raise RuntimeError(f"failed with {kwargs['password']}")

        with self.assertRaises(DatabaseConnectionError) as raised:
            open_connection(
                _oauth_config(),
                workspace_client_factory=lambda: workspace_client,
                connect_factory=fail_connect,
            )

        self.assertNotIn(generated_credential, str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)

    def test_legacy_url_does_not_generate_a_credential(self) -> None:
        config = DatabaseConfig(lakebase_url="postgresql://example.invalid/db")
        client_factory = Mock()
        connection = Mock()
        connect = Mock(return_value=connection)

        result = open_connection(
            config,
            workspace_client_factory=client_factory,
            connect_factory=connect,
        )

        self.assertIs(result, connection)
        client_factory.assert_not_called()
        connect.assert_called_once()

    def test_importing_db_does_not_construct_workspace_client(self) -> None:
        module_path = Path(__file__).resolve().parents[1] / "app" / "db.py"
        spec = importlib.util.spec_from_file_location(
            "offline_db_import_test",
            module_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)

        with patch("databricks.sdk.WorkspaceClient") as workspace_client_class:
            spec.loader.exec_module(module)

        workspace_client_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
