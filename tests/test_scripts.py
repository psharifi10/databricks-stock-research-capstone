"""Offline tests for deterministic Lakebase script helpers."""

import unittest

from scripts.apply_schema import apply_schema, read_schema
from scripts.check_lakebase import get_connection_details
from scripts.verify_schema import EXPECTED_TABLES, schema_differences


class _Cursor:
    def __init__(self, *, row=None) -> None:
        self.row = row
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def execute(self, statement, parameters=None) -> None:
        self.executions.append((statement, parameters))

    def fetchone(self):
        return self.row


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _Cursor:
        return self._cursor


class LakebaseScriptTests(unittest.TestCase):
    def test_connection_details_exclude_database_version(self) -> None:
        cursor = _Cursor(
            row={
                "current_user": "test_user",
                "current_database": "test_database",
                "version": "PostgreSQL test version",
            }
        )

        details = get_connection_details(_Connection(cursor))

        self.assertEqual(
            details,
            {
                "current_user": "test_user",
                "current_database": "test_database",
            },
        )
        self.assertIn("version()", cursor.executions[0][0])

    def test_apply_schema_uses_the_canonical_sql_file(self) -> None:
        schema = read_schema()
        cursor = _Cursor()

        apply_schema(_Connection(cursor), schema)

        self.assertIn("CREATE TABLE IF NOT EXISTS users", schema)
        self.assertEqual(cursor.executions, [(schema, None)])

    def test_schema_differences_require_the_exact_table_set(self) -> None:
        missing, unexpected = schema_differences(EXPECTED_TABLES)
        self.assertEqual((missing, unexpected), ([], []))

        missing, unexpected = schema_differences(
            (EXPECTED_TABLES - {"users"}) | {"other_table"}
        )
        self.assertEqual(missing, ["users"])
        self.assertEqual(unexpected, ["other_table"])


if __name__ == "__main__":
    unittest.main()
