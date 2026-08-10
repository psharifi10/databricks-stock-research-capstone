"""Tests for shared domain validation."""

import unittest

from app.validation import (
    ValidationError,
    normalize_bounded_text,
    normalize_email,
    normalize_ticker,
)


class TickerValidationTests(unittest.TestCase):
    def test_lowercase_and_whitespace_are_normalized(self) -> None:
        self.assertEqual(normalize_ticker("  aapl  "), "AAPL")

    def test_common_share_class_forms_are_supported(self) -> None:
        self.assertEqual(normalize_ticker("brk.b"), "BRK.B")
        self.assertEqual(normalize_ticker("brk-b"), "BRK-B")

    def test_malformed_tickers_are_rejected(self) -> None:
        for value in ("", "../AAPL", "AAPL?key=value", "TOO-LONG-SUFFIX", 123):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    normalize_ticker(value)  # type: ignore[arg-type]


class EmailValidationTests(unittest.TestCase):
    def test_email_is_trimmed_and_lowercased(self) -> None:
        self.assertEqual(normalize_email(" User@Example.com "), "user@example.com")

    def test_bad_email_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            normalize_email("not an email")


class BoundedTextValidationTests(unittest.TestCase):
    def test_surrounding_whitespace_is_trimmed_without_rewriting_content(self) -> None:
        self.assertEqual(
            normalize_bounded_text(
                "  First line\n  second line  ",
                field_name="Note",
                maximum=100,
            ),
            "First line\n  second line",
        )

    def test_non_string_blank_and_oversized_text_are_rejected(self) -> None:
        for value in (None, "   ", "abcd"):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    normalize_bounded_text(
                        value,  # type: ignore[arg-type]
                        field_name="Note",
                        maximum=3,
                    )


if __name__ == "__main__":
    unittest.main()
