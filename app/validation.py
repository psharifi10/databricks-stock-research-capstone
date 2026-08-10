"""Shared validation and normalization for application-domain values."""

from __future__ import annotations

import re


_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{0,4}(?:[.-][A-Z0-9]{1,2})?$")
MAX_SEMANTIC_SEARCH_RESULTS = 20


class ValidationError(ValueError):
    """Raised when a caller supplies an invalid domain value."""


def normalize_ticker(value: str) -> str:
    """Return a normalized US equity ticker or raise ``ValidationError``.

    The deliberately small grammar supports common symbols and share-class
    forms such as ``BRK.B`` and ``BRK-B`` without accepting arbitrary path or
    query-string characters.
    """

    if not isinstance(value, str):
        raise ValidationError("Ticker must be a string.")
    ticker = value.strip().upper()
    if not ticker or not _TICKER_PATTERN.fullmatch(ticker):
        raise ValidationError(
            "Ticker must be 1-5 letters/digits beginning with a letter, "
            "optionally followed by a dot or hyphen share-class suffix."
        )
    return ticker


def normalize_email(value: str) -> str:
    """Normalize a user email while rejecting clearly malformed input."""

    if not isinstance(value, str):
        raise ValidationError("Email must be a string.")
    email = value.strip().lower()
    if (
        not email
        or any(character.isspace() for character in email)
        or email.count("@") != 1
        or not all(email.split("@"))
    ):
        raise ValidationError("A valid email address is required.")
    return email


def normalize_top_k(value: int) -> int:
    """Validate a positive result count and clamp it to the safe search bound."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError("top_k must be a positive integer.")
    return min(value, MAX_SEMANTIC_SEARCH_RESULTS)
