"""Static frontend asset loading for the shared FastMCP Databricks App."""

from __future__ import annotations

from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_ASSETS = {
    "index.html": "text/html",
    "app.js": "text/javascript",
    "styles.css": "text/css",
}


def load_static_asset(filename: str) -> tuple[str, str]:
    """Return a known UTF-8 asset and its media type without path traversal."""

    try:
        media_type = STATIC_ASSETS[filename]
    except KeyError as error:
        raise ValueError("Unknown frontend asset.") from error
    return (STATIC_DIR / filename).read_text(encoding="utf-8"), media_type
