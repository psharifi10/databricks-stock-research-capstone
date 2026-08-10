"""Summarize and verify persisted stock data without calling Massive."""

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg

from app.config import ConfigurationError
from app.db import DatabaseConnectionError, database_connection
from app.validation import ValidationError, normalize_ticker


TITLE_LIMIT = 3


def get_stock_summary(connection: Any, ticker: str) -> dict[str, Any]:
    """Read parameterized company, price, and news summaries for one ticker."""

    symbol = normalize_ticker(ticker)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT ticker, name, exchange, updated_at
            FROM companies
            WHERE ticker = %s
            """,
            (symbol,),
        )
        companies = [dict(row) for row in cursor.fetchall()]

        cursor.execute(
            """
            SELECT COUNT(*) AS row_count,
                   MIN(price_date) AS minimum_price_date,
                   MAX(price_date) AS maximum_price_date
            FROM price_snapshots
            WHERE ticker = %s
            """,
            (symbol,),
        )
        prices = dict(cursor.fetchone())

        cursor.execute(
            """
            SELECT COUNT(*) AS association_count,
                   COUNT(DISTINCT nat.article_id) AS distinct_article_count,
                   MAX(na.published_at) AS newest_published_at
            FROM news_article_tickers AS nat
            JOIN news_articles AS na ON na.id = nat.article_id
            WHERE nat.ticker = %s
            """,
            (symbol,),
        )
        news = dict(cursor.fetchone())

        cursor.execute(
            """
            SELECT na.title, na.published_at, nat.sentiment,
                   ARRAY(
                       SELECT related.ticker
                       FROM news_article_tickers AS related
                       WHERE related.article_id = na.id
                       ORDER BY related.ticker
                   ) AS related_tickers
            FROM news_article_tickers AS nat
            JOIN news_articles AS na ON na.id = nat.article_id
            WHERE nat.ticker = %s
            ORDER BY na.published_at DESC NULLS LAST
            LIMIT %s
            """,
            (symbol, TITLE_LIMIT),
        )
        articles = [dict(row) for row in cursor.fetchall()]

    return {
        "ticker": symbol,
        "company_count": len(companies),
        "company": companies[0] if len(companies) == 1 else None,
        "price_count": int(prices["row_count"]),
        "minimum_price_date": prices["minimum_price_date"],
        "maximum_price_date": prices["maximum_price_date"],
        "distinct_news_articles": int(news["distinct_article_count"]),
        "news_associations": int(news["association_count"]),
        "newest_published_at": news["newest_published_at"],
        "articles": articles,
    }


def validation_errors(summary: dict[str, Any]) -> list[str]:
    """Return deterministic relational/data-presence failures."""

    errors: list[str] = []
    if summary["company_count"] != 1:
        errors.append("expected exactly one company row")
    if summary["price_count"] < 1:
        errors.append("expected at least one price row")
    if summary["distinct_news_articles"] < 1:
        errors.append("expected at least one associated news article")
    if summary["news_associations"] != summary["distinct_news_articles"]:
        errors.append("ticker associations contain duplicate article relationships")
    return errors


def _safe_line(value: Any, *, maximum: int = 160) -> str:
    normalized = " ".join(str(value or "").split())
    return normalized[:maximum]


def print_stock_summary(summary: dict[str, Any]) -> None:
    print(f"Ticker: {summary['ticker']}")
    print(f"Company rows: {summary['company_count']}")
    company = summary["company"]
    if company:
        print(
            "Company metadata: "
            f"{_safe_line(company['name'])} | "
            f"{_safe_line(company['exchange'])} | "
            f"updated {company['updated_at']}"
        )
    print(f"Price rows: {summary['price_count']}")
    print(
        "Price range: "
        f"{summary['minimum_price_date']} to {summary['maximum_price_date']}"
    )
    print(f"Distinct associated news articles: {summary['distinct_news_articles']}")
    print(f"Ticker/article associations: {summary['news_associations']}")
    print(f"Newest published_at: {summary['newest_published_at']}")
    for article in summary["articles"]:
        tickers = ",".join(article["related_tickers"] or [])
        print(
            "News: "
            f"{_safe_line(article['published_at'])} | "
            f"{_safe_line(article['sentiment']) or 'no sentiment'} | "
            f"tickers={_safe_line(tickers)} | "
            f"{_safe_line(article['title'])}"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify persisted stock data without making Massive calls."
    )
    parser.add_argument("ticker", nargs="?", default="AAPL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        symbol = normalize_ticker(args.ticker)
        with database_connection() as connection:
            summary = get_stock_summary(connection, symbol)
    except (ConfigurationError, ValidationError) as error:
        print(f"Stock data verification failed: {error}", file=sys.stderr)
        return 1
    except (DatabaseConnectionError, psycopg.Error):
        print("Stock data verification failed safely.", file=sys.stderr)
        return 1

    print_stock_summary(summary)
    errors = validation_errors(summary)
    if errors:
        for error in errors:
            print(f"Validation error: {error}", file=sys.stderr)
        return 1
    print("Stock data verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
