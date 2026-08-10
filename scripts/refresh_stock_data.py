"""Refresh one ticker through the normal Massive/service/repository path."""

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ConfigurationError
from app.massive_client import MassiveClient, MassiveClientError
from app.repositories import RepositoryError, StockRepository
from app.services import StockResearchService
from app.validation import ValidationError, normalize_ticker


NEWS_LIMIT = 5
VALIDATION_MAX_PAGES = 1


def recent_completed_price_range(as_of: date | None = None) -> tuple[date, date]:
    """Return a short range ending on the last completed weekday."""

    end_date = (as_of or date.today()) - timedelta(days=1)
    while end_date.weekday() >= 5:
        end_date -= timedelta(days=1)
    return end_date - timedelta(days=10), end_date


def refresh_ticker(
    service: StockResearchService,
    ticker: str,
    *,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Run the three bounded refresh operations through the service layer."""

    symbol = normalize_ticker(ticker)
    start_date, end_date = recent_completed_price_range(as_of)
    company = service.refresh_company(symbol)
    prices = service.refresh_price_history(
        symbol,
        start_date,
        end_date,
        max_pages=VALIDATION_MAX_PAGES,
    )
    news = service.refresh_news(
        symbol,
        limit=NEWS_LIMIT,
        published_after=start_date,
        max_pages=VALIDATION_MAX_PAGES,
    )
    return {
        "ticker": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "company": company,
        "prices": prices,
        "news": news,
    }


def _safe_line(value: Any, *, maximum: int = 160) -> str:
    normalized = " ".join(str(value or "").split())
    return normalized[:maximum]


def print_refresh_summary(result: dict[str, Any]) -> None:
    company = result["company"]
    prices = result["prices"]
    news = result["news"]

    print("Company refreshed:")
    print(f"  ticker: {result['ticker']}")
    print(f"  name: {_safe_line(company.get('name'))}")
    print(f"  exchange: {_safe_line(company.get('exchange'))}")
    print("Prices refreshed:")
    print(f"  ticker: {result['ticker']}")
    print(f"  date range: {result['start_date']} to {result['end_date']}")
    print(f"  returned bars: {len(prices)}")
    print("News refreshed:")
    print(f"  ticker: {result['ticker']}")
    print(f"  returned articles: {len(news)}")
    for article in news[:3]:
        print(
            "  - "
            f"{_safe_line(article.get('published_at'))}: "
            f"{_safe_line(article.get('title'))}"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh bounded stock data through the application service layer."
    )
    parser.add_argument("ticker", nargs="?", default="AAPL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    client: MassiveClient | None = None
    try:
        symbol = normalize_ticker(args.ticker)
        client = MassiveClient()
        result = refresh_ticker(
            StockResearchService(client, StockRepository()),
            symbol,
        )
    except (ConfigurationError, ValidationError, MassiveClientError) as error:
        print(f"Stock refresh failed: {error}", file=sys.stderr)
        return 1
    except RepositoryError:
        print("Stock refresh failed during database persistence.", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()

    print_refresh_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
