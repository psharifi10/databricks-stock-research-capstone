"""Offline contract tests for the Massive REST client."""

from datetime import date, datetime, timezone
import unittest
from unittest.mock import MagicMock

import requests

from app.config import MassiveConfig
from app.massive_client import MassiveClient, MassiveClientError


def _response(payload, *, request_error=None):
    response = MagicMock()
    response.raise_for_status.side_effect = request_error
    response.json.return_value = payload
    return response


class MassiveClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = MagicMock(spec=requests.Session)
        self.session.headers = {}
        self.config = MassiveConfig(
            api_key="unit-test-key",
            base_url="https://api.massive.com",
            request_timeout_seconds=12.5,
        )
        self.client = MassiveClient(self.config, session=self.session)

    def test_company_endpoint_and_normalization(self) -> None:
        self.session.get.return_value = _response(
            {
                "status": "OK",
                "results": {
                    "ticker": "AAPL",
                    "name": "Apple Inc.",
                    "description": "Consumer technology company",
                    "market_cap": 3000000000000,
                    "market": "stocks",
                    "primary_exchange": "XNAS",
                    "type": "CS",
                    "active": True,
                    "list_date": "1980-12-12",
                    "sic_code": "3571",
                    "sic_description": "Electronic Computers",
                    "homepage_url": "https://www.apple.com",
                    "currency_name": "usd",
                    "locale": "us",
                },
            }
        )

        company = self.client.get_company_overview(" aapl ")

        self.session.get.assert_called_once_with(
            "https://api.massive.com/v3/reference/tickers/AAPL",
            params=None,
            timeout=12.5,
        )
        self.assertEqual(
            self.session.headers["Authorization"],
            "Bearer unit-test-key",
        )
        self.assertEqual(company["ticker"], "AAPL")
        self.assertEqual(company["exchange"], "XNAS")
        self.assertEqual(company["security_type"], "CS")
        self.assertEqual(company["list_date"], date(1980, 12, 12))
        self.assertEqual(company["industry"], "Electronic Computers")
        self.assertEqual(company["raw_source_payload"]["ticker"], "AAPL")

    def test_price_endpoint_parameters_and_ohlc_normalization(self) -> None:
        self.session.get.return_value = _response(
            {
                "status": "OK",
                "results": [
                    {
                        "o": 74.06,
                        "h": 75.15,
                        "l": 73.7975,
                        "c": 75.0875,
                        "v": 135647456,
                        "vw": 74.6099,
                        "t": 1577941200000,
                    }
                ],
            }
        )

        rows = self.client.get_historical_prices(
            "aapl",
            "2020-01-02",
            "2020-01-03",
        )

        self.session.get.assert_called_once_with(
            "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/2020-01-02/2020-01-03",
            params={"adjusted": "true", "sort": "asc", "limit": 50000},
            timeout=12.5,
        )
        self.assertEqual(
            rows,
            [
                {
                    "ticker": "AAPL",
                    "price_date": date(2020, 1, 2),
                    "open": 74.06,
                    "high": 75.15,
                    "low": 73.7975,
                    "close": 75.0875,
                    "volume": 135647456,
                    "vwap": 74.6099,
                }
            ],
        )

    def test_news_normalizes_per_ticker_relationships(self) -> None:
        raw_article = {
            "id": "article-1",
            "title": "Apple and Microsoft update",
            "description": "Two companies reported updates.",
            "author": "Reporter",
            "publisher": {"name": "Example News"},
            "article_url": "https://news.example.invalid/article-1",
            "published_utc": "2026-08-09T12:00:00Z",
            "keywords": ["technology", "earnings"],
            "tickers": ["AAPL", "MSFT", "NVDA"],
            "insights": [
                {
                    "ticker": "AAPL",
                    "sentiment": "positive",
                    "sentiment_reasoning": "Strong demand.",
                },
                {
                    "ticker": "MSFT",
                    "sentiment": "negative",
                    "sentiment_reasoning": "Demand weakened.",
                },
                {
                    "ticker": "GOOG",
                    "sentiment": "neutral",
                    "sentiment_reasoning": "Mentioned only in context.",
                },
            ],
        }
        self.session.get.return_value = _response(
            {"status": "OK", "results": [raw_article]}
        )

        articles = self.client.get_news(
            " aapl ",
            limit=10,
            published_after=date(2026, 8, 1),
        )

        _, call_kwargs = self.session.get.call_args
        self.assertEqual(
            self.session.get.call_args.args[0],
            "https://api.massive.com/v2/reference/news",
        )
        self.assertEqual(call_kwargs["params"]["ticker"], "AAPL")
        self.assertEqual(call_kwargs["params"]["published_utc.gte"], "2026-08-01")
        article = articles[0]
        self.assertEqual(
            article["ticker_insights"],
            [
                {
                    "ticker": "AAPL",
                    "sentiment": "positive",
                    "sentiment_reasoning": "Strong demand.",
                },
                {
                    "ticker": "MSFT",
                    "sentiment": "negative",
                    "sentiment_reasoning": "Demand weakened.",
                },
                {
                    "ticker": "NVDA",
                    "sentiment": None,
                    "sentiment_reasoning": None,
                },
                {
                    "ticker": "GOOG",
                    "sentiment": "neutral",
                    "sentiment_reasoning": "Mentioned only in context.",
                },
            ],
        )
        self.assertNotIn("sentiment", article)
        self.assertNotIn("sentiment_reasoning", article)
        self.assertNotIn("tickers", article)
        self.assertNotIn("insights", article)
        self.assertEqual(article["publisher"], "Example News")
        self.assertEqual(
            article["published_at"],
            datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(article["raw_payload"], raw_article)

    def test_next_url_pagination_uses_real_contract_and_preserves_auth(self) -> None:
        first = {
            "results": [
                {
                    "id": "one",
                    "title": "First",
                    "published_utc": "2026-08-09T12:00:00Z",
                    "tickers": ["AAPL"],
                }
            ],
            "next_url": "https://api.massive.com/v2/reference/news?cursor=next",
        }
        second = {
            "results": [
                {
                    "id": "two",
                    "title": "Second",
                    "published_utc": "2026-08-08T12:00:00Z",
                    "tickers": ["AAPL"],
                }
            ]
        }
        self.session.get.side_effect = [_response(first), _response(second)]

        articles = self.client.get_news("AAPL", limit=2)

        self.assertEqual([article["id"] for article in articles], ["one", "two"])
        self.assertEqual(self.session.get.call_count, 2)
        second_call = self.session.get.call_args_list[1]
        self.assertEqual(
            second_call.args[0],
            "https://api.massive.com/v2/reference/news?cursor=next",
        )
        self.assertIsNone(second_call.kwargs["params"])
        self.assertEqual(
            self.session.headers["Authorization"],
            "Bearer unit-test-key",
        )

    def test_timeout_becomes_safe_error_without_key_leak(self) -> None:
        self.session.get.side_effect = requests.Timeout(
            "timeout while using unit-test-key"
        )

        with self.assertRaises(MassiveClientError) as raised:
            self.client.get_company_overview("AAPL")

        self.assertNotIn("unit-test-key", str(raised.exception))
        self.assertNotIn("timeout while", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
