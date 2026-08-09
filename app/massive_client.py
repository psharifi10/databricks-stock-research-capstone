"""Reusable client for the Massive Stocks REST API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import quote, urljoin, urlparse
from zoneinfo import ZoneInfo

import requests

from app.config import MassiveConfig, load_massive_config
from app.validation import ValidationError, normalize_ticker


MAX_NEWS_LIMIT = 100
_MARKET_TIMEZONE = ZoneInfo("America/New_York")


class MassiveClientError(RuntimeError):
    """A safe error raised when Massive cannot provide a valid response."""


class MassiveClient:
    """Thin Massive client with shared authentication and normalization."""

    def __init__(
        self,
        config: MassiveConfig | None = None,
        *,
        session: requests.Session | None = None,
        api_key_provider: Callable[[], str] | None = None,
    ) -> None:
        self._config = config or load_massive_config()
        provider = api_key_provider or self._config.require_api_key
        provided_key = provider()
        if not isinstance(provided_key, str) or not provided_key.strip():
            raise MassiveClientError("Massive API credentials are unavailable.")
        api_key = provided_key.strip()

        self._base_url = self._config.base_url.rstrip("/")
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )

    def get_company_overview(self, ticker: str) -> dict[str, Any]:
        """Fetch and normalize the current overview for one stock ticker."""

        symbol = normalize_ticker(ticker)
        payload = self._request_json(
            f"/v3/reference/tickers/{quote(symbol, safe='.-')}"
        )
        result = payload.get("results")
        if not isinstance(result, Mapping):
            raise MassiveClientError("Massive returned an invalid company response.")
        return _normalize_company(result, symbol)

    def get_historical_prices(
        self,
        ticker: str,
        start_date: date | str,
        end_date: date | str,
    ) -> list[dict[str, Any]]:
        """Fetch available daily bars without filling non-trading dates."""

        symbol = normalize_ticker(ticker)
        start = _normalize_date(start_date, "start_date")
        end = _normalize_date(end_date, "end_date")
        if start > end:
            raise ValidationError("start_date must be on or before end_date.")

        path = (
            f"/v2/aggs/ticker/{quote(symbol, safe='.-')}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        rows = self._collect_results(
            path,
            params={"adjusted": "true", "sort": "asc", "limit": 50000},
        )
        return [_normalize_price_bar(row, symbol) for row in rows]

    def get_news(
        self,
        ticker: str,
        *,
        limit: int = 25,
        published_after: date | datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch and normalize recent news for one ticker."""

        symbol = normalize_ticker(ticker)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValidationError("News limit must be an integer.")
        if limit < 1 or limit > MAX_NEWS_LIMIT:
            raise ValidationError(
                f"News limit must be between 1 and {MAX_NEWS_LIMIT}."
            )

        params: dict[str, Any] = {
            "ticker": symbol,
            "limit": limit,
            "sort": "published_utc",
            "order": "desc",
        }
        if published_after is not None:
            params["published_utc.gte"] = _normalize_published_after(
                published_after
            )

        articles = self._collect_results(
            "/v2/reference/news",
            params=params,
            max_results=limit,
        )
        return [_normalize_news_article(article) for article in articles]

    def close(self) -> None:
        """Close the reusable HTTP session."""

        self._session.close()

    def _collect_results(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        max_results: int | None = None,
    ) -> list[Mapping[str, Any]]:
        results: list[Mapping[str, Any]] = []
        target = path
        request_params = params
        followed_urls: set[str] = set()

        while True:
            payload = self._request_json(target, params=request_params)
            page_results = payload.get("results", [])
            if not isinstance(page_results, list) or not all(
                isinstance(item, Mapping) for item in page_results
            ):
                raise MassiveClientError("Massive returned an invalid results list.")

            remaining = (
                None if max_results is None else max_results - len(results)
            )
            if remaining is not None:
                results.extend(page_results[:remaining])
            else:
                results.extend(page_results)

            if max_results is not None and len(results) >= max_results:
                break

            next_url = payload.get("next_url")
            if not next_url:
                break
            if not isinstance(next_url, str) or next_url in followed_urls:
                raise MassiveClientError("Massive returned an invalid pagination link.")
            followed_urls.add(next_url)
            target = next_url
            request_params = None

        return results

    def _request_json(
        self,
        target: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._resolve_url(target)
        try:
            response = self._session.get(
                url,
                params=params,
                timeout=self._config.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise MassiveClientError(
                "The Massive market-data service request failed. Please try again."
            ) from error
        except ValueError as error:
            raise MassiveClientError(
                "The Massive market-data service returned invalid JSON."
            ) from error

        if not isinstance(payload, dict):
            raise MassiveClientError(
                "The Massive market-data service returned an invalid response."
            )
        return payload

    def _resolve_url(self, target: str) -> str:
        if target.startswith("/"):
            return urljoin(f"{self._base_url}/", target.lstrip("/"))

        candidate = urlparse(target)
        expected = urlparse(self._base_url)
        if (
            candidate.scheme != expected.scheme
            or candidate.hostname != expected.hostname
        ):
            raise MassiveClientError("Massive returned an unsafe pagination link.")
        return target


def _normalize_company(
    result: Mapping[str, Any],
    requested_ticker: str,
) -> dict[str, Any]:
    ticker = normalize_ticker(str(result.get("ticker") or requested_ticker))
    name = _required_text(result, "name", "company")
    sic_description = _optional_text(result.get("sic_description"))
    return {
        "ticker": ticker,
        "name": name,
        "legal_name": name,
        "description": _optional_text(result.get("description")),
        "market_cap": result.get("market_cap"),
        "market": _optional_text(result.get("market")),
        "exchange": _optional_text(result.get("primary_exchange")),
        "security_type": _optional_text(result.get("type")),
        "active": result.get("active") if isinstance(result.get("active"), bool) else None,
        "list_date": _optional_date(result.get("list_date")),
        "sic_code": _optional_text(result.get("sic_code")),
        "sic_description": sic_description,
        "industry": sic_description,
        "homepage_url": _optional_text(result.get("homepage_url")),
        "currency_name": _optional_text(result.get("currency_name")),
        "locale": _optional_text(result.get("locale")),
        "raw_source_payload": dict(result),
    }


def _normalize_price_bar(
    result: Mapping[str, Any],
    ticker: str,
) -> dict[str, Any]:
    timestamp = result.get("t")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise MassiveClientError("Massive returned a price bar without a timestamp.")
    try:
        market_date = datetime.fromtimestamp(
            timestamp / 1000,
            tz=timezone.utc,
        ).astimezone(_MARKET_TIMEZONE).date()
    except (OverflowError, OSError, ValueError) as error:
        raise MassiveClientError(
            "Massive returned a price bar with an invalid timestamp."
        ) from error

    return {
        "ticker": ticker,
        "price_date": market_date,
        "open": result.get("o"),
        "high": result.get("h"),
        "low": result.get("l"),
        "close": result.get("c"),
        "volume": result.get("v"),
        "vwap": result.get("vw"),
    }


def _normalize_news_article(result: Mapping[str, Any]) -> dict[str, Any]:
    article_id = _required_text(result, "id", "news article")
    title = _required_text(result, "title", "news article")

    relationships: dict[str, dict[str, Any]] = {}
    for raw_ticker in result.get("tickers") or []:
        try:
            ticker = normalize_ticker(str(raw_ticker))
        except ValidationError:
            continue
        relationships.setdefault(
            ticker,
            {
                "ticker": ticker,
                "sentiment": None,
                "sentiment_reasoning": None,
            },
        )

    for raw_insight in result.get("insights") or []:
        if not isinstance(raw_insight, Mapping):
            continue
        try:
            insight_ticker = normalize_ticker(str(raw_insight.get("ticker") or ""))
        except ValidationError:
            continue
        relationships[insight_ticker] = {
            "ticker": insight_ticker,
            "sentiment": _optional_text(raw_insight.get("sentiment")),
            "sentiment_reasoning": _optional_text(
                raw_insight.get("sentiment_reasoning")
            ),
        }
    publisher = result.get("publisher")
    publisher_name = (
        _optional_text(publisher.get("name"))
        if isinstance(publisher, Mapping)
        else _optional_text(publisher)
    )
    keywords = [
        value.strip()
        for value in (result.get("keywords") or [])
        if isinstance(value, str) and value.strip()
    ]

    return {
        "id": article_id,
        "title": title,
        "description": _optional_text(result.get("description")),
        "author": _optional_text(result.get("author")),
        "publisher": publisher_name,
        "article_url": _optional_text(result.get("article_url")),
        "published_at": _optional_datetime(result.get("published_utc")),
        "keywords": keywords,
        "raw_payload": dict(result),
        "ticker_insights": list(relationships.values()),
    }


def _normalize_date(value: date | str, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as error:
            raise ValidationError(f"{name} must use YYYY-MM-DD format.") from error
    raise ValidationError(f"{name} must be a date or YYYY-MM-DD string.")


def _normalize_published_after(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValidationError("published_after datetime must include a timezone.")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        normalized = value.strip()
        try:
            if "T" in normalized:
                datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            else:
                date.fromisoformat(normalized)
        except ValueError as error:
            raise ValidationError(
                "published_after must be an ISO date or timezone-aware datetime."
            ) from error
        return normalized
    raise ValidationError(
        "published_after must be an ISO date or timezone-aware datetime."
    )


def _optional_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise MassiveClientError("Massive returned an invalid company date.") from error


def _optional_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise MassiveClientError(
            "Massive returned an invalid publication timestamp."
        ) from error
    if parsed.tzinfo is None:
        raise MassiveClientError(
            "Massive returned a publication timestamp without a timezone."
        )
    return parsed


def _required_text(
    mapping: Mapping[str, Any],
    key: str,
    object_name: str,
) -> str:
    value = _optional_text(mapping.get(key))
    if value is None:
        raise MassiveClientError(
            f"Massive returned a {object_name} without required {key}."
        )
    return value


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
