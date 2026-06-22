"""Company news fetching via the Finnhub free tier."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

_FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"


def fetch_company_news(ticker: str, days_back: int = 7) -> List[Dict[str, Any]]:
    """Fetch recent company news for a ticker from Finnhub.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. ``"AAPL"``.
    days_back : int, optional
        How many days of news history to request, counting back from today.
        Defaults to ``7``.

    Returns
    -------
    list of dict
        One dict per article with keys ``headline``, ``summary``,
        ``datetime`` (a human-readable date string), ``url``, and ``source``.
        An invalid ticker or a response with no articles returns an empty
        list rather than raising.

    Raises
    ------
    ValueError
        If ``FINNHUB_API_KEY`` is not set in the environment, or if the
        request to Finnhub fails (network error or non-2xx response).
    """
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError(
            "FINNHUB_API_KEY not found. Copy .env.example to .env and add "
            "your free Finnhub API key."
        )

    today = datetime.now().date()
    start = today - timedelta(days=days_back)

    params = {
        "symbol": ticker.strip().upper(),
        "from": start.isoformat(),
        "to": today.isoformat(),
        "token": api_key,
    }

    try:
        response = requests.get(_FINNHUB_NEWS_URL, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json()
    except Exception as exc:
        raise ValueError(
            "Could not fetch news. Check your API key and internet connection."
        ) from exc

    if not isinstance(articles, list) or not articles:
        return []

    news: List[Dict[str, Any]] = []
    for item in articles:
        published = datetime.fromtimestamp(item.get("datetime", 0))
        news.append(
            {
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "datetime": published.strftime("%Y-%m-%d %H:%M"),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
            }
        )
    return news
