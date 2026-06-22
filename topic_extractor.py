"""Lightweight keyword and source extraction for news articles.

Deterministic, dependency-free helpers used to describe what a batch of news
articles was about — no LLM calls involved.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

# A reasonably sized list of common English stopwords (not exhaustive).
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "in", "on", "for", "with", "at", "by", "from",
    "up", "out", "as", "it", "its", "this", "that", "these", "those", "they",
    "them", "their", "his", "her", "he", "she", "you", "your", "we", "our",
    "but", "not", "no", "so", "if", "then", "than", "too", "very", "can",
    "will", "would", "could", "should", "may", "might", "must", "do", "does",
    "did", "has", "have", "had", "says", "said", "say", "after", "before",
    "over", "under", "about", "more", "most", "new", "also", "into", "amid",
    "what", "which", "who", "when", "where", "why", "how", "all", "any",
    "some", "such", "just", "now", "still", "back", "down", "off", "per",
    "vs", "via", "amp", "inc", "corp", "ltd", "company", "stock", "shares",
    "market", "report", "reports", "update", "news",
}

# Tokens are sequences of letters/digits/apostrophes.
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def extract_top_keywords(articles: List[Dict[str, Any]], top_n: int = 5) -> List[str]:
    """Extract the most frequent meaningful keywords across articles.

    Parameters
    ----------
    articles : list of dict
        Article dicts, each with ``headline`` and ``summary`` text.
    top_n : int, optional
        Number of keywords to return. Defaults to ``5``.

    Returns
    -------
    list of str
        Up to ``top_n`` keywords ordered by frequency (most frequent first).
        Each keyword is returned in its most common casing variant as seen in
        the source text. Stopwords, tokens shorter than 3 characters, and pure
        numbers are excluded. An empty article list returns an empty list.
    """
    if not articles:
        return []

    lower_counts: Counter[str] = Counter()
    # For each lowercased word, track how often each casing variant appeared.
    casing_counts: Dict[str, Counter[str]] = {}

    for article in articles:
        text = f"{article.get('headline', '')} {article.get('summary', '')}"
        for token in _TOKEN_RE.findall(text):
            lowered = token.lower()
            if len(lowered) < 3 or lowered.isdigit() or lowered in _STOPWORDS:
                continue
            lower_counts[lowered] += 1
            casing_counts.setdefault(lowered, Counter())[token] += 1

    top_lowered = [word for word, _ in lower_counts.most_common(top_n)]
    # Map each chosen word back to its most common original casing.
    return [casing_counts[word].most_common(1)[0][0] for word in top_lowered]


def extract_notable_sources(articles: List[Dict[str, Any]], top_n: int = 3) -> List[str]:
    """Find the news sources that published the most articles.

    Parameters
    ----------
    articles : list of dict
        Article dicts, each with a ``source`` field.
    top_n : int, optional
        Number of sources to return. Defaults to ``3``.

    Returns
    -------
    list of str
        Up to ``top_n`` source names ordered by article count (most frequent
        first). An empty article list returns an empty list.
    """
    if not articles:
        return []

    source_counts: Counter[str] = Counter()
    for article in articles:
        source = (article.get("source") or "").strip()
        if source:
            source_counts[source] += 1

    return [source for source, _ in source_counts.most_common(top_n)]
