"""Plain-language summary of a batch of scored news articles.

Deterministic, template-based text generation — no LLM calls. Describes the
*content* of the news (topics, sources, tone) to complement the numeric
sentiment metrics shown elsewhere in the app.
"""

from __future__ import annotations

from typing import Any, Dict, List

from topic_extractor import extract_notable_sources, extract_top_keywords

# Phrase variants per sentiment band, so the summary doesn't read identically
# every time. Keyed by a short band name.
_TONE_PHRASES: Dict[str, str] = {
    "mostly_positive": "The overall tone of the coverage was mostly positive",
    "mildly_positive": "The overall tone of the coverage leaned mildly positive",
    "mostly_negative": "The overall tone of the coverage was mostly negative",
    "mildly_negative": "The overall tone of the coverage leaned mildly negative",
    "mixed": "The overall tone of the coverage was mixed to neutral",
}


def _tone_band(simple_average: float) -> str:
    """Bucket a simple-average signed score into a tone band name."""
    if simple_average > 0.15:
        return "mostly_positive"
    if simple_average > 0:
        return "mildly_positive"
    if simple_average < -0.15:
        return "mostly_negative"
    if simple_average < 0:
        return "mildly_negative"
    return "mixed"


def _join_naturally(items: List[str]) -> str:
    """Join a list of strings into a natural English enumeration.

    ``["A"] -> "A"``, ``["A", "B"] -> "A and B"``,
    ``["A", "B", "C"] -> "A, B, and C"``.
    """
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _truncate_headline(headline: str, limit: int = 80) -> str:
    """Truncate a headline to ``limit`` characters, adding an ellipsis if cut."""
    headline = headline.strip()
    if len(headline) <= limit:
        return headline
    return headline[:limit].rstrip() + "..."


def generate_news_summary(
    articles: List[Dict[str, Any]],
    aggregated_sentiment: Dict[str, Any],
    days_back: int,
) -> str:
    """Build a short plain-language summary of the fetched news.

    Parameters
    ----------
    articles : list of dict
        The scored article list; each dict has at least ``headline``,
        ``source``, and ``label``.
    aggregated_sentiment : dict
        Output of :func:`aggregator.aggregate_sentiment`, with keys such as
        ``simple_average`` and ``total_articles``.
    days_back : int
        Number of days of news the user requested.

    Returns
    -------
    str
        A roughly 7-10 sentence paragraph describing the volume, tone, recurring
        topics, main sources, strongest individual signals, positive/negative
        balance, and neutral share of the coverage. If no articles are present,
        a short message saying so is returned instead.
    """
    total = aggregated_sentiment.get("total_articles", 0)
    if total == 0:
        return (
            "No articles were found for this period, so no summary could be "
            "generated."
        )

    day_word = "day" if days_back == 1 else "days"
    article_word = "article" if total == 1 else "articles"
    verb = "was" if total == 1 else "were"

    pos = aggregated_sentiment.get("positive_count", 0)
    neg = aggregated_sentiment.get("negative_count", 0)
    neu = aggregated_sentiment.get("neutral_count", 0)

    sentences: List[str] = []

    # 1) Volume + time window.
    sentences.append(
        f"Over the last {days_back} {day_word}, {total} {article_word} {verb} "
        f"found for this ticker, giving a reasonable read on recent coverage."
    )

    # 2) Overall tone in words, with the raw breakdown.
    band = _tone_band(float(aggregated_sentiment.get("simple_average", 0.0)))
    sentences.append(
        f"{_TONE_PHRASES[band]}, with {pos} positive, {neg} negative, and "
        f"{neu} neutral headlines."
    )

    # 3) Sentiment balance detail (ratio, guarding against divide-by-zero).
    if neg == 0:
        if pos > 0:
            sentences.append(
                "No negative coverage was found, so the positive and neutral "
                "stories shaped the overall picture."
            )
        else:
            sentences.append(
                "No negative coverage was found during this window."
            )
    elif pos == 0:
        sentences.append(
            "No positive coverage was found, so the negative and neutral "
            "stories dominated the picture."
        )
    else:
        ratio = round(pos / neg, 1)
        sentences.append(
            f"Positive articles outnumbered negative ones by roughly "
            f"{ratio} to 1."
        )

    # 4) Neutral share context.
    neutral_fraction = neu / total
    neutral_phrase = _fraction_word(neutral_fraction)
    sentences.append(
        f"{neutral_phrase[0].upper()}{neutral_phrase[1:]} of the coverage was "
        f"neutral in tone, often reflecting purely factual reporting."
    )

    # 5) Strongest individual signals (most positive / most negative headline).
    most_positive = max(articles, key=lambda a: a.get("signed_score", 0.0))
    most_negative = min(articles, key=lambda a: a.get("signed_score", 0.0))
    has_pos = most_positive.get("signed_score", 0.0) > 0
    has_neg = most_negative.get("signed_score", 0.0) < 0
    pos_hl = _truncate_headline(most_positive.get("headline", ""))
    neg_hl = _truncate_headline(most_negative.get("headline", ""))
    if has_pos and has_neg:
        sentences.append(
            f"The most positive coverage came from \"{pos_hl}\", while the most "
            f"negative was \"{neg_hl}\"."
        )
    elif has_pos:
        sentences.append(f"The most positive coverage came from \"{pos_hl}\".")
    elif has_neg:
        sentences.append(f"The most negative coverage came from \"{neg_hl}\".")

    # 6) Recurring topics.
    keywords = extract_top_keywords(articles, top_n=5)
    if keywords:
        sentences.append(
            "Recurring topics across the coverage include "
            f"{_join_naturally(keywords)}, which together hint at what was "
            "driving the news flow."
        )

    # 7) Most common sources.
    sources = extract_notable_sources(articles, top_n=3)
    if sources:
        sentences.append(
            f"Most of the reporting came from {_join_naturally(sources)}."
        )

    return " ".join(sentences)


def _fraction_word(fraction: float) -> str:
    """Describe a 0-1 fraction in rough plain-language terms."""
    if fraction == 0:
        return "none"
    if fraction < 0.15:
        return "a small fraction"
    if fraction < 0.30:
        return "around a quarter"
    if fraction < 0.42:
        return "about a third"
    if fraction < 0.58:
        return "about half"
    if fraction < 0.75:
        return "around two-thirds"
    return "the bulk"
