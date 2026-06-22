"""Aggregation of per-article sentiment scores into a portfolio-level signal."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def aggregate_sentiment(scored_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-article sentiment scores into a single summary.

    Parameters
    ----------
    scored_articles : list of dict
        Each dict must contain at least ``signed_score`` (float) and
        ``datetime`` (a date/datetime string parseable by
        :func:`pandas.to_datetime`).

    Returns
    -------
    dict
        ``simple_average``
            The unweighted mean of ``signed_score``.
        ``recency_weighted_average``
            A linearly recency-weighted mean: each article is weighted by the
            rank of its calendar day among the unique days present (oldest
            day = weight 1, most recent day = the highest weight), so newer
            news contributes more to the result.
        ``positive_count``, ``negative_count``, ``neutral_count``
            Counts inferred from the sign of ``signed_score``
            (``> 0`` positive, ``< 0`` negative, ``== 0`` neutral).
        ``total_articles``
            Total number of articles.

        All numeric values are ``0.0``/``0`` for empty input (no division by
        zero).
    """
    if not scored_articles:
        return {
            "simple_average": 0.0,
            "recency_weighted_average": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "total_articles": 0,
        }

    df = pd.DataFrame(scored_articles)
    df["signed_score"] = df["signed_score"].astype(float)
    df["_day"] = pd.to_datetime(df["datetime"]).dt.date

    unique_days = sorted(df["_day"].unique())
    day_weight = {day: rank + 1 for rank, day in enumerate(unique_days)}
    df["_weight"] = df["_day"].map(day_weight)

    simple_average = float(df["signed_score"].mean())
    recency_weighted_average = float(
        (df["signed_score"] * df["_weight"]).sum() / df["_weight"].sum()
    )

    positive_count = int((df["signed_score"] > 0).sum())
    negative_count = int((df["signed_score"] < 0).sum())
    neutral_count = int((df["signed_score"] == 0).sum())

    return {
        "simple_average": simple_average,
        "recency_weighted_average": recency_weighted_average,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "total_articles": int(len(df)),
    }
