"""Plotly chart builders for sentiment results."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go

_LABEL_COLORS = {
    "positive": "#27AE60",
    "negative": "#E74C3C",
    "neutral": "#95A5A6",
}


def plot_sentiment_timeline(scored_articles: List[Dict[str, Any]]) -> go.Figure:
    """Build a scatter chart of signed sentiment score over time.

    Parameters
    ----------
    scored_articles : list of dict
        Each dict must contain ``datetime``, ``signed_score``, and ``label``
        (``"positive"``, ``"negative"``, or ``"neutral"``); ``headline`` is
        used for hover text if present.

    Returns
    -------
    plotly.graph_objects.Figure
        Points colored by label (positive=green, negative=red, neutral=gray),
        ordered by date.
    """
    df = pd.DataFrame(scored_articles)
    df["_date"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("_date")

    fig = go.Figure()
    for label, color in _LABEL_COLORS.items():
        subset = df[df["label"] == label]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["_date"],
                y=subset["signed_score"],
                mode="markers",
                name=label.capitalize(),
                marker=dict(color=color, size=10),
                text=subset.get("headline", ""),
                hovertemplate="%{text}<br>%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Signed sentiment score",
        template="plotly_white",
    )
    return fig


def plot_sentiment_distribution(aggregated_dict: Dict[str, Any]) -> go.Figure:
    """Build a bar chart of positive/negative/neutral article counts.

    Parameters
    ----------
    aggregated_dict : dict
        The output of :func:`aggregator.aggregate_sentiment`, containing
        ``positive_count``, ``negative_count``, and ``neutral_count``.

    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart of the three counts.
    """
    labels = ["Positive", "Negative", "Neutral"]
    counts = [
        aggregated_dict.get("positive_count", 0),
        aggregated_dict.get("negative_count", 0),
        aggregated_dict.get("neutral_count", 0),
    ]
    colors = [
        _LABEL_COLORS["positive"],
        _LABEL_COLORS["negative"],
        _LABEL_COLORS["neutral"],
    ]

    fig = go.Figure(data=[go.Bar(x=labels, y=counts, marker_color=colors)])
    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Number of articles",
        template="plotly_white",
    )
    return fig
