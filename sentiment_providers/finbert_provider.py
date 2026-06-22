"""FinBERT sentiment provider — free, local, no API key required."""

from __future__ import annotations

from typing import Any, Dict

from transformers import pipeline

from .base import SentimentProvider

# FinBERT (BERT-base) has a 512-token limit; the pipeline truncates for us.
_MAX_TOKENS = 512


class FinBERTProvider(SentimentProvider):
    """Sentiment provider backed by ``ProsusAI/finbert``, run locally."""

    name = "finbert"
    requires_api_key = False

    def __init__(self) -> None:
        """Load the FinBERT pipeline once.

        Loading happens here so callers control caching (e.g. via
        ``st.cache_resource`` around the provider instance) rather than
        inside this class.
        """
        self._pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=_MAX_TOKENS,
        )

    def score(self, text: str) -> Dict[str, Any]:
        """Score the sentiment of ``text`` using FinBERT.

        Parameters
        ----------
        text : str
            The text to analyze. Truncated to FinBERT's max input length
            before scoring.

        Returns
        -------
        dict
            ``{"label": ..., "score": ..., "signed_score": ...}`` — see
            :class:`sentiment_providers.base.SentimentProvider`.
        """
        result = self._pipeline(text)[0]

        label = result["label"].lower()
        score = float(result["score"])

        if label == "positive":
            signed_score = score
        elif label == "negative":
            signed_score = -score
        else:
            label = "neutral"
            signed_score = 0.0

        return {"label": label, "score": score, "signed_score": signed_score}
