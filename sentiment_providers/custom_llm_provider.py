"""Vendor-neutral LLM sentiment provider.

Works with any OpenAI-compatible chat-completions endpoint — the user supplies
the endpoint URL, model name, API key, and (optionally) the auth header scheme.
No vendor name is baked in.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import requests

from .base import SentimentProvider

_MAX_TOKENS = 50
_TIMEOUT_SECONDS = 30

_PROMPT_TEMPLATE = (
    "Classify the sentiment of this financial news text as it would likely "
    "affect the stock price. Respond with ONLY a JSON object, no other text:\n"
    '{{"label": "positive" | "negative" | "neutral", "score": <float 0 to 1>}}\n\n'
    "Text: {text}"
)


class CustomLLMProvider(SentimentProvider):
    """Sentiment provider backed by any OpenAI-compatible chat API."""

    name = "custom_llm"
    requires_api_key = True

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        model_name: str,
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
    ) -> None:
        """Store the endpoint configuration. Does not call the API yet.

        Parameters
        ----------
        endpoint_url : str
            Full chat-completions endpoint URL, e.g.
            ``https://api.openai.com/v1/chat/completions``.
        api_key : str
            API key for the endpoint.
        model_name : str
            Model identifier to send in the request body.
        auth_header : str, optional
            Name of the HTTP header carrying the credential. Defaults to
            ``"Authorization"``.
        auth_scheme : str, optional
            Scheme prefix placed before the key in the auth header (e.g.
            ``"Bearer"``). Pass an empty string for schemes that send the raw
            key. Defaults to ``"Bearer"``.
        """
        self._endpoint_url = endpoint_url
        self._api_key = api_key
        self._model_name = model_name
        self._auth_header = auth_header
        self._auth_scheme = auth_scheme

    def _auth_value(self) -> str:
        """Build the auth header value, honoring an empty scheme."""
        if self._auth_scheme:
            return f"{self._auth_scheme} {self._api_key}"
        return self._api_key

    def score(self, text: str) -> Dict[str, Any]:
        """Score the sentiment of ``text`` via the configured LLM endpoint.

        Parameters
        ----------
        text : str
            The text to analyze.

        Returns
        -------
        dict
            ``{"label": ..., "score": ..., "signed_score": ...}`` — see
            :class:`sentiment_providers.base.SentimentProvider`.

        Raises
        ------
        ValueError
            If the HTTP request fails (network error or non-200 status), or the
            response cannot be parsed into the expected JSON object. The message
            includes the underlying error detail so the user can debug their
            endpoint configuration.
        """
        prompt = _PROMPT_TEMPLATE.format(text=text)
        headers = {
            self._auth_header: self._auth_value(),
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": _MAX_TOKENS,
        }

        try:
            response = requests.post(
                self._endpoint_url,
                headers=headers,
                json=payload,
                timeout=_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            raw = data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise ValueError(
                f"Custom LLM request failed: {exc}. Check your endpoint URL, "
                "API key, and model name."
            ) from exc

        # Strip markdown code fences if the model wrapped the JSON in them.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        try:
            parsed = json.loads(raw)
            label = str(parsed["label"]).lower()
            score = float(parsed["score"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not parse the LLM's sentiment response as JSON: {raw!r}. "
                "Check your endpoint URL, API key, and model name."
            ) from exc

        if label == "positive":
            signed_score = score
        elif label == "negative":
            signed_score = -score
        else:
            label = "neutral"
            signed_score = 0.0

        return {"label": label, "score": score, "signed_score": signed_score}
