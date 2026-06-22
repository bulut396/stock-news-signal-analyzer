"""Sentiment provider package.

Exposes :func:`get_provider`, the only supported way to obtain a concrete
provider instance. Adding a new provider means writing one new file here and
adding one branch to this factory — no other code should ever import a
concrete provider class directly.
"""

from __future__ import annotations

from typing import Any

from .base import SentimentProvider
from .custom_llm_provider import CustomLLMProvider
from .finbert_provider import FinBERTProvider

_CUSTOM_LLM_REQUIRED_KWARGS = ("endpoint_url", "api_key", "model_name")


def get_provider(name: str, **kwargs: Any) -> SentimentProvider:
    """Instantiate a sentiment provider by name.

    Parameters
    ----------
    name : str
        Provider identifier: ``"finbert"`` or ``"custom_llm"``.
    **kwargs
        Provider-specific configuration. For ``"custom_llm"``, the required
        keys are ``endpoint_url``, ``api_key``, and ``model_name``; the optional
        keys ``auth_header`` and ``auth_scheme`` are passed through if given.

    Returns
    -------
    SentimentProvider
        A ready-to-use provider instance.

    Raises
    ------
    ValueError
        If ``name`` is ``"custom_llm"`` and any required kwarg is missing, or if
        ``name`` is not a recognized provider.
    """
    if name == "finbert":
        return FinBERTProvider()
    if name == "custom_llm":
        missing = [
            key for key in _CUSTOM_LLM_REQUIRED_KWARGS if not kwargs.get(key)
        ]
        if missing:
            raise ValueError(
                "The custom LLM provider requires: "
                f"{', '.join(missing)}."
            )
        return CustomLLMProvider(
            endpoint_url=kwargs["endpoint_url"],
            api_key=kwargs["api_key"],
            model_name=kwargs["model_name"],
            **{
                k: kwargs[k]
                for k in ("auth_header", "auth_scheme")
                if k in kwargs
            },
        )
    raise ValueError(f"Unknown provider: {name!r}")
