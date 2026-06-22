"""Abstract interface for sentiment-analysis providers.

Every concrete provider (FinBERT, Claude, and any added later) implements this
interface so the rest of the app can treat them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict


class SentimentProvider(ABC):
    """Common interface implemented by every sentiment-analysis provider.

    Attributes
    ----------
    name : str
        Short identifier used to select this provider (e.g. ``"finbert"``).
    requires_api_key : bool
        Whether instantiating this provider requires an API key.
    """

    name: ClassVar[str]
    requires_api_key: ClassVar[bool]

    @abstractmethod
    def score(self, text: str) -> Dict[str, Any]:
        """Score the sentiment of a piece of text.

        Parameters
        ----------
        text : str
            The text to analyze (e.g. a news headline + summary).

        Returns
        -------
        dict
            A dict with keys ``"label"`` (``"positive"``, ``"negative"``, or
            ``"neutral"``), ``"score"`` (confidence in ``[0, 1]``), and
            ``"signed_score"`` (``+score`` for positive, ``-score`` for
            negative, ``0.0`` for neutral).
        """
        raise NotImplementedError
