"""stock-news-signal-analyzer — sentiment-driven news analysis for a stock ticker.

Fetches recent company news via Finnhub, scores each headline/summary with a
pluggable sentiment provider (FinBERT by default, or any OpenAI-compatible LLM
API you bring), and visualizes the aggregated signal. Run with::

    streamlit run app.py
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from aggregator import aggregate_sentiment
from charts import plot_sentiment_distribution, plot_sentiment_timeline
from news_fetcher import fetch_company_news
from news_summary import generate_news_summary
from sentiment_providers import SentimentProvider, get_provider

load_dotenv()

PROVIDER_OPTIONS = {
    "finbert": "FinBERT — free, local",
    "custom_llm": "Custom LLM API — bring your own key",
}

# Threshold (in signed-score units) for bucketing the overall sentiment label.
_NEUTRAL_BAND = 0.05


@st.cache_resource(show_spinner=False)
def _get_cached_finbert_provider() -> SentimentProvider:
    """Load and cache the FinBERT provider once per Streamlit session."""
    return get_provider("finbert")


def _instantiate_provider(name: str, provider_kwargs: Dict[str, Any]) -> SentimentProvider:
    """Instantiate the selected provider, caching only the FinBERT instance.

    FinBERT is loaded once per session via ``st.cache_resource``; the custom LLM
    provider is a lightweight HTTP client and is recreated on every call.
    """
    if name == "finbert":
        return _get_cached_finbert_provider()
    return get_provider(name, **provider_kwargs)


def _overall_label(score: float) -> str:
    """Bucket an aggregate signed score into a plain-language label."""
    if score > _NEUTRAL_BAND:
        return "Positive"
    if score < -_NEUTRAL_BAND:
        return "Negative"
    return "Neutral"


def _score_articles(
    provider: SentimentProvider,
    articles: List[Dict[str, Any]],
    provider_name: str,
) -> List[Dict[str, Any]]:
    """Score every article and merge the result into the article dict.

    Shows a progress bar when using a remote LLM API, since each article is a
    separate, non-trivial network call.
    """
    scored: List[Dict[str, Any]] = []
    total = len(articles)
    progress = (
        st.progress(0.0, text="Scoring articles via LLM API...")
        if provider_name == "custom_llm"
        else None
    )

    for i, article in enumerate(articles):
        text = f"{article['headline']} {article['summary']}".strip()
        result = provider.score(text)
        scored.append({**article, **result})
        if progress is not None:
            progress.progress(
                (i + 1) / total,
                text=f"Scoring articles via LLM API... ({i + 1}/{total})",
            )

    if progress is not None:
        progress.empty()

    return scored


def run_analysis(
    ticker: str,
    days_back: int,
    provider_name: str,
    provider_kwargs: Dict[str, Any],
) -> None:
    """Fetch news, score it, aggregate, and render the full results view."""
    if provider_name == "finbert":
        with st.spinner(
            "Loading FinBERT model (first run may take a minute to download)..."
        ):
            provider = _instantiate_provider(provider_name, provider_kwargs)
    else:
        provider = _instantiate_provider(provider_name, provider_kwargs)

    with st.spinner(f"Fetching news for {ticker.upper()}..."):
        articles = fetch_company_news(ticker, days_back=days_back)

    if not articles:
        st.warning(
            f"No recent news found for {ticker.upper()} in the last "
            f"{days_back} day(s). Try a longer time window or a different ticker."
        )
        return

    scored_articles = _score_articles(provider, articles, provider_name)
    aggregated = aggregate_sentiment(scored_articles)

    st.subheader(f"Sentiment analysis for {ticker.upper()} — last {days_back} day(s)")

    label = _overall_label(aggregated["recency_weighted_average"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall sentiment", label)
    c2.metric("Positive articles", aggregated["positive_count"])
    c3.metric("Negative articles", aggregated["negative_count"])
    c4.metric("Neutral articles", aggregated["neutral_count"])
    st.caption(
        f"Simple average: {aggregated['simple_average']:.3f} · "
        f"Recency-weighted average: {aggregated['recency_weighted_average']:.3f} · "
        f"Total articles: {aggregated['total_articles']}"
    )

    st.plotly_chart(plot_sentiment_timeline(scored_articles), use_container_width=True)
    st.plotly_chart(plot_sentiment_distribution(aggregated), use_container_width=True)

    st.subheader("Articles")
    table_rows = [
        {
            "Headline": a["headline"],
            "Date": a["datetime"],
            "Source": a["source"],
            "Label": a["label"],
            "Score": round(a["signed_score"], 3),
            "URL": a["url"],
        }
        for a in scored_articles
    ]
    st.dataframe(
        table_rows,
        column_config={"URL": st.column_config.LinkColumn("URL")},
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("News Summary")
    st.info(generate_news_summary(scored_articles, aggregated, days_back))


def main() -> None:
    """Render the Streamlit page and wire up the sidebar controls."""
    st.set_page_config(page_title="stock-news-signal-analyzer", page_icon="📰")

    st.title("📰 Stock News Signal Analyzer")
    st.write(
        "Enter a stock ticker to fetch its recent news and see an aggregated "
        "sentiment signal — powered by a pluggable sentiment engine (FinBERT "
        "for free local analysis, or any OpenAI-compatible LLM API you supply)."
    )

    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker symbol", value="AAPL")
        days_back = st.slider("Days of news to fetch", min_value=1, max_value=30, value=7)

        provider_name = st.selectbox(
            "Sentiment provider",
            options=list(PROVIDER_OPTIONS.keys()),
            format_func=lambda key: PROVIDER_OPTIONS[key],
            index=0,
        )

        provider_kwargs: Dict[str, Any] = {}
        custom_fields_missing = False
        if provider_name == "custom_llm":
            endpoint_url = st.text_input(
                "API Endpoint URL",
                placeholder="https://api.openai.com/v1/chat/completions",
            )
            model_name = st.text_input(
                "Model name",
                placeholder="e.g. gpt-4o-mini or claude-haiku-4-5-20251001",
            )
            api_key = st.text_input("API Key", type="password")
            st.caption(
                "Works with any OpenAI-compatible chat completions API — "
                "including OpenAI, Anthropic (via their OpenAI-compatible "
                "endpoint if available), Gemini (via compatible proxy), or "
                "self-hosted models. You provide your own endpoint, model name, "
                "and key."
            )
            provider_kwargs = {
                "endpoint_url": endpoint_url.strip(),
                "model_name": model_name.strip(),
                "api_key": api_key.strip(),
            }
            custom_fields_missing = not all(provider_kwargs.values())

        run = st.button("Analyze", type="primary")

    if not run:
        st.caption("👈 Set your options in the sidebar and click **Analyze**.")
        return

    if not ticker.strip():
        st.warning("Please enter a ticker symbol to get started (e.g. AAPL).")
        return

    if provider_name == "custom_llm" and custom_fields_missing:
        st.warning(
            "Fill in the endpoint, model name, and API key to use a custom LLM "
            "provider."
        )
        return

    try:
        run_analysis(ticker, days_back, provider_name, provider_kwargs)
    except ValueError as exc:
        st.error(str(exc))
    except Exception:  # noqa: BLE001 - last-resort guard so the app never crashes
        st.error(
            "Something went wrong while analyzing this ticker. Please try again "
            "or pick a different ticker/provider."
        )


if __name__ == "__main__":
    main()
