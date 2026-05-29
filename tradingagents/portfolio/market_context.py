"""Best-effort portfolio-wide market context collection."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio.data_inputs import _resolve_benchmark_symbol
from tradingagents.portfolio.schemas import PortfolioRequest


DEFAULT_TEXT_LIMIT = 1800
DEFAULT_INDICATORS = ("close_50_sma", "close_200_sma", "rsi", "macd")


@dataclass(frozen=True)
class PortfolioMarketContext:
    """Broad market inputs for portfolio-level allocation decisions."""

    benchmark_symbol: str | None = None
    lookback_days: int = 7
    global_news: str = ""
    benchmark_news: str = ""
    benchmark_fundamentals: str = ""
    benchmark_indicators: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_portfolio_market_context(
    portfolio_request: PortfolioRequest,
    *,
    benchmark_symbol: str | None = None,
    config: dict[str, Any] | None = None,
    text_limit: int = DEFAULT_TEXT_LIMIT,
    indicators: tuple[str, ...] = DEFAULT_INDICATORS,
) -> PortfolioMarketContext:
    """Collect broad market/news/fundamental context for portfolio agents.

    This is independent from per-holding single-stock conclusions. All provider
    calls are optional; failures become warnings so portfolio analysis can
    continue with deterministic analytics and holding-level evidence.
    """

    effective_config = config or DEFAULT_CONFIG
    trade_date = _coerce_date(portfolio_request.trade_date)
    trade_date_str = trade_date.isoformat()
    lookback_days = int(effective_config.get("global_news_lookback_days", 7) or 7)
    limit = int(effective_config.get("global_news_article_limit", 10) or 10)
    benchmark = benchmark_symbol or _resolve_benchmark_symbol(
        portfolio_request,
        effective_config,
    )
    start_date = (trade_date - timedelta(days=lookback_days)).isoformat()
    warnings: list[str] = []

    global_news = _fetch_text(
        "get_global_news",
        trade_date_str,
        lookback_days,
        limit,
        warning_label="portfolio global market news",
        warnings=warnings,
        text_limit=text_limit,
        compact_news=True,
    )

    benchmark_news = ""
    benchmark_fundamentals = ""
    benchmark_indicators: dict[str, str] = {}
    if benchmark:
        benchmark_news = _fetch_text(
            "get_news",
            benchmark,
            start_date,
            trade_date_str,
            warning_label=f"{benchmark} market news",
            warnings=warnings,
            text_limit=text_limit,
            compact_news=True,
        )
        benchmark_fundamentals = _fetch_text(
            "get_fundamentals",
            benchmark,
            trade_date_str,
            warning_label=f"{benchmark} fundamentals",
            warnings=warnings,
            text_limit=text_limit,
        )
        for indicator in indicators:
            indicator_text = _fetch_text(
                "get_indicators",
                benchmark,
                indicator,
                trade_date_str,
                90,
                warning_label=f"{benchmark} {indicator}",
                warnings=warnings,
                text_limit=900,
            )
            if indicator_text:
                benchmark_indicators[indicator] = indicator_text

    return PortfolioMarketContext(
        benchmark_symbol=benchmark,
        lookback_days=lookback_days,
        global_news=global_news,
        benchmark_news=benchmark_news,
        benchmark_fundamentals=benchmark_fundamentals,
        benchmark_indicators=benchmark_indicators,
        warnings=warnings,
    )


def _fetch_text(
    method: str,
    *args: Any,
    warning_label: str,
    warnings: list[str],
    text_limit: int,
    compact_news: bool = False,
) -> str:
    try:
        response = route_to_vendor(method, *args)
    except Exception as exc:
        warnings.append(f"{warning_label} unavailable: {exc}")
        return ""

    if compact_news:
        text = _compact_news_response(response)
    else:
        text = _stringify_response(response)
    if not text or _looks_unavailable(text):
        warnings.append(f"{warning_label} unavailable")
        return ""
    return _truncate_text(text, text_limit)


def _compact_news_response(response: Any) -> str:
    if isinstance(response, str):
        stripped = response.strip()
        if stripped.startswith("{"):
            try:
                response = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
        else:
            return stripped

    if not isinstance(response, dict):
        return _stringify_response(response)
    if any(key in response for key in ("Information", "Error Message", "Note")):
        return _stringify_response(response)

    feed = response.get("feed")
    if not isinstance(feed, list):
        return _stringify_response(response)

    labels: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    scores: list[float] = []
    headlines: list[str] = []
    for article in feed:
        if not isinstance(article, dict):
            continue
        label = article.get("overall_sentiment_label")
        if label:
            labels[str(label)] += 1
        try:
            scores.append(float(article.get("overall_sentiment_score")))
        except (TypeError, ValueError):
            pass
        for topic in article.get("topics") or []:
            if isinstance(topic, dict) and topic.get("topic"):
                topics[str(topic["topic"])] += 1
        title = str(article.get("title") or "").strip()
        source = str(article.get("source") or "").strip()
        if title:
            headlines.append(f"- {title}" + (f" ({source})" if source else ""))

    average_score = sum(scores) / len(scores) if scores else None
    summary = {
        "article_count": len(feed),
        "average_sentiment_score": round(average_score, 4)
        if average_score is not None
        else None,
        "sentiment_label_counts": dict(labels.most_common()),
        "top_topics": dict(topics.most_common(8)),
        "top_headlines": headlines[:8],
    }
    return json.dumps(summary, indent=2, sort_keys=True)


def _stringify_response(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    return json.dumps(response, indent=2, sort_keys=True, default=str)


def _looks_unavailable(text: str) -> bool:
    lowered = text.strip().lower()
    return (
        not lowered
        or lowered.startswith("error ")
        or lowered.startswith("error fetching")
        or lowered.startswith("no news found")
        or lowered.startswith("no global news found")
        or "premium endpoint" in lowered
    )


def _truncate_text(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 18, 0)].rstrip() + "\n... [truncated]"


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
