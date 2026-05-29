import json

import pytest

from tradingagents.portfolio import (
    AssetType,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
    collect_portfolio_market_context,
)


def stock_position(symbol: str, weight: float) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol=symbol, asset_type=AssetType.STOCK),
        current_weight=weight,
        quantity=10,
    )


@pytest.mark.unit
def test_collects_portfolio_wide_market_context(monkeypatch):
    request = PortfolioRequest(
        positions=[stock_position("AAPL", 0.60), stock_position("MSFT", 0.40)],
        trade_date="2026-05-27",
    )
    calls = []

    def fake_route(method, *args):
        calls.append((method, args))
        if method == "get_global_news":
            return json.dumps(
                {
                    "feed": [
                        {
                            "title": "Fed policy weighs on growth stocks",
                            "source": "Example Wire",
                            "overall_sentiment_score": "-0.15",
                            "overall_sentiment_label": "Somewhat-Bearish",
                            "topics": [{"topic": "Economy - Monetary"}],
                        },
                        {
                            "title": "AI capex supports mega-cap earnings",
                            "source": "Example News",
                            "overall_sentiment_score": "0.20",
                            "overall_sentiment_label": "Somewhat-Bullish",
                            "topics": [{"topic": "Financial Markets"}],
                        },
                    ]
                }
            )
        if method == "get_news":
            return "## SPY News\nBroad market ETF flows cooled."
        if method == "get_fundamentals":
            return "Sector: Broad Market\nIndustry: ETF"
        if method == "get_indicators":
            return f"{args[1]} value for {args[0]}"
        raise AssertionError(f"unexpected route: {method} {args}")

    monkeypatch.setattr(
        "tradingagents.portfolio.market_context.route_to_vendor",
        fake_route,
    )

    context = collect_portfolio_market_context(
        request,
        benchmark_symbol="SPY",
        config={"global_news_lookback_days": 5, "global_news_article_limit": 3},
    )

    assert context.benchmark_symbol == "SPY"
    assert context.lookback_days == 5
    assert context.warnings == []
    assert "average_sentiment_score" in context.global_news
    assert "Fed policy weighs on growth stocks" in context.global_news
    assert "Broad market ETF flows cooled" in context.benchmark_news
    assert context.benchmark_fundamentals == "Sector: Broad Market\nIndustry: ETF"
    assert context.benchmark_indicators["rsi"] == "rsi value for SPY"
    assert ("get_global_news", ("2026-05-27", 5, 3)) in calls


@pytest.mark.unit
def test_market_context_warns_and_continues_when_provider_data_fails(monkeypatch):
    request = PortfolioRequest(
        positions=[stock_position("AAPL", 1.0)],
        trade_date="2026-05-27",
    )

    def fake_route(method, *args):
        if method == "get_global_news":
            raise RuntimeError("news provider offline")
        if method == "get_news":
            return "No news found for SPY"
        if method == "get_fundamentals":
            return '{"Information": "premium endpoint"}'
        if method == "get_indicators":
            return ""
        raise AssertionError(f"unexpected route: {method} {args}")

    monkeypatch.setattr(
        "tradingagents.portfolio.market_context.route_to_vendor",
        fake_route,
    )

    context = collect_portfolio_market_context(
        request,
        benchmark_symbol="SPY",
        config={"global_news_lookback_days": 7, "global_news_article_limit": 10},
    )

    assert context.global_news == ""
    assert context.benchmark_news == ""
    assert context.benchmark_indicators == {}
    assert any("portfolio global market news unavailable" in warning for warning in context.warnings)
    assert any("SPY market news unavailable" in warning for warning in context.warnings)
