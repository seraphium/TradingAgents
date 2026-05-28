from inspect import signature

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.portfolio import (
    AssetType,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
)


def stock_position(symbol: str, weight: float) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol=symbol, asset_type=AssetType.STOCK),
        current_weight=weight,
        quantity=10,
    )


def cash_position(weight: float) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol="CASH", asset_type=AssetType.CASH),
        current_weight=weight,
        market_value=1000,
    )


def option_position(symbol: str, underlying: str, weight: float) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(
            symbol=symbol,
            asset_type=AssetType.OPTION,
            underlying=underlying,
            expiry="2025-06-20",
            strike=200,
            right="C",
        ),
        current_weight=weight,
        quantity=1,
    )


def final_state(symbol: str) -> dict:
    return {
        "company_of_interest": symbol,
        "trade_date": "2026-05-27",
        "market_report": f"market report for {symbol}",
        "sentiment_report": f"sentiment report for {symbol}",
        "news_report": f"news report for {symbol}",
        "fundamentals_report": f"fundamentals report for {symbol}",
        "investment_debate_state": {
            "bull_history": f"bull {symbol}",
            "bear_history": f"bear {symbol}",
            "history": f"research debate {symbol}",
            "current_response": f"research response {symbol}",
            "judge_decision": f"research judge {symbol}",
        },
        "risk_debate_state": {
            "aggressive_history": f"aggressive {symbol}",
            "conservative_history": f"conservative {symbol}",
            "neutral_history": f"neutral {symbol}",
            "history": f"risk debate {symbol}",
            "judge_decision": f"risk judge {symbol}",
        },
        "investment_plan": f"investment plan for {symbol}",
        "trader_investment_plan": f"trader plan for {symbol}",
        "final_trade_decision": f"Rating: Buy\nDecision for {symbol}",
    }


def graph_with_stubbed_propagate():
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    calls = []

    def propagate(symbol, trade_date, asset_type="stock"):
        calls.append((symbol, trade_date, asset_type))
        return final_state(symbol), f"signal for {symbol}"

    graph.propagate = propagate
    return graph, calls


@pytest.mark.unit
def test_propagate_portfolio_analyzes_stocks_and_skips_cash():
    graph, calls = graph_with_stubbed_propagate()
    request = PortfolioRequest(
        positions=[
            stock_position("AAPL", 0.60),
            cash_position(0.40),
        ],
        trade_date="2026-05-27",
    )

    result = graph.propagate_portfolio(request)

    assert calls == [("AAPL", "2026-05-27", "stock")]
    assert result["trade_date"] == "2026-05-27"
    assert result["base_currency"] == "USD"
    assert list(result["analyses_by_symbol"]) == ["AAPL"]

    stock = result["holdings"][0]
    assert stock["symbol"] == "AAPL"
    assert stock["analysis_status"] == "analyzed"
    assert stock["analysis_symbol"] == "AAPL"
    assert stock["analysis"]["signal"] == "signal for AAPL"
    assert stock["analysis"]["reports"]["market"] == "market report for AAPL"
    assert stock["analysis"]["trader_investment_plan"] == "trader plan for AAPL"
    assert stock["analysis"]["final_trade_decision"].startswith("Rating: Buy")

    cash = result["holdings"][1]
    assert cash["symbol"] == "CASH"
    assert cash["analysis_status"] == "skipped"
    assert cash["analysis"] is None
    assert "cash positions" in cash["skip_reason"]


@pytest.mark.unit
def test_propagate_portfolio_reuses_underlying_analysis_for_options():
    graph, calls = graph_with_stubbed_propagate()
    request = PortfolioRequest(
        positions=[
            stock_position("AAPL", 0.50),
            option_position("AAPL250620C00200000", "AAPL", 0.30),
            cash_position(0.20),
        ],
        trade_date="2026-05-27",
    )

    result = graph.propagate_portfolio(request)

    assert calls == [("AAPL", "2026-05-27", "stock")]
    option = result["holdings"][1]
    assert option["symbol"] == "AAPL250620C00200000"
    assert option["analysis_status"] == "underlying_analyzed"
    assert option["analysis_symbol"] == "AAPL"
    assert option["underlying_symbol"] == "AAPL"
    assert option["contract_analysis"] is None
    assert option["contract_analysis_status"] == "options_data_optional"
    assert option["analysis"] is result["analyses_by_symbol"]["AAPL"]


@pytest.mark.unit
def test_propagate_portfolio_continues_when_option_holding_analysis_fails():
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    calls = []

    def propagate(symbol, trade_date, asset_type="stock"):
        calls.append((symbol, trade_date, asset_type))
        if symbol == "BAD":
            raise RuntimeError("option underlying data unavailable")
        return final_state(symbol), f"signal for {symbol}"

    graph.propagate = propagate
    request = PortfolioRequest(
        positions=[
            option_position("BAD260620C00100000", "BAD", 0.30),
            stock_position("AAPL", 0.50),
            cash_position(0.20),
        ],
        trade_date="2026-05-27",
    )

    result = graph.propagate_portfolio(request)

    assert calls == [
        ("BAD", "2026-05-27", "stock"),
        ("AAPL", "2026-05-27", "stock"),
    ]
    option = result["holdings"][0]
    assert option["analysis_status"] == "failed"
    assert option["contract_analysis_status"] == "options_data_unavailable"
    assert "continued" in option["skip_reason"]

    stock = result["holdings"][1]
    assert stock["symbol"] == "AAPL"
    assert stock["analysis_status"] == "analyzed"
    assert list(result["analyses_by_symbol"]) == ["AAPL"]


@pytest.mark.unit
def test_propagate_portfolio_keeps_single_ticker_signature_compatible():
    params = signature(TradingAgentsGraph.propagate).parameters

    assert list(params) == ["self", "company_name", "trade_date", "asset_type"]
    assert params["asset_type"].default == "stock"
