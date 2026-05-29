import pytest

from tradingagents.portfolio import (
    AssetType,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
    collect_portfolio_analytics_inputs,
)


def stock_position(symbol: str, weight: float) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol=symbol, asset_type=AssetType.STOCK),
        current_weight=weight,
        quantity=10,
    )


def option_position(symbol: str, weight: float, underlying: str) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(
            symbol=symbol,
            asset_type=AssetType.OPTION,
            underlying=underlying,
            expiry="2026-06-19",
            strike=200,
            right="C",
        ),
        current_weight=weight,
        quantity=1,
    )


def cash_position(weight: float) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol="CASH", asset_type=AssetType.CASH),
        current_weight=weight,
        market_value=1000,
    )


@pytest.mark.unit
def test_collects_correlation_inputs_from_underlyings_and_benchmark(monkeypatch):
    request = PortfolioRequest(
        positions=[
            stock_position("AAPL", 0.40),
            option_position("AAPL260619C00200000", 0.20, "AAPL"),
            stock_position("MSFT", 0.30),
            cash_position(0.10),
        ],
        trade_date="2026-05-27",
    )
    calls = []

    def fake_route(method, *args):
        calls.append((method, args[0]))
        if method == "get_stock_data":
            symbol = args[0]
            if symbol == "AAPL":
                return (
                    "# Stock data for AAPL\n\n"
                    "Date,Open,High,Low,Close,Volume\n"
                    "2026-05-25,100,101,99,100,1000\n"
                    "2026-05-26,101,102,100,102,1000\n"
                    "2026-05-27,103,104,101,103,1000\n"
                )
            if symbol == "MSFT":
                return (
                    "timestamp,open,high,low,close,volume\n"
                    "2026-05-25,50,51,49,50,1000\n"
                    "2026-05-26,52,53,51,52,1000\n"
                    "2026-05-27,51,52,50,51,1000\n"
                )
            if symbol == "SPY":
                return (
                    "timestamp,open,high,low,close,volume\n"
                    "2026-05-25,400,401,399,400,1000\n"
                    "2026-05-26,402,403,401,402,1000\n"
                    "2026-05-27,403,404,402,403,1000\n"
                )
        if method == "get_fundamentals":
            symbol = args[0]
            return {
                "Sector": "Technology",
                "Industry": "Consumer Electronics" if symbol == "AAPL" else "Software",
            }
        raise AssertionError(f"unexpected route: {method} {args}")

    monkeypatch.setattr(
        "tradingagents.portfolio.data_inputs.route_to_vendor",
        fake_route,
    )

    inputs = collect_portfolio_analytics_inputs(
        request,
        config={"benchmark_ticker": None, "benchmark_map": {"": "SPY"}},
    )

    assert inputs.warnings == []
    assert inputs.historical_prices["AAPL"] == [100, 102, 103]
    assert inputs.historical_prices["AAPL260619C00200000"] == [100, 102, 103]
    assert inputs.historical_prices["MSFT"] == [50, 52, 51]
    assert inputs.benchmark_symbol == "SPY"
    assert inputs.benchmark_prices == [400, 402, 403]
    assert inputs.sector_by_symbol == {
        "AAPL": "Technology / Consumer Electronics",
        "MSFT": "Technology / Software",
    }
    assert ("get_stock_data", "AAPL260619C00200000") not in calls


@pytest.mark.unit
def test_collect_inputs_warns_and_continues_when_data_is_unavailable(monkeypatch):
    request = PortfolioRequest(
        positions=[stock_position("AAPL", 0.80), cash_position(0.20)],
        trade_date="2026-05-27",
    )

    def fake_route(method, *args):
        if method == "get_stock_data" and args[0] == "AAPL":
            return '{"Information": "premium endpoint"}'
        if method == "get_stock_data" and args[0] == "SPY":
            return (
                "timestamp,open,high,low,close,volume\n"
                "2026-05-26,400,401,399,400,1000\n"
                "2026-05-27,402,403,401,402,1000\n"
            )
        if method == "get_fundamentals":
            raise RuntimeError("provider offline")
        raise AssertionError(f"unexpected route: {method} {args}")

    monkeypatch.setattr(
        "tradingagents.portfolio.data_inputs.route_to_vendor",
        fake_route,
    )

    inputs = collect_portfolio_analytics_inputs(
        request,
        config={"benchmark_ticker": None, "benchmark_map": {"": "SPY"}},
    )

    assert inputs.historical_prices == {}
    assert inputs.benchmark_prices == [400, 402]
    assert inputs.sector_by_symbol == {}
    assert any("holding price history unavailable for AAPL" in warning for warning in inputs.warnings)
    assert any("sector label unavailable for AAPL" in warning for warning in inputs.warnings)
