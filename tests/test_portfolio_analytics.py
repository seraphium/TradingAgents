import pytest

from tradingagents.portfolio import (
    AssetType,
    OptionGreeks,
    PortfolioConstraints,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
    calculate_portfolio_analytics,
    portfolio_analytics_to_dict,
)


def stock_position(
    symbol: str,
    weight: float,
    *,
    market_value: float | None = None,
    target_min_weight: float | None = None,
    target_max_weight: float | None = None,
) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol=symbol, asset_type=AssetType.STOCK),
        current_weight=weight,
        quantity=10,
        market_value=market_value,
        target_min_weight=target_min_weight,
        target_max_weight=target_max_weight,
    )


def option_position(
    symbol: str,
    weight: float,
    *,
    market_value: float | None = None,
) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(
            symbol=symbol,
            asset_type=AssetType.OPTION,
            underlying="AAPL",
            expiry="2026-06-20",
            strike=200,
            right="C",
        ),
        current_weight=weight,
        quantity=1,
        market_value=market_value,
    )


def cash_position(weight: float, *, market_value: float | None = None) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol="CASH", asset_type=AssetType.CASH),
        current_weight=weight,
        market_value=market_value,
    )


def portfolio_request() -> PortfolioRequest:
    return PortfolioRequest(
        positions=[
            stock_position("AAPL", 0.50, market_value=5000, target_max_weight=0.45),
            stock_position("MSFT", 0.20, market_value=2000, target_min_weight=0.25),
            option_position("AAPL260620C00200000", 0.10, market_value=1000),
            cash_position(0.20, market_value=2000),
        ],
        trade_date="2026-05-27",
        constraints=PortfolioConstraints(max_single_position_weight=0.60),
    )


@pytest.mark.unit
def test_calculates_weights_market_values_and_concentration():
    analytics = calculate_portfolio_analytics(
        portfolio_request(),
        sector_by_symbol={"AAPL": "Technology", "MSFT": "Technology"},
    )

    assert analytics.total_market_value == pytest.approx(10000)
    assert analytics.weights_by_symbol["AAPL"] == pytest.approx(0.50)
    assert analytics.computed_weights_by_symbol["MSFT"] == pytest.approx(0.20)
    assert analytics.market_values_by_symbol["CASH"] == pytest.approx(2000)
    assert analytics.asset_type_exposure == {
        "stock": pytest.approx(0.70),
        "option": pytest.approx(0.10),
        "cash": pytest.approx(0.20),
    }
    assert analytics.underlying_exposure["AAPL"] == pytest.approx(0.60)
    assert analytics.underlying_exposure["MSFT"] == pytest.approx(0.20)
    assert analytics.sector_exposure["Technology"] == pytest.approx(0.80)


@pytest.mark.unit
def test_calculates_returns_volatility_correlation_beta_and_risk_contribution():
    analytics = calculate_portfolio_analytics(
        portfolio_request(),
        historical_prices={
            "AAPL": [100, 110, 105, 120],
            "MSFT": [50, 55, 60, 58],
            "AAPL260620C00200000": [10, 12, 9, 13],
        },
        benchmark_prices=[100, 102, 101, 104],
    )

    assert analytics.historical_returns_by_symbol["AAPL"][0] == pytest.approx(0.10)
    assert analytics.volatility_by_symbol["AAPL"] is not None
    assert analytics.correlation_matrix["AAPL"]["AAPL"] == pytest.approx(1.0)
    assert analytics.correlation_matrix["AAPL"]["MSFT"] is not None
    assert analytics.beta_by_symbol["AAPL"] is not None
    assert analytics.portfolio_volatility is not None
    assert analytics.risk_contribution_by_symbol["CASH"] == pytest.approx(0.0)

    risky_contribution = sum(
        analytics.risk_contribution_by_symbol[symbol]
        for symbol in ("AAPL", "MSFT", "AAPL260620C00200000")
    )
    assert risky_contribution == pytest.approx(1.0)


@pytest.mark.unit
def test_calculates_delta_adjusted_option_exposure_and_aggregate_greeks():
    analytics = calculate_portfolio_analytics(
        portfolio_request(),
        option_greeks={
            "AAPL260620C00200000": OptionGreeks(
                delta=0.55,
                gamma=0.02,
                theta=-0.01,
                vega=0.12,
                rho=0.03,
            )
        },
    )

    assert analytics.delta_adjusted_exposure_by_symbol["AAPL"] == pytest.approx(0.50)
    assert analytics.delta_adjusted_exposure_by_symbol["AAPL260620C00200000"] == pytest.approx(0.055)
    assert analytics.portfolio_delta_adjusted_exposure == pytest.approx(0.755)
    assert analytics.option_greeks_by_symbol["AAPL260620C00200000"]["delta"] == pytest.approx(0.55)
    assert analytics.aggregate_option_greeks["delta"] == pytest.approx(0.055)
    assert analytics.aggregate_option_greeks["theta"] == pytest.approx(-0.001)


@pytest.mark.unit
def test_reports_position_constraint_flags_and_serializes_to_dict():
    analytics = calculate_portfolio_analytics(portfolio_request())

    assert [flag.kind for flag in analytics.constraint_flags] == [
        "overweight",
        "underweight",
    ]
    assert analytics.constraint_flags[0].symbol == "AAPL"

    as_dict = portfolio_analytics_to_dict(analytics)
    assert as_dict["constraint_flags"][0]["kind"] == "overweight"
    assert as_dict["weights_by_symbol"]["AAPL"] == pytest.approx(0.50)
