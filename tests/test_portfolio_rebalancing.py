import pytest

from tradingagents.portfolio import (
    AssetType,
    PortfolioConstraints,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
    calculate_portfolio_analytics,
    extract_ratings_from_portfolio_result,
    generate_rebalance_proposal,
    render_rebalance_proposal,
)


def stock_position(
    symbol: str,
    weight: float,
    *,
    target_min_weight: float | None = None,
    target_max_weight: float | None = None,
) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol=symbol, asset_type=AssetType.STOCK),
        current_weight=weight,
        quantity=10,
        target_min_weight=target_min_weight,
        target_max_weight=target_max_weight,
    )


def option_position(symbol: str, weight: float) -> PortfolioPosition:
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
    )


def cash_position(weight: float) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol="CASH", asset_type=AssetType.CASH),
        current_weight=weight,
        market_value=1000,
    )


def request_with_constraints() -> PortfolioRequest:
    return PortfolioRequest(
        positions=[
            stock_position("AAPL", 0.45, target_max_weight=0.46),
            stock_position("MSFT", 0.27, target_min_weight=0.24),
            option_position("AAPL260620C00200000", 0.08),
            cash_position(0.20),
        ],
        trade_date="2026-05-27",
        constraints=PortfolioConstraints(
            max_single_position_weight=0.50,
            max_options_weight=0.08,
            min_cash_weight=0.15,
            custom={"max_portfolio_volatility": 0.20},
        ),
    )


@pytest.mark.unit
def test_rebalance_converts_ratings_to_target_weights_and_actions():
    request = PortfolioRequest(
        positions=[
            stock_position("AAPL", 0.40),
            stock_position("MSFT", 0.30),
            cash_position(0.30),
        ],
        trade_date="2026-05-27",
    )
    analytics = calculate_portfolio_analytics(request)

    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol={"AAPL": "Rating: Buy", "MSFT": "Rating: Sell"},
    )

    assert proposal.score_by_symbol["AAPL"] == pytest.approx(1.0)
    assert proposal.score_by_symbol["MSFT"] == pytest.approx(-1.0)
    assert proposal.target_weights_by_symbol["AAPL"] > 0.40
    assert proposal.target_weights_by_symbol["MSFT"] < 0.30
    assert {component.symbol: component.action for component in proposal.component_proposals} == {
        "AAPL": "Add",
        "MSFT": "Trim",
        "CASH": "Hold",
    }
    assert sum(proposal.target_weights_by_symbol.values()) == pytest.approx(1.0)


@pytest.mark.unit
def test_rebalance_penalizes_volatility_correlation_concentration_and_liquidity():
    request = request_with_constraints()
    analytics = calculate_portfolio_analytics(
        request,
        historical_prices={
            "AAPL": [100, 130, 90, 140],
            "MSFT": [50, 53, 56, 59],
            "AAPL260620C00200000": [10, 15, 7, 16],
        },
    )

    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol={
            "AAPL": "Buy",
            "MSFT": "Overweight",
            "AAPL260620C00200000": "Buy",
        },
        liquidity_by_symbol={"AAPL260620C00200000": 0.10},
    )

    option = next(
        component
        for component in proposal.component_proposals
        if component.symbol == "AAPL260620C00200000"
    )
    assert option.target_weight <= 0.08
    assert option.action == "Trim"
    assert any("volatility" in note for note in proposal.risk_notes)
    assert any("concentrated" in note for note in proposal.risk_notes)
    assert any("liquidity score" in note for note in proposal.risk_notes)


@pytest.mark.unit
def test_rebalance_caps_proposed_option_exposure():
    request = PortfolioRequest(
        positions=[
            stock_position("MSFT", 0.72),
            option_position("AAPL260620C00200000", 0.08),
            cash_position(0.20),
        ],
        trade_date="2026-05-27",
        constraints=PortfolioConstraints(max_options_weight=0.08),
    )
    analytics = calculate_portfolio_analytics(request)

    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol={"MSFT": "Hold", "AAPL260620C00200000": "Buy"},
    )

    option = next(
        component
        for component in proposal.component_proposals
        if component.symbol == "AAPL260620C00200000"
    )
    assert option.target_weight == pytest.approx(0.08)
    assert "max_options_weight" in option.constraints_applied


@pytest.mark.unit
def test_rebalance_labels_cash_reduction_as_increase_risk():
    request = PortfolioRequest(
        positions=[
            stock_position("AAPL", 0.50),
            cash_position(0.50),
        ],
        trade_date="2026-05-27",
    )
    analytics = calculate_portfolio_analytics(request)

    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol={"AAPL": "Buy"},
    )

    assert proposal.portfolio_action == "Increase Risk"
    assert proposal.target_weights_by_symbol["CASH"] < 0.50


@pytest.mark.unit
def test_rebalance_applies_risk_budget_to_cash_when_portfolio_volatility_is_high():
    request = request_with_constraints()
    analytics = calculate_portfolio_analytics(
        request,
        historical_prices={
            "AAPL": [100, 150, 80, 160],
            "MSFT": [50, 75, 40, 80],
            "AAPL260620C00200000": [10, 18, 5, 20],
        },
    )

    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol={"AAPL": "Hold", "MSFT": "Hold", "AAPL260620C00200000": "Hold"},
    )

    assert proposal.portfolio_action == "De-risk"
    assert proposal.target_weights_by_symbol["CASH"] > 0.20
    assert any(
        "risk_budget" in component.constraints_applied
        for component in proposal.component_proposals
    )


@pytest.mark.unit
def test_mean_variance_optimizer_sets_targets_and_diagnostics():
    request = PortfolioRequest(
        positions=[
            stock_position("AAPL", 0.35, target_max_weight=0.50),
            stock_position("MSFT", 0.35, target_min_weight=0.20),
            cash_position(0.30),
        ],
        trade_date="2026-05-27",
    )
    analytics = calculate_portfolio_analytics(
        request,
        historical_prices={
            "AAPL": [100, 101, 102, 103, 104],
            "MSFT": [100, 99, 98, 97, 96],
        },
    )

    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol={"AAPL": "Buy", "MSFT": "Sell"},
        optimizer="mean_variance",
    )

    assert proposal.diagnostics["optimizer"] == "mean_variance"
    assert proposal.diagnostics["optimizer_iterations"] > 0
    assert proposal.target_weights_by_symbol["AAPL"] > 0.35
    assert proposal.target_weights_by_symbol["MSFT"] < 0.35
    assert proposal.target_weights_by_symbol["AAPL"] <= 0.50
    assert proposal.target_weights_by_symbol["MSFT"] >= 0.20
    assert sum(proposal.target_weights_by_symbol.values()) == pytest.approx(1.0)
    assert any(
        "mean_variance_optimizer" in component.constraints_applied
        for component in proposal.component_proposals
    )


@pytest.mark.unit
def test_mean_variance_optimizer_penalizes_high_volatility_equal_rating():
    request = PortfolioRequest(
        positions=[
            stock_position("LOWVOL", 0.35),
            stock_position("HIGHVOL", 0.35),
            cash_position(0.30),
        ],
        trade_date="2026-05-27",
    )
    analytics = calculate_portfolio_analytics(
        request,
        historical_prices={
            "LOWVOL": [100, 101, 102, 103, 104],
            "HIGHVOL": [100, 140, 80, 150, 70],
        },
    )

    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol={"LOWVOL": "Buy", "HIGHVOL": "Buy"},
        optimizer="mean_variance",
    )

    assert proposal.target_weights_by_symbol["LOWVOL"] > proposal.target_weights_by_symbol["HIGHVOL"]


@pytest.mark.unit
def test_extract_ratings_from_portfolio_result_and_render_markdown():
    ratings = extract_ratings_from_portfolio_result(
        {
            "holdings": [
                {
                    "symbol": "AAPL",
                    "analysis": {"final_trade_decision": "**Rating**: Overweight"},
                },
                {
                    "symbol": "CASH",
                    "analysis": None,
                },
            ]
        }
    )

    assert ratings == {"AAPL": "Overweight", "CASH": "Hold"}

    request = PortfolioRequest(
        positions=[stock_position("AAPL", 0.80), cash_position(0.20)],
        trade_date="2026-05-27",
    )
    proposal = generate_rebalance_proposal(
        request,
        calculate_portfolio_analytics(request),
        ratings_by_symbol=ratings,
    )
    markdown = render_rebalance_proposal(proposal)

    assert "**Portfolio Action**:" in markdown
    assert "| AAPL |" in markdown
