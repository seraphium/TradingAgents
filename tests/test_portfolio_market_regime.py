import pytest
from pydantic import ValidationError

from tradingagents.portfolio import (
    AssetType,
    MarketOverlay,
    MarketRegime,
    MarketRegimeLabel,
    PortfolioConstraints,
    PortfolioInstrument,
    PortfolioMarketContext,
    PortfolioPosition,
    PortfolioRequest,
    calculate_portfolio_analytics,
    derive_market_regime,
    generate_rebalance_proposal,
)


def stock_position(symbol: str, weight: float, **kwargs) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol=symbol, asset_type=AssetType.STOCK),
        current_weight=weight,
        quantity=1,
        **kwargs,
    )


def cash_position(weight: float, **kwargs) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol="CASH", asset_type=AssetType.CASH),
        current_weight=weight,
        **kwargs,
    )


def request() -> PortfolioRequest:
    return PortfolioRequest(
        trade_date="2026-05-27",
        positions=[
            stock_position("AAPL", 0.40),
            stock_position("XOM", 0.40),
            cash_position(0.20),
        ],
        constraints=PortfolioConstraints(min_cash_weight=0.15, max_cash_weight=0.25),
    )


@pytest.mark.unit
def test_risk_on_and_risk_off_regimes_produce_predictably_different_proposals():
    portfolio_request = request()
    analytics = calculate_portfolio_analytics(portfolio_request)
    context = PortfolioMarketContext(benchmark_symbol="SPY")
    risk_on = derive_market_regime(
        context, benchmark_prices=[100 + index for index in range(220)]
    )
    risk_off = derive_market_regime(
        context, benchmark_prices=[320 - index for index in range(220)]
    )

    on_proposal = generate_rebalance_proposal(
        portfolio_request,
        analytics,
        ratings_by_symbol={"AAPL": "Hold", "XOM": "Hold"},
        market_regime=risk_on,
    )
    off_proposal = generate_rebalance_proposal(
        portfolio_request,
        analytics,
        ratings_by_symbol={"AAPL": "Hold", "XOM": "Hold"},
        market_regime=risk_off,
    )

    assert risk_on.label == MarketRegimeLabel.RISK_ON
    assert risk_off.label == MarketRegimeLabel.RISK_OFF
    assert (
        on_proposal.target_weights_by_symbol["CASH"]
        < off_proposal.target_weights_by_symbol["CASH"]
    )
    assert off_proposal.diagnostics["market_regime"]["label"] == "risk_off"
    assert any(
        "market_regime_overlay" in component.constraints_applied
        for component in off_proposal.component_proposals
    )


@pytest.mark.unit
def test_market_overlay_cannot_violate_hard_cash_or_component_constraints():
    portfolio_request = PortfolioRequest(
        trade_date="2026-05-27",
        positions=[
            stock_position("AAPL", 0.75, target_min_weight=0.70),
            cash_position(0.25),
        ],
        constraints=PortfolioConstraints(min_cash_weight=0.20, max_cash_weight=0.25),
    )
    analytics = calculate_portfolio_analytics(portfolio_request)
    regime = MarketRegime(
        label=MarketRegimeLabel.RISK_ON,
        confidence=1.0,
        summary="Risk on.",
        overlay=MarketOverlay(risk_adjustment=1.0, cash_weight_adjustment=-0.05),
    )

    proposal = generate_rebalance_proposal(
        portfolio_request,
        analytics,
        ratings_by_symbol={"AAPL": "Buy"},
        market_regime=regime,
    )

    assert proposal.target_weights_by_symbol["CASH"] >= 0.20
    assert proposal.target_weights_by_symbol["AAPL"] >= 0.70
    assert sum(proposal.target_weights_by_symbol.values()) == pytest.approx(1.0)


@pytest.mark.unit
def test_low_confidence_market_context_has_no_allocation_impact():
    portfolio_request = request()
    analytics = calculate_portfolio_analytics(portfolio_request)
    low_confidence = derive_market_regime(
        PortfolioMarketContext(benchmark_symbol="SPY", global_news="market update"),
        benchmark_prices=[100.0, 101.0],
    )
    baseline = generate_rebalance_proposal(
        portfolio_request, analytics, ratings_by_symbol={"AAPL": "Hold", "XOM": "Hold"}
    )
    proposal = generate_rebalance_proposal(
        portfolio_request,
        analytics,
        ratings_by_symbol={"AAPL": "Hold", "XOM": "Hold"},
        market_regime=low_confidence,
    )

    assert low_confidence.confidence < 0.25
    assert low_confidence.overlay.cash_weight_adjustment == 0.0
    assert proposal.target_weights_by_symbol == baseline.target_weights_by_symbol


@pytest.mark.unit
def test_market_overlay_schema_rejects_unbounded_or_low_confidence_adjustments():
    with pytest.raises(ValidationError):
        MarketOverlay(cash_weight_adjustment=0.06)
    with pytest.raises(ValidationError):
        MarketRegime(
            label=MarketRegimeLabel.RISK_OFF,
            confidence=0.1,
            summary="Uncertain.",
            overlay=MarketOverlay(cash_weight_adjustment=0.01),
        )


@pytest.mark.unit
def test_validated_sector_overlay_shifts_only_mapped_sector_before_constraints():
    portfolio_request = request()
    analytics = calculate_portfolio_analytics(portfolio_request)
    regime = MarketRegime(
        label=MarketRegimeLabel.RISK_ON,
        confidence=1.0,
        summary="Favor energy.",
        overlay=MarketOverlay(sector_weight_adjustments={"Energy": 0.03}),
    )

    proposal = generate_rebalance_proposal(
        portfolio_request,
        analytics,
        ratings_by_symbol={"AAPL": "Hold", "XOM": "Hold"},
        market_regime=regime,
        sector_by_symbol={"AAPL": "Technology", "XOM": "Energy"},
    )

    assert proposal.target_weights_by_symbol["XOM"] > 0.40
    assert proposal.target_weights_by_symbol["AAPL"] < 0.40
    assert sum(proposal.target_weights_by_symbol.values()) == pytest.approx(1.0)
