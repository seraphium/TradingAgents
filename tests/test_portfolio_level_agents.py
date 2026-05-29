from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import (
    create_portfolio_allocation_manager,
    create_portfolio_manager,
)
from tradingagents.agents.managers.portfolio_rebalancer import create_portfolio_rebalancer
from tradingagents.agents.risk_mgmt.portfolio_risk_analyst import (
    create_portfolio_risk_analyst,
)
from tradingagents.agents.schemas import (
    ComponentAction,
    ComponentRecommendation,
    PortfolioAllocationAction,
    PortfolioAllocationDecision,
)
from tradingagents.portfolio import (
    AssetType,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
    calculate_portfolio_analytics,
    generate_rebalance_proposal,
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


def _portfolio_state() -> dict:
    request = PortfolioRequest(
        positions=[
            stock_position("AAPL", 0.60),
            stock_position("MSFT", 0.20),
            cash_position(0.20),
        ],
        trade_date="2026-05-27",
    )
    analytics = calculate_portfolio_analytics(request)
    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol={"AAPL": "Underweight", "MSFT": "Overweight"},
    )
    return {
        "portfolio_request": request,
        "portfolio_analytics": analytics,
        "rebalance_proposal": proposal,
        "holdings": [
            {
                "symbol": "AAPL",
                "asset_type": "stock",
                "current_weight": 0.60,
                "analysis_status": "analyzed",
                "analysis_symbol": "AAPL",
                "analysis": {
                    "signal": "Underweight",
                    "final_trade_decision": "**Rating**: Underweight",
                    "trader_investment_plan": "Trader recommends trimming AAPL.",
                    "investment_debate_state": {
                        "judge_decision": "Research manager is cautious on AAPL.",
                    },
                    "risk_debate_state": {
                        "judge_decision": "Risk manager recommends smaller AAPL exposure.",
                        "aggressive_history": "Aggressive analyst would keep upside exposure.",
                        "conservative_history": "Conservative analyst wants a larger trim.",
                        "neutral_history": "Neutral analyst supports a moderate trim.",
                    },
                },
            },
            {
                "symbol": "MSFT",
                "asset_type": "stock",
                "current_weight": 0.20,
                "analysis_status": "analyzed",
                "analysis_symbol": "MSFT",
                "analysis": {
                    "signal": "Overweight",
                    "final_trade_decision": "**Rating**: Overweight",
                    "risk_debate_state": {
                        "judge_decision": "Risk manager accepts adding MSFT.",
                    },
                },
            },
        ],
    }


def _plain_llm(response: str, captured: dict):
    llm = MagicMock()
    llm.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or MagicMock(content=response)
    )
    return llm


def _structured_allocation_llm(captured: dict):
    decision = PortfolioAllocationDecision(
        portfolio_action=PortfolioAllocationAction.REBALANCE,
        summary="Trim AAPL and add MSFT.",
        component_recommendations=[
            ComponentRecommendation(
                symbol="AAPL",
                current_weight=0.60,
                target_weight=0.55,
                weight_change=-0.05,
                action=ComponentAction.TRIM,
                component_summary="Single-stock PM and risk debate support trimming AAPL.",
                rationale="Concentration should be reduced.",
            ),
            ComponentRecommendation(
                symbol="MSFT",
                current_weight=0.20,
                target_weight=0.25,
                weight_change=0.05,
                action=ComponentAction.ADD,
                component_summary="Single-stock PM supports adding MSFT.",
                rationale="Improves diversification.",
            ),
            ComponentRecommendation(
                symbol="CASH",
                current_weight=0.20,
                target_weight=0.20,
                weight_change=0.0,
                action=ComponentAction.HOLD,
                component_summary="Cash remains the liquidity reserve.",
                rationale="Cash reserve is adequate.",
            ),
        ],
        risk_notes="AAPL concentration is the main residual risk.",
    )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or decision
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
def test_portfolio_risk_analyst_prompt_uses_deterministic_metrics():
    captured = {}
    risk_node = create_portfolio_risk_analyst(
        _plain_llm("Risk review markdown.", captured)
    )

    result = risk_node(_portfolio_state())

    assert result["portfolio_risk_analysis"] == "Risk review markdown."
    assert "using only the supplied deterministic metrics" in captured["prompt"]
    assert "Single-Stock Decision Evidence" in captured["prompt"]
    assert "Conservative analyst wants a larger trim" in captured["prompt"]
    assert "Deterministic Portfolio Analytics" in captured["prompt"]
    assert "AAPL" in captured["prompt"]


@pytest.mark.unit
def test_portfolio_rebalancer_prompt_reviews_target_weights():
    captured = {}
    state = _portfolio_state()
    state["portfolio_risk_analysis"] = "Risk analyst says concentration is high."
    rebalance_node = create_portfolio_rebalancer(
        _plain_llm("Rebalance review markdown.", captured)
    )

    result = rebalance_node(state)

    assert result["portfolio_rebalance_review"] == "Rebalance review markdown."
    assert "numeric source of truth" in captured["prompt"]
    assert "Single-Stock Decision Evidence" in captured["prompt"]
    assert "Risk manager recommends smaller AAPL exposure" in captured["prompt"]
    assert "Deterministic Rebalance Proposal" in captured["prompt"]
    assert "Risk analyst says concentration is high." in captured["prompt"]


@pytest.mark.unit
def test_portfolio_allocation_manager_returns_structured_final_decision():
    captured = {}
    state = _portfolio_state()
    state["portfolio_risk_analysis"] = "Risk notes."
    state["portfolio_rebalance_review"] = "Rebalance review."
    manager_node = create_portfolio_allocation_manager(
        _structured_allocation_llm(captured)
    )

    result = manager_node(state)

    decision = result["final_portfolio_decision"]
    assert "**Portfolio Action**: Rebalance" in decision
    assert "**Component Summary**:" in decision
    assert "Single-stock PM and risk debate support trimming AAPL" in decision
    assert "| AAPL | 60.00% | 55.00% | -5.00% | Trim |" in decision
    assert "Use the deterministic analytics" in captured["prompt"]
    assert "single-stock Portfolio Manager decisions" in captured["prompt"]
    assert "**Rating**: Underweight" in captured["prompt"]
    assert "summary must be a portfolio-level recommendation" in captured["prompt"]


@pytest.mark.unit
def test_portfolio_allocation_manager_falls_back_to_freetext():
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("unsupported")
    llm.invoke.return_value = MagicMock(content="Free-text portfolio decision.")
    manager_node = create_portfolio_allocation_manager(llm)

    result = manager_node(_portfolio_state())

    assert result["final_portfolio_decision"] == "Free-text portfolio decision."


@pytest.mark.unit
def test_single_instrument_portfolio_manager_factory_still_available():
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("unsupported")

    node = create_portfolio_manager(llm)

    assert callable(node)
