from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import (
    create_portfolio_allocation_manager,
    create_portfolio_manager,
)
from tradingagents.agents.managers.portfolio_rebalancer import (
    create_portfolio_rebalancer,
)
from tradingagents.agents.risk_mgmt.portfolio_risk_analyst import (
    create_portfolio_risk_analyst,
)
from tradingagents.portfolio.decision_protocol import (
    PortfolioApprovalDecision,
    ReviewDecision,
)
from tradingagents.portfolio import (
    AssetType,
    PortfolioInstrument,
    PortfolioMarketContext,
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
        "portfolio_market_context": PortfolioMarketContext(
            benchmark_symbol="SPY",
            global_news="Macro news points to tighter liquidity.",
            benchmark_news="SPY news shows broad market risk appetite cooling.",
            benchmark_fundamentals="SPY fundamentals unavailable for ETF.",
            benchmark_indicators={"rsi": "RSI is elevated."},
        ),
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


def _structured_allocation_llm(captured: dict, state: dict):
    proposal = state["rebalance_proposal"]
    decision = PortfolioApprovalDecision(
        proposal_id=proposal.proposal_id,
        proposal_version=proposal.version,
        decision=ReviewDecision.APPROVE,
        summary="Trim AAPL and add MSFT.",
        rationale="The deterministic proposal satisfies the review evidence.",
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
    assert "Portfolio-Wide Market Context" in captured["prompt"]
    assert "tighter liquidity" in captured["prompt"]
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
    assert "Portfolio-Wide Market Context" in captured["prompt"]
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
        _structured_allocation_llm(captured, state)
    )

    result = manager_node(state)

    decision = result["final_portfolio_decision"]
    assert "**Portfolio Action**: Rebalance" in decision
    assert "**Decision**: Approved" in decision
    assert (
        f"| AAPL | {state['rebalance_proposal'].target_weights_by_symbol['AAPL']:.2%} | Trim |"
        in decision
    )
    assert (
        result["final_portfolio_result"].target_weights_by_symbol
        == state["rebalance_proposal"].target_weights_by_symbol
    )
    assert "Use the deterministic analytics" in captured["prompt"]
    assert "Portfolio-Wide Market Context" in captured["prompt"]
    assert "summary must be a portfolio-level recommendation" in captured["prompt"]


@pytest.mark.unit
def test_portfolio_allocation_manager_falls_back_to_freetext():
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("unsupported")
    llm.invoke.return_value = MagicMock(content="Free-text portfolio decision.")
    manager_node = create_portfolio_allocation_manager(llm)

    result = manager_node(_portfolio_state())

    assert result["final_portfolio_result"].approved is False
    assert result["final_portfolio_result"].target_weights_by_symbol == {}
    assert (
        "No executable trade list was produced." in result["final_portfolio_decision"]
    )


@pytest.mark.unit
def test_single_instrument_portfolio_manager_factory_still_available():
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("unsupported")

    node = create_portfolio_manager(llm)

    assert callable(node)
