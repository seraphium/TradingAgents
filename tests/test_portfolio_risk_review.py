from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tradingagents.portfolio import (
    AssetType,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
)
from tradingagents.portfolio.risk_review import (
    ConstraintAdjustmentRequest,
    extract_constraint_adjustment_request,
    validate_constraint_adjustment_requests,
)
from tradingagents.portfolio import workflow


def _request() -> PortfolioRequest:
    return PortfolioRequest(
        trade_date="2025-01-02",
        positions=[
            PortfolioPosition(
                instrument=PortfolioInstrument(
                    symbol="AAPL", asset_type=AssetType.STOCK
                ),
                current_weight=0.8,
                quantity=1,
            ),
            PortfolioPosition(
                instrument=PortfolioInstrument(
                    symbol="CASH", asset_type=AssetType.CASH
                ),
                current_weight=0.2,
            ),
        ],
    )


@pytest.mark.unit
def test_extracts_and_validates_bounded_tighter_constraint_request():
    request = extract_constraint_adjustment_request(
        'Review.\n```json\n{"constraint_adjustments":{"reason":"Reduce concentration",'
        '"max_single_position_weight":0.75,"min_cash_weight":0.25}}\n```'
    )

    assert request is not None
    validated = validate_constraint_adjustment_requests(_request(), [("risk", request)])

    assert validated is not None
    assert validated.adjustments == {
        "max_single_position_weight": 0.75,
        "min_cash_weight": 0.25,
    }
    assert validated.reasons == ["risk: Reduce concentration"]


@pytest.mark.unit
def test_rejects_relaxed_or_out_of_bound_constraint_requests():
    portfolio_request = _request()
    too_large = ConstraintAdjustmentRequest(reason="Too large", min_cash_weight=0.35)

    assert (
        validate_constraint_adjustment_requests(
            portfolio_request, [("risk", too_large)], max_adjustment=0.10
        )
        is None
    )


@pytest.mark.unit
def test_workflow_reruns_optimizer_once_and_preserves_proposal_history(monkeypatch):
    request = _request()
    graph = MagicMock()
    graph.deep_thinking_llm = MagicMock()
    graph.propagate_portfolio.return_value = {
        "portfolio_request": request,
        "holdings": [],
    }
    inputs = SimpleNamespace(
        historical_prices={},
        benchmark_prices=None,
        sector_by_symbol={},
        benchmark_symbol="SPY",
        warnings=[],
    )
    monkeypatch.setattr(
        workflow, "collect_portfolio_analytics_inputs", lambda *a, **k: inputs
    )
    monkeypatch.setattr(
        workflow, "calculate_portfolio_analytics", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        workflow,
        "collect_portfolio_market_context",
        lambda *a, **k: SimpleNamespace(warnings=[]),
    )
    monkeypatch.setattr(
        workflow, "extract_ratings_from_portfolio_result", lambda result: {}
    )
    monkeypatch.setattr(workflow, "extract_instrument_proposals", lambda result: {})
    monkeypatch.setattr(
        workflow, "extract_holding_evidence_from_portfolio_result", lambda result: []
    )
    proposals = [SimpleNamespace(name="v1"), SimpleNamespace(name="v2")]
    generate = MagicMock(side_effect=proposals)
    monkeypatch.setattr(workflow, "generate_rebalance_proposal", generate)
    calls = {"risk": 0}

    def risk_node(state):
        calls["risk"] += 1
        result = {"portfolio_risk_analysis": f"risk {calls['risk']}"}
        if calls["risk"] == 1:
            result["risk_constraint_request"] = ConstraintAdjustmentRequest(
                reason="Reduce concentration", max_single_position_weight=0.75
            )
        return result

    monkeypatch.setattr(
        workflow, "create_portfolio_risk_analyst", lambda llm: risk_node
    )
    monkeypatch.setattr(
        workflow,
        "create_portfolio_rebalancer",
        lambda llm: lambda state: {"portfolio_rebalance_review": "review"},
    )
    monkeypatch.setattr(
        workflow,
        "create_portfolio_allocation_manager",
        lambda llm: lambda state: {"final_portfolio_decision": "approved"},
    )

    result = workflow.run_portfolio_workflow(
        request, selected_analysts=[], config={}, graph=graph
    )

    assert generate.call_count == 2
    assert generate.call_args_list[1].kwargs["constraint_overrides"] == {
        "max_single_position_weight": 0.75
    }
    assert result["rebalance_proposal"] is proposals[1]
    assert result["reoptimization_count"] == 1
    assert [entry["version"] for entry in result["proposal_history"]] == [1, 2]
    assert (
        result["proposal_history"][1]["reason"]
        == "risk_controller: Reduce concentration"
    )
    assert calls["risk"] == 2


@pytest.mark.unit
def test_optimizer_applies_validated_constraint_overrides_deterministically():
    from tradingagents.portfolio import (
        calculate_portfolio_analytics,
        generate_rebalance_proposal,
    )

    request = PortfolioRequest(
        trade_date="2025-01-02",
        positions=[
            PortfolioPosition(
                instrument=PortfolioInstrument(
                    symbol="AAPL", asset_type=AssetType.STOCK
                ),
                current_weight=0.6,
                quantity=1,
            ),
            PortfolioPosition(
                instrument=PortfolioInstrument(
                    symbol="MSFT", asset_type=AssetType.STOCK
                ),
                current_weight=0.2,
                quantity=1,
            ),
            PortfolioPosition(
                instrument=PortfolioInstrument(
                    symbol="CASH", asset_type=AssetType.CASH
                ),
                current_weight=0.2,
            ),
        ],
    )
    analytics = calculate_portfolio_analytics(request)

    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol={"AAPL": "Buy", "MSFT": "Hold"},
        constraint_overrides={
            "max_single_position_weight": 0.55,
            "min_cash_weight": 0.25,
        },
    )

    assert proposal.target_weights_by_symbol["AAPL"] <= 0.55
    assert proposal.target_weights_by_symbol["CASH"] >= 0.25
    assert proposal.diagnostics["constraint_overrides"] == {
        "max_single_position_weight": 0.55,
        "min_cash_weight": 0.25,
    }
