from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tradingagents.portfolio import (
    AssetType,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
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
                quantity=1,
                current_weight=0.8,
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
def test_run_portfolio_workflow_is_reusable_and_emits_stable_events(monkeypatch):
    request = _request()
    graph = MagicMock()
    graph.deep_thinking_llm = MagicMock()

    def propagate_portfolio(request, progress_callback):
        progress_callback(
            "holding_started",
            {"symbol": "AAPL", "asset_type": "stock", "weight": 0.8},
        )
        progress_callback("holding_completed", {"symbol": "AAPL", "status": "analyzed"})
        return {
            "portfolio_request": request,
            "holdings": [{"symbol": "AAPL", "analysis_status": "analyzed"}],
        }

    graph.propagate_portfolio.side_effect = propagate_portfolio
    inputs = SimpleNamespace(
        historical_prices={},
        benchmark_prices=None,
        sector_by_symbol={},
        benchmark_symbol="SPY",
        warnings=["price warning"],
    )
    market_context = SimpleNamespace(warnings=["market warning"])
    analytics = object()
    proposal = object()
    monkeypatch.setattr(
        workflow, "collect_portfolio_analytics_inputs", lambda *args, **kwargs: inputs
    )
    monkeypatch.setattr(
        workflow, "calculate_portfolio_analytics", lambda *args, **kwargs: analytics
    )
    monkeypatch.setattr(
        workflow,
        "collect_portfolio_market_context",
        lambda *args, **kwargs: market_context,
    )
    monkeypatch.setattr(
        workflow,
        "extract_ratings_from_portfolio_result",
        lambda result: {"AAPL": "Buy"},
    )
    monkeypatch.setattr(
        workflow,
        "extract_holding_evidence_from_portfolio_result",
        lambda result: result["holdings"],
    )
    monkeypatch.setattr(
        workflow, "generate_rebalance_proposal", lambda *args, **kwargs: proposal
    )
    monkeypatch.setattr(
        workflow,
        "create_portfolio_risk_analyst",
        lambda llm: lambda state: {"portfolio_risk_analysis": "risk"},
    )
    monkeypatch.setattr(
        workflow,
        "create_portfolio_rebalancer",
        lambda llm: lambda state: {"portfolio_rebalance_review": "review"},
    )
    monkeypatch.setattr(
        workflow,
        "create_portfolio_allocation_manager",
        lambda llm: lambda state: {"final_portfolio_decision": "decision"},
    )
    emitted = []

    result = workflow.run_portfolio_workflow(
        request,
        selected_analysts=["market"],
        config={},
        graph=graph,
        progress_callback=emitted.append,
    )

    assert result["portfolio_analytics"] is analytics
    assert result["rebalance_proposal"] is proposal
    assert result["final_portfolio_decision"] == "decision"
    assert result["warnings"] == ["price warning", "market warning"]
    assert result["events"] == emitted
    assert {event["stage"] for event in emitted} == {
        "instrument_analysis",
        "portfolio_analytics",
        "deterministic_rebalance",
        "portfolio_review",
    }
    assert [
        event["payload"]["agent"]
        for event in emitted
        if event["name"] == "agent_started"
    ] == [
        "risk_controller",
        "allocation_proposal_reviewer",
        "portfolio_decision_approver",
    ]


@pytest.mark.unit
def test_run_portfolio_workflow_constructs_graph_for_programmatic_caller(monkeypatch):
    request = _request()
    graph = MagicMock()
    graph.deep_thinking_llm = MagicMock()
    graph.propagate_portfolio.return_value = {
        "portfolio_request": request,
        "holdings": [],
    }
    graph_factory = MagicMock(return_value=graph)
    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.TradingAgentsGraph", graph_factory
    )
    monkeypatch.setattr(
        workflow,
        "collect_portfolio_analytics_inputs",
        lambda *args, **kwargs: SimpleNamespace(
            historical_prices={},
            benchmark_prices=None,
            sector_by_symbol={},
            benchmark_symbol="SPY",
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        workflow, "calculate_portfolio_analytics", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        workflow,
        "collect_portfolio_market_context",
        lambda *args, **kwargs: SimpleNamespace(warnings=[]),
    )
    monkeypatch.setattr(
        workflow, "extract_ratings_from_portfolio_result", lambda result: {}
    )
    monkeypatch.setattr(
        workflow, "extract_holding_evidence_from_portfolio_result", lambda result: []
    )
    monkeypatch.setattr(
        workflow, "generate_rebalance_proposal", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        workflow, "create_portfolio_risk_analyst", lambda llm: lambda state: {}
    )
    monkeypatch.setattr(
        workflow, "create_portfolio_rebalancer", lambda llm: lambda state: {}
    )
    monkeypatch.setattr(
        workflow, "create_portfolio_allocation_manager", lambda llm: lambda state: {}
    )

    workflow.run_portfolio_workflow(
        request,
        selected_analysts=["market"],
        config={"key": "value"},
        callbacks=["callback"],
    )

    graph_factory.assert_called_once_with(
        ["market"], config={"key": "value"}, debug=False, callbacks=["callback"]
    )
