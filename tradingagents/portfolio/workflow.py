"""Reusable orchestration for the existing portfolio analysis workflow.

This module intentionally preserves the current allocation behavior while keeping
business orchestration independent from CLI rendering, prompts, and persistence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypedDict
from uuid import uuid4

from tradingagents.agents.managers.portfolio_manager import (
    create_portfolio_allocation_manager,
)
from tradingagents.agents.managers.portfolio_rebalancer import (
    create_portfolio_rebalancer,
)
from tradingagents.agents.risk_mgmt.portfolio_risk_analyst import (
    create_portfolio_risk_analyst,
)
from tradingagents.portfolio.analytics import (
    PortfolioAnalytics,
    calculate_portfolio_analytics,
)
from tradingagents.portfolio.data_quality import (
    DataQualityAssessment,
    DataQualityPolicy,
    assess_data_quality,
)
from tradingagents.portfolio.data_inputs import (
    PortfolioAnalyticsInputs,
    collect_portfolio_analytics_inputs,
)
from tradingagents.portfolio.instrument_proposals import (
    InstrumentProposal,
    extract_instrument_proposals,
)
from tradingagents.portfolio.market_regime import (
    MarketRegime,
    create_market_regime_agent,
)
from tradingagents.portfolio.market_context import (
    PortfolioMarketContext,
    collect_portfolio_market_context,
)
from tradingagents.portfolio.rebalancing import (
    RebalanceProposal,
    extract_holding_evidence_from_portfolio_result,
    extract_ratings_from_portfolio_result,
    generate_rebalance_proposal,
)
from tradingagents.portfolio.schemas import PortfolioRequest

if TYPE_CHECKING:
    from tradingagents.graph.trading_graph import TradingAgentsGraph


class PortfolioWorkflowEvent(TypedDict):
    """A stable progress event emitted by the portfolio workflow."""

    name: str
    stage: str
    payload: dict[str, Any]


class PortfolioWorkflowState(TypedDict, total=False):
    """Typed, progressively populated state for a portfolio workflow run.

    Legacy keys returned by ``propagate_portfolio`` and consumed by existing
    reports remain present alongside the explicitly typed workflow fields.
    """

    run_id: str
    portfolio_request: PortfolioRequest
    trade_date: str
    base_currency: str
    holdings: list[dict[str, Any]]
    portfolio_analytics_inputs: PortfolioAnalyticsInputs
    portfolio_analytics: PortfolioAnalytics
    portfolio_market_context: PortfolioMarketContext
    market_regime: MarketRegime
    ratings_by_symbol: dict[str, str]
    instrument_proposals: dict[str, InstrumentProposal]
    data_quality_assessment: DataQualityAssessment
    rebalance_proposal: RebalanceProposal
    portfolio_risk_analysis: str
    portfolio_rebalance_review: str
    final_portfolio_decision: str
    warnings: list[str]
    events: list[PortfolioWorkflowEvent]


ProgressCallback = Callable[[PortfolioWorkflowEvent], None]

STAGE_INSTRUMENT_ANALYSIS = "instrument_analysis"
STAGE_PORTFOLIO_ANALYTICS = "portfolio_analytics"
STAGE_DETERMINISTIC_REBALANCE = "deterministic_rebalance"
STAGE_PORTFOLIO_REVIEW = "portfolio_review"


def run_portfolio_workflow(
    request: PortfolioRequest,
    *,
    selected_analysts: Sequence[str],
    config: dict[str, Any],
    callbacks: Sequence[Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    graph: TradingAgentsGraph | None = None,
) -> PortfolioWorkflowState:
    """Run the existing four-stage portfolio workflow without UI side effects.

    ``graph`` is injectable for programmatic callers and tests. When omitted, the
    workflow constructs the same ``TradingAgentsGraph`` previously built by the
    CLI. Progress is exposed through stable, typed events rather than CLI text.
    """

    if graph is None:
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = TradingAgentsGraph(
            list(selected_analysts),
            config=config,
            debug=False,
            callbacks=list(callbacks or []),
        )
    workflow_graph = graph
    state: PortfolioWorkflowState = {
        "run_id": uuid4().hex,
        "portfolio_request": request,
        "trade_date": str(request.trade_date),
        "base_currency": request.base_currency,
        "warnings": [],
        "events": [],
    }

    def emit(name: str, stage: str, **payload: Any) -> None:
        event: PortfolioWorkflowEvent = {
            "name": name,
            "stage": stage,
            "payload": payload,
        }
        state["events"].append(event)
        if progress_callback is not None:
            progress_callback(event)

    emit("stage_started", STAGE_INSTRUMENT_ANALYSIS, positions=len(request.positions))

    def holding_progress(event: str, payload: dict[str, Any]) -> None:
        event_name = {
            "portfolio_started": "analysis_fanout_started",
            "portfolio_completed": "analysis_fanout_completed",
            "holding_started": "holding_started",
            "holding_completed": "holding_completed",
        }.get(event, event)
        emit(event_name, STAGE_INSTRUMENT_ANALYSIS, **payload)

    portfolio_result = workflow_graph.propagate_portfolio(
        request,
        progress_callback=holding_progress,
    )
    state.update(portfolio_result)
    emit(
        "stage_completed",
        STAGE_INSTRUMENT_ANALYSIS,
        holdings=len(portfolio_result.get("holdings", [])),
    )

    emit("stage_started", STAGE_PORTFOLIO_ANALYTICS)
    analytics_inputs = collect_portfolio_analytics_inputs(request, config=config)
    analytics = calculate_portfolio_analytics(
        request,
        historical_prices=analytics_inputs.historical_prices,
        benchmark_prices=analytics_inputs.benchmark_prices,
        sector_by_symbol=analytics_inputs.sector_by_symbol,
    )
    market_context = collect_portfolio_market_context(
        request,
        benchmark_symbol=analytics_inputs.benchmark_symbol,
        config=config,
    )
    state["portfolio_analytics_inputs"] = analytics_inputs
    state["portfolio_analytics"] = analytics
    state["portfolio_market_context"] = market_context
    regime_node = create_market_regime_agent(
        max_cash_shift=float(config.get("market_overlay_max_cash_shift", 0.05))
    )
    state.update(regime_node(state))
    for warning in analytics_inputs.warnings:
        state["warnings"].append(warning)
        emit("warning", STAGE_PORTFOLIO_ANALYTICS, source="data", message=warning)
    for warning in market_context.warnings:
        state["warnings"].append(warning)
        emit(
            "warning",
            STAGE_PORTFOLIO_ANALYTICS,
            source="market_context",
            message=warning,
        )
    emit("stage_completed", STAGE_PORTFOLIO_ANALYTICS, warnings=len(state["warnings"]))

    emit("stage_started", STAGE_DETERMINISTIC_REBALANCE)
    ratings = extract_ratings_from_portfolio_result(portfolio_result)
    instrument_proposals = extract_instrument_proposals(portfolio_result)
    holding_evidence = extract_holding_evidence_from_portfolio_result(portfolio_result)
    data_quality = assess_data_quality(
        request,
        analytics,
        holding_evidence,
        instrument_proposals,
        policy=DataQualityPolicy.from_config(config),
    )
    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol=ratings,
        instrument_proposals=instrument_proposals,
        data_quality=data_quality,
        market_regime=state["market_regime"],
        sector_by_symbol=analytics_inputs.sector_by_symbol,
        holdings=holding_evidence,
    )
    state["ratings_by_symbol"] = ratings
    state["instrument_proposals"] = instrument_proposals
    state["data_quality_assessment"] = data_quality
    state["rebalance_proposal"] = proposal
    emit("stage_completed", STAGE_DETERMINISTIC_REBALANCE)

    emit("stage_started", STAGE_PORTFOLIO_REVIEW)
    review_nodes = (
        (
            "risk_controller",
            create_portfolio_risk_analyst(workflow_graph.deep_thinking_llm),
        ),
        (
            "allocation_proposal_reviewer",
            create_portfolio_rebalancer(workflow_graph.deep_thinking_llm),
        ),
        (
            "portfolio_decision_approver",
            create_portfolio_allocation_manager(workflow_graph.deep_thinking_llm),
        ),
    )
    for agent_name, node in review_nodes:
        emit("agent_started", STAGE_PORTFOLIO_REVIEW, agent=agent_name)
        state.update(node(state))
        emit("agent_completed", STAGE_PORTFOLIO_REVIEW, agent=agent_name)
    emit("stage_completed", STAGE_PORTFOLIO_REVIEW)

    return state
