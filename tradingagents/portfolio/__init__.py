"""Portfolio analysis models and utilities."""

from tradingagents.portfolio.schemas import (
    AssetType,
    OptionRight,
    PortfolioConstraints,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
)
from tradingagents.portfolio.io import (
    PortfolioParseError,
    load_portfolio_file,
    normalize_portfolio_symbol,
    parse_portfolio_csv,
    parse_portfolio_json,
    parse_portfolio_json_payload,
)
from tradingagents.portfolio.analytics import (
    ConstraintFlag,
    OptionGreeks,
    PortfolioAnalytics,
    calculate_portfolio_analytics,
    portfolio_analytics_to_dict,
)
from tradingagents.portfolio.data_quality import (
    DataQualityAssessment,
    DataQualityPolicy,
    assess_data_quality,
)
from tradingagents.portfolio.instrument_proposals import (
    InstrumentProposal,
    ProposalSource,
    extract_instrument_proposals,
)
from tradingagents.portfolio.decision_protocol import (
    FinalPortfolioResult,
    PortfolioApprovalDecision,
    ProposalReview,
    ReviewDecision,
    RiskValidationResult,
    build_final_portfolio_result,
    render_final_portfolio_result,
    validate_rebalance_proposal,
)
from tradingagents.portfolio.data_inputs import (
    PortfolioAnalyticsInputs,
    collect_portfolio_analytics_inputs,
)
from tradingagents.portfolio.market_regime import (
    BenchmarkTrendFeatures,
    MarketOverlay,
    MarketRegime,
    MarketRegimeLabel,
    create_market_regime_agent,
    derive_benchmark_trend_features,
    derive_market_regime,
)
from tradingagents.portfolio.market_context import (
    PortfolioMarketContext,
    collect_portfolio_market_context,
)
from tradingagents.portfolio.rebalancing import (
    RATING_SCORES,
    RebalanceComponent,
    RebalanceProposal,
    extract_holding_evidence_from_portfolio_result,
    extract_portfolio_allocation_summary,
    extract_ratings_from_portfolio_result,
    generate_rebalance_proposal,
    rebalance_proposal_to_dict,
    render_rebalance_proposal,
)
from tradingagents.portfolio.reports import (
    PortfolioReportPaths,
    default_portfolio_report_dir,
    render_complete_portfolio_report,
    render_holding_report,
    save_portfolio_report_to_disk,
)

__all__ = [
    "AssetType",
    "OptionRight",
    "PortfolioParseError",
    "PortfolioConstraints",
    "PortfolioInstrument",
    "PortfolioPosition",
    "PortfolioRequest",
    "ConstraintFlag",
    "DataQualityAssessment",
    "DataQualityPolicy",
    "FinalPortfolioResult",
    "PortfolioApprovalDecision",
    "ProposalReview",
    "ReviewDecision",
    "RiskValidationResult",
    "InstrumentProposal",
    "ProposalSource",
    "OptionGreeks",
    "PortfolioAnalytics",
    "PortfolioAnalyticsInputs",
    "PortfolioMarketContext",
    "BenchmarkTrendFeatures",
    "MarketOverlay",
    "MarketRegime",
    "MarketRegimeLabel",
    "RATING_SCORES",
    "RebalanceComponent",
    "RebalanceProposal",
    "PortfolioReportPaths",
    "PortfolioWorkflowEvent",
    "PortfolioWorkflowState",
    "assess_data_quality",
    "build_final_portfolio_result",
    "calculate_portfolio_analytics",
    "collect_portfolio_analytics_inputs",
    "collect_portfolio_market_context",
    "create_market_regime_agent",
    "derive_benchmark_trend_features",
    "derive_market_regime",
    "default_portfolio_report_dir",
    "extract_holding_evidence_from_portfolio_result",
    "extract_instrument_proposals",
    "extract_portfolio_allocation_summary",
    "extract_ratings_from_portfolio_result",
    "generate_rebalance_proposal",
    "load_portfolio_file",
    "normalize_portfolio_symbol",
    "parse_portfolio_csv",
    "parse_portfolio_json",
    "parse_portfolio_json_payload",
    "portfolio_analytics_to_dict",
    "rebalance_proposal_to_dict",
    "render_complete_portfolio_report",
    "render_final_portfolio_result",
    "render_holding_report",
    "render_rebalance_proposal",
    "run_portfolio_workflow",
    "validate_rebalance_proposal",
    "save_portfolio_report_to_disk",
]


def __getattr__(name: str):
    """Lazily expose workflow objects to avoid agent/portfolio import cycles."""

    if name in {
        "PortfolioWorkflowEvent",
        "PortfolioWorkflowState",
        "run_portfolio_workflow",
    }:
        from tradingagents.portfolio import workflow

        return getattr(workflow, name)
    raise AttributeError(name)
