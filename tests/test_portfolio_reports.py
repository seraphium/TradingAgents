from datetime import datetime
import json

import pytest

from tradingagents.portfolio import (
    AssetType,
    DataQualityAssessment,
    InstrumentProposal,
    PortfolioInstrument,
    PortfolioMarketContext,
    derive_market_regime,
    PortfolioPosition,
    PortfolioRequest,
    calculate_portfolio_analytics,
    default_portfolio_report_dir,
    generate_rebalance_proposal,
    build_final_result,
    render_complete_portfolio_report,
    render_portfolio_decision_summary,
    render_holding_report,
    save_portfolio_report_to_disk,
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


def portfolio_state() -> dict:
    request = PortfolioRequest(
        positions=[
            stock_position("BRK.B", 0.60),
            stock_position("MSFT", 0.20),
            cash_position(0.20),
        ],
        trade_date="2026-05-27",
    )
    analytics = calculate_portfolio_analytics(request)
    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol={"BRK.B": "Hold", "MSFT": "Overweight"},
    )
    return {
        "portfolio_request": request,
        "trade_date": "2026-05-27",
        "base_currency": "USD",
        "holdings": [
            {
                "symbol": "BRK.B",
                "asset_type": "stock",
                "current_weight": 0.60,
                "quantity": 10,
                "analysis_status": "analyzed",
                "analysis": {
                    "signal": "Hold",
                    "reports": {"market": "Market report."},
                    "trader_investment_plan": "Trader plan.",
                    "final_trade_decision": "**Rating**: Hold",
                },
            },
            {
                "symbol": "MSFT",
                "asset_type": "stock",
                "current_weight": 0.20,
                "analysis_status": "analyzed",
                "analysis": {
                    "signal": "Overweight",
                    "reports": {},
                    "final_trade_decision": "**Rating**: Overweight",
                },
            },
            {
                "symbol": "CASH",
                "asset_type": "cash",
                "current_weight": 0.20,
                "analysis_status": "skipped",
                "skip_reason": "cash positions do not require analysis",
                "analysis": None,
            },
        ],
        "portfolio_analytics": analytics,
        "portfolio_market_context": PortfolioMarketContext(
            benchmark_symbol="SPY",
            global_news="Broad market news.",
        ),
        "market_regime": derive_market_regime(
            PortfolioMarketContext(benchmark_symbol="SPY"),
            benchmark_prices=[100 + index for index in range(220)],
        ),
        "instrument_proposals": {
            "BRK.B": InstrumentProposal(
                symbol="BRK.B",
                rating="Hold",
                conviction=0,
                confidence=0.8,
                summary="Stable initial view.",
            ),
            "MSFT": InstrumentProposal(
                symbol="MSFT",
                rating="Overweight",
                conviction=0.5,
                confidence=0.9,
                summary="Positive initial view.",
            ),
        },
        "data_quality_assessment": DataQualityAssessment(
            status="restricted",
            weighted_analysis_coverage=1.0,
            metric_coverage={"volatility": 0.8},
            warnings=["Volatility coverage is incomplete."],
        ),
        "rebalance_proposal": proposal,
        "proposal_history": [
            {
                "version": 1,
                "reason": "Initial deterministic proposal",
                "proposal": proposal,
            }
        ],
        "portfolio_risk_analysis": "Risk analysis.",
        "portfolio_rebalance_review": "Rebalance review.",
        "final_portfolio_decision": "**Portfolio Action**: Rebalance",
    }


@pytest.mark.unit
def test_default_portfolio_report_dir_uses_required_prefix(tmp_path):
    path = default_portfolio_report_dir(
        tmp_path,
        timestamp=datetime(2026, 5, 27, 12, 34, 56),
    )

    assert path == tmp_path / "portfolio_20260527_123456"


@pytest.mark.unit
def test_render_holding_report_includes_single_holding_sections():
    holding = portfolio_state()["holdings"][0]

    markdown = render_holding_report(holding)

    assert markdown.startswith("# BRK.B")
    assert "## Analyst Reports" in markdown
    assert "### Market" in markdown
    assert "## Trader Plan" in markdown
    assert "## Final Decision" in markdown


@pytest.mark.unit
def test_render_complete_portfolio_report_is_decision_first_and_moves_narratives_to_appendix():
    markdown = render_complete_portfolio_report(portfolio_state())

    assert markdown.startswith("# Portfolio Decision Report")
    assert markdown.index("## Executive Decision") < markdown.index(
        "## Final Rebalance Table"
    )
    assert markdown.index("## Final Rebalance Table") < markdown.index(
        "## Portfolio Risk and Exposure Summary"
    )
    assert "| MSFT | Overweight | 20.00% |" in markdown
    assert "## Instrument Analysis Summaries" in markdown
    assert "## Rebalance Rationale and Proposal History" in markdown
    assert markdown.index("## Appendix") < markdown.index("Risk analysis.")


@pytest.mark.unit
def test_render_portfolio_decision_summary_excludes_verbose_agent_narratives():
    markdown = render_portfolio_decision_summary(portfolio_state())

    assert "## Executive Decision" in markdown
    assert "## Final Rebalance Table" in markdown
    assert "Risk analysis." not in markdown
    assert "Rebalance review." not in markdown


@pytest.mark.unit
def test_build_final_result_contains_exact_allocations_and_history():
    result = build_final_result(portfolio_state())

    assert result["portfolio_action"] == "Increase Risk"
    assert result["approval_status"] == "recorded"
    assert result["final_allocations"][1]["symbol"] == "MSFT"
    assert result["final_allocations"][1]["initial_rating"] == "Overweight"
    assert result["proposal_history"][0]["version"] == 1


@pytest.mark.unit
def test_save_portfolio_report_to_disk_writes_decision_first_shape(tmp_path):
    paths = save_portfolio_report_to_disk(
        portfolio_state(),
        tmp_path / "portfolio_20260527_123456",
    )

    assert paths.root_dir.name == "portfolio_20260527_123456"
    assert paths.complete_report.exists()
    assert (paths.holdings_dir / "BRK.B.md").exists()
    assert (paths.holdings_dir / "MSFT.md").exists()
    assert (paths.holdings_dir / "CASH.md").exists()
    assert paths.analytics_json and paths.analytics_json.exists()
    assert paths.market_context_json and paths.market_context_json.exists()
    assert paths.market_regime_json and paths.market_regime_json.exists()
    assert paths.rebalance_json and paths.rebalance_json.exists()
    assert paths.rebalance_markdown and paths.rebalance_markdown.exists()
    assert paths.risk_markdown and paths.risk_markdown.exists()
    assert paths.rebalance_review_markdown and paths.rebalance_review_markdown.exists()
    assert paths.final_decision_markdown and paths.final_decision_markdown.exists()
    assert paths.final_result_json and paths.final_result_json.exists()
    assert paths.proposals_dir and (paths.proposals_dir / "proposal_v1.json").exists()
    assert paths.risk_validation_json and paths.risk_validation_json.exists()
    assert paths.proposal_review_json and paths.proposal_review_json.exists()
    assert paths.approval_json and paths.approval_json.exists()

    final_result = json.loads(paths.final_result_json.read_text(encoding="utf-8"))
    assert final_result["portfolio_action"] == "Increase Risk"
    assert final_result["final_allocations"][0]["symbol"] == "BRK.B"

    analytics = json.loads(paths.analytics_json.read_text(encoding="utf-8"))
    market_context = json.loads(paths.market_context_json.read_text(encoding="utf-8"))
    assert analytics["weights_by_symbol"]["MSFT"] == pytest.approx(0.20)
    assert market_context["benchmark_symbol"] == "SPY"
    assert "Executive Decision" in paths.complete_report.read_text(encoding="utf-8")
