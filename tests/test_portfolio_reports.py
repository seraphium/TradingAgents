from datetime import datetime
import json

import pytest

from tradingagents.portfolio import (
    AssetType,
    PortfolioInstrument,
    PortfolioMarketContext,
    PortfolioPosition,
    PortfolioRequest,
    calculate_portfolio_analytics,
    default_portfolio_report_dir,
    generate_rebalance_proposal,
    render_complete_portfolio_report,
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
        "rebalance_proposal": proposal,
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
def test_render_complete_portfolio_report_assembles_all_sections():
    markdown = render_complete_portfolio_report(portfolio_state())

    assert "# Portfolio Analysis Report" in markdown
    assert "## Holdings" in markdown
    assert "## Deterministic Portfolio Analytics" in markdown
    assert "## Portfolio-Wide Market Context" in markdown
    assert "## Deterministic Rebalance Proposal" in markdown
    assert "## Portfolio Risk Analysis" in markdown
    assert "## Portfolio Rebalance Review" in markdown
    assert "## Final Portfolio Decision" in markdown


@pytest.mark.unit
def test_save_portfolio_report_to_disk_writes_phase_10_shape(tmp_path):
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
    assert paths.rebalance_json and paths.rebalance_json.exists()
    assert paths.rebalance_markdown and paths.rebalance_markdown.exists()
    assert paths.risk_markdown and paths.risk_markdown.exists()
    assert paths.rebalance_review_markdown and paths.rebalance_review_markdown.exists()
    assert paths.final_decision_markdown and paths.final_decision_markdown.exists()

    analytics = json.loads(paths.analytics_json.read_text(encoding="utf-8"))
    market_context = json.loads(paths.market_context_json.read_text(encoding="utf-8"))
    assert analytics["weights_by_symbol"]["MSFT"] == pytest.approx(0.20)
    assert market_context["benchmark_symbol"] == "SPY"
    assert "Final Portfolio Decision" in paths.complete_report.read_text(encoding="utf-8")
