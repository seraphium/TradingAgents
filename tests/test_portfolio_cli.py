from inspect import signature

import pytest
import typer

from cli.main import (
    analyze,
    load_portfolio_request_for_cli,
    parse_cli_float,
    resolve_analysis_mode,
)
from tradingagents.portfolio import save_portfolio_report_to_disk
from tradingagents.graph.trading_graph import TradingAgentsGraph
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


@pytest.mark.unit
def test_resolve_analysis_mode_accepts_portfolio_file():
    assert resolve_analysis_mode(portfolio_file="portfolio.csv") == "portfolio"
    assert resolve_analysis_mode("single") == "single"
    assert resolve_analysis_mode("portfolio") == "portfolio"

    with pytest.raises(typer.BadParameter):
        resolve_analysis_mode("single", portfolio_file="portfolio.csv")


@pytest.mark.unit
def test_parse_cli_float_accepts_decimal_and_percent_weights():
    assert parse_cli_float("0.25", weight=True) == pytest.approx(0.25)
    assert parse_cli_float("25%", weight=True) == pytest.approx(0.25)
    assert parse_cli_float("25", weight=True) == pytest.approx(0.25)
    assert parse_cli_float("25", weight=False) == pytest.approx(25.0)


@pytest.mark.unit
def test_load_portfolio_request_for_cli_uses_portfolio_file(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text(
        "symbol,asset_type,current_weight,quantity\n"
        "AAPL,stock,0.80,10\n"
        "CASH,cash,0.20,\n",
        encoding="utf-8",
    )

    request = load_portfolio_request_for_cli(path, trade_date="2026-05-27")

    assert request.trade_date.isoformat() == "2026-05-27"
    assert [position.instrument.symbol for position in request.positions] == ["AAPL", "CASH"]


@pytest.mark.unit
def test_analyze_exposes_portfolio_file_option():
    params = signature(analyze).parameters

    assert "portfolio_file" in params
    assert "mode" in params
    assert "checkpoint" in params


@pytest.mark.unit
def test_save_portfolio_report_to_disk_writes_expected_shape(tmp_path):
    request = PortfolioRequest(
        positions=[stock_position("AAPL", 0.80), cash_position(0.20)],
        trade_date="2026-05-27",
    )
    analytics = calculate_portfolio_analytics(request)
    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol={"AAPL": "Overweight"},
    )
    state = {
        "portfolio_request": request,
        "trade_date": "2026-05-27",
        "holdings": [
            {
                "symbol": "AAPL",
                "asset_type": "stock",
                "current_weight": 0.80,
                "analysis_status": "analyzed",
                "analysis": {
                    "signal": "Overweight",
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
        "rebalance_proposal": proposal,
        "portfolio_risk_analysis": "Risk notes.",
        "portfolio_rebalance_review": "Rebalance review.",
        "final_portfolio_decision": "**Portfolio Action**: Rebalance",
    }

    report_paths = save_portfolio_report_to_disk(state, tmp_path / "portfolio_report")

    assert report_paths.complete_report.name == "complete_report.md"
    assert (tmp_path / "portfolio_report" / "holdings" / "AAPL.md").exists()
    assert (tmp_path / "portfolio_report" / "portfolio" / "analytics.json").exists()
    assert (tmp_path / "portfolio_report" / "portfolio" / "rebalance.md").exists()
    assert (tmp_path / "portfolio_report" / "portfolio" / "risk.md").exists()
    assert (tmp_path / "portfolio_report" / "portfolio" / "final_decision.md").exists()
    assert "Portfolio Analysis Report" in report_paths.complete_report.read_text(encoding="utf-8")


@pytest.mark.unit
def test_propagate_portfolio_emits_progress_events():
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)

    def propagate(symbol, trade_date, asset_type="stock"):
        return (
            {
                "market_report": "",
                "sentiment_report": "",
                "news_report": "",
                "fundamentals_report": "",
                "investment_debate_state": {},
                "risk_debate_state": {},
                "investment_plan": "",
                "trader_investment_plan": "",
                "final_trade_decision": "**Rating**: Hold",
            },
            "Hold",
        )

    graph.propagate = propagate
    request = PortfolioRequest(
        positions=[stock_position("AAPL", 0.80), cash_position(0.20)],
        trade_date="2026-05-27",
    )
    events = []

    graph.propagate_portfolio(
        request,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    event_names = [event for event, _ in events]
    assert event_names[0] == "portfolio_started"
    assert event_names.count("holding_started") == 2
    assert event_names.count("holding_completed") == 2
    assert event_names[-1] == "portfolio_completed"
