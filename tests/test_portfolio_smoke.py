import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.portfolio import (
    AssetType,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
    calculate_portfolio_analytics,
    extract_ratings_from_portfolio_result,
    generate_rebalance_proposal,
    load_portfolio_file,
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


def stubbed_graph() -> TradingAgentsGraph:
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)

    def propagate(symbol, trade_date, asset_type="stock"):
        rating = "Buy" if symbol == "AAPL" else "Underweight"
        return (
            {
                "company_of_interest": symbol,
                "trade_date": trade_date,
                "market_report": f"Market report for {symbol}.",
                "sentiment_report": "",
                "news_report": "",
                "fundamentals_report": "",
                "investment_debate_state": {},
                "risk_debate_state": {},
                "investment_plan": "",
                "trader_investment_plan": f"Trader plan for {symbol}.",
                "final_trade_decision": f"**Rating**: {rating}",
            },
            rating,
        )

    graph.propagate = propagate
    return graph


@pytest.mark.smoke
def test_minimal_two_stock_portfolio_smoke(tmp_path):
    request = PortfolioRequest(
        positions=[
            stock_position("AAPL", 0.45),
            stock_position("MSFT", 0.35),
            cash_position(0.20),
        ],
        trade_date="2026-05-28",
    )

    portfolio_result = stubbed_graph().propagate_portfolio(request)
    analytics = calculate_portfolio_analytics(
        request,
        historical_prices={
            "AAPL": [100, 101, 102, 103],
            "MSFT": [100, 99, 98, 97],
        },
    )
    proposal = generate_rebalance_proposal(
        request,
        analytics,
        ratings_by_symbol=extract_ratings_from_portfolio_result(portfolio_result),
        optimizer="mean_variance",
    )
    report_paths = save_portfolio_report_to_disk(
        {
            **portfolio_result,
            "portfolio_analytics": analytics,
            "rebalance_proposal": proposal,
            "final_portfolio_decision": "**Portfolio Action**: Rebalance",
        },
        tmp_path / "portfolio_smoke",
    )

    assert proposal.target_weights_by_symbol["AAPL"] > request.positions[0].current_weight
    assert proposal.target_weights_by_symbol["MSFT"] < request.positions[1].current_weight
    assert report_paths.complete_report.exists()
    assert (report_paths.holdings_dir / "AAPL.md").exists()
    assert (report_paths.portfolio_dir / "rebalance.md").exists()


@pytest.mark.unit
def test_example_portfolio_files_parse():
    csv_request = load_portfolio_file(
        "examples/portfolio_sample.csv",
        trade_date="2026-05-28",
    )
    json_request = load_portfolio_file("examples/portfolio_sample.json")

    assert len(csv_request.positions) == 4
    assert len(json_request.positions) == 4
    assert json_request.constraints.custom["rebalance_optimizer"] == "mean_variance"
