from types import SimpleNamespace

import pytest

from tradingagents.portfolio import (
    AssetType,
    InstrumentProposal,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
    ProposalSource,
    assess_data_quality,
    calculate_portfolio_analytics,
    extract_instrument_proposals,
    generate_rebalance_proposal,
    render_complete_portfolio_report,
)


def _position(symbol: str, weight: float, asset_type=AssetType.STOCK, **kwargs):
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol=symbol, asset_type=asset_type, **kwargs),
        current_weight=weight,
        quantity=None if asset_type == AssetType.CASH else 1,
    )


@pytest.mark.unit
def test_structured_instrument_proposal_is_preferred_over_markdown_rating():
    proposals = extract_instrument_proposals(
        {
            "holdings": [
                {
                    "symbol": "AAPL",
                    "asset_type": "stock",
                    "analysis_status": "analyzed",
                    "instrument_proposal": {
                        "symbol": "AAPL",
                        "rating": "Sell",
                        "conviction": -0.8,
                        "confidence": 0.75,
                        "horizon": "3 months",
                        "risk_flags": ["valuation"],
                    },
                    "analysis": {"final_trade_decision": "**Rating**: Buy"},
                }
            ]
        }
    )

    assert proposals["AAPL"].rating == "Sell"
    assert proposals["AAPL"].allocation_score == pytest.approx(-0.6)
    assert proposals["AAPL"].source == ProposalSource.STRUCTURED


@pytest.mark.unit
def test_failed_holding_cannot_silently_become_positive_allocation_signal():
    request = PortfolioRequest(
        trade_date="2026-06-09",
        positions=[_position("BAD", 0.5), _position("CASH", 0.5, AssetType.CASH)],
    )
    holdings = [
        {
            "symbol": "BAD",
            "asset_type": "stock",
            "analysis_status": "failed",
            "analysis": {"final_trade_decision": "**Rating**: Buy"},
        },
        {"symbol": "CASH", "asset_type": "cash", "analysis_status": "skipped"},
    ]
    proposals = extract_instrument_proposals({"holdings": holdings})
    analytics = calculate_portfolio_analytics(request)
    quality = assess_data_quality(request, analytics, holdings, proposals)
    rebalance = generate_rebalance_proposal(
        request,
        analytics,
        instrument_proposals=proposals,
        data_quality=quality,
        holdings=holdings,
    )

    assert proposals["BAD"].rating == "Hold"
    assert proposals["BAD"].confidence == 0
    assert quality.status == "insufficient"
    assert rebalance.target_weights_by_symbol["BAD"] <= 0.5


@pytest.mark.unit
def test_option_without_contract_evidence_is_restricted_from_increase():
    request = PortfolioRequest(
        trade_date="2026-06-09",
        positions=[
            _position(
                "AAPL260620C00200000",
                0.2,
                AssetType.OPTION,
                underlying="AAPL",
                expiry="2026-06-20",
                strike=200,
                right="C",
            ),
            _position("CASH", 0.8, AssetType.CASH),
        ],
    )
    holdings = [
        {
            "symbol": "AAPL260620C00200000",
            "asset_type": "option",
            "analysis_status": "underlying_analyzed",
            "contract_analysis": None,
            "analysis": {"final_trade_decision": "**Rating**: Buy"},
        },
        {"symbol": "CASH", "asset_type": "cash", "analysis_status": "skipped"},
    ]
    proposals = extract_instrument_proposals({"holdings": holdings})
    analytics = calculate_portfolio_analytics(request)
    quality = assess_data_quality(request, analytics, holdings, proposals)
    rebalance = generate_rebalance_proposal(
        request, analytics, instrument_proposals=proposals, data_quality=quality
    )

    assert quality.status == "restricted"
    assert "AAPL260620C00200000" in quality.restricted_symbols
    assert rebalance.target_weights_by_symbol["AAPL260620C00200000"] <= 0.2


@pytest.mark.unit
def test_complete_report_displays_weighted_analysis_and_metric_coverage():
    quality = SimpleNamespace(
        status="restricted",
        weighted_analysis_coverage=0.75,
        metric_coverage={"volatility": 0.5, "benchmark_beta": 1.0},
        restrictions=["BAD cannot increase."],
    )

    report = render_complete_portfolio_report(
        {
            "holdings": [],
            "data_quality_assessment": quality,
        }
    )

    assert "Weighted analysis coverage: 75.00%" in report
    assert "volatility 50.00%" in report
    assert "Restriction: BAD cannot increase." in report
