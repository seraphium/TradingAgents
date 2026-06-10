from dataclasses import replace

import pytest

from tradingagents.portfolio import (
    AssetType,
    PortfolioApprovalDecision,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
    ReviewDecision,
    build_final_portfolio_result,
    calculate_portfolio_analytics,
    generate_rebalance_proposal,
    validate_rebalance_proposal,
)


def _request():
    return PortfolioRequest(
        positions=[
            PortfolioPosition(
                instrument=PortfolioInstrument(
                    symbol="AAPL", asset_type=AssetType.STOCK
                ),
                current_weight=0.8,
                quantity=10,
            ),
            PortfolioPosition(
                instrument=PortfolioInstrument(
                    symbol="CASH", asset_type=AssetType.CASH
                ),
                current_weight=0.2,
                market_value=1000,
            ),
        ],
        trade_date="2026-06-10",
    )


@pytest.mark.unit
def test_proposal_has_stable_identity_version_and_input_hash():
    request = _request()
    analytics = calculate_portfolio_analytics(request)
    first = generate_rebalance_proposal(
        request, analytics, ratings_by_symbol={"AAPL": "Hold"}
    )
    second = generate_rebalance_proposal(
        request, analytics, ratings_by_symbol={"AAPL": "Hold"}
    )
    version_two = generate_rebalance_proposal(
        request, analytics, ratings_by_symbol={"AAPL": "Hold"}, proposal_version=2
    )

    assert first.input_hash == second.input_hash == version_two.input_hash
    assert first.proposal_id == second.proposal_id
    assert version_two.version == 2
    assert version_two.proposal_id != first.proposal_id


@pytest.mark.unit
def test_final_result_copies_only_valid_approved_proposal_weights():
    request = _request()
    proposal = generate_rebalance_proposal(
        request,
        calculate_portfolio_analytics(request),
        ratings_by_symbol={"AAPL": "Hold"},
    )
    approval = PortfolioApprovalDecision(
        proposal_id=proposal.proposal_id,
        proposal_version=proposal.version,
        decision=ReviewDecision.APPROVE,
        summary="Approve deterministic proposal.",
        rationale="All reviews passed.",
    )

    result = build_final_portfolio_result(proposal, request, approval)

    assert result.approved is True
    assert result.target_weights_by_symbol == proposal.target_weights_by_symbol


@pytest.mark.unit
def test_malformed_proposal_or_mismatched_approval_cannot_be_executable():
    request = _request()
    proposal = generate_rebalance_proposal(
        request,
        calculate_portfolio_analytics(request),
        ratings_by_symbol={"AAPL": "Hold"},
    )
    malformed = replace(proposal, target_weights_by_symbol={"AAPL": 1.2, "CASH": 0.2})
    approval = PortfolioApprovalDecision(
        proposal_id="invented-proposal",
        proposal_version=proposal.version,
        decision=ReviewDecision.APPROVE,
        summary="Approve.",
        rationale="Attempted approval.",
    )

    assert validate_rebalance_proposal(malformed, request)
    result = build_final_portfolio_result(malformed, request, approval)
    assert result.approved is False
    assert result.target_weights_by_symbol == {}
    assert result.actions_by_symbol == {}
