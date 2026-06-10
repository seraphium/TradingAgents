"""Typed review and approval protocol for deterministic portfolio proposals.

Portfolio-level agents may approve, reject, or challenge a proposal, but they
never author executable weights.  All executable weights come from a validated
:class:`~tradingagents.portfolio.rebalancing.RebalanceProposal`.
"""

from __future__ import annotations

from enum import Enum
from math import isclose
from pydantic import BaseModel, ConfigDict, Field

from tradingagents.portfolio.rebalancing import RebalanceProposal
from tradingagents.portfolio.schemas import AssetType, PortfolioRequest


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


class RiskValidationResult(BaseModel):
    """Risk controller's structured validation of one proposal version."""

    model_config = ConfigDict(frozen=True)

    proposal_id: str
    proposal_version: int = Field(ge=1)
    decision: ReviewDecision
    summary: str
    violations: list[str] = Field(default_factory=list)
    risk_controls: list[str] = Field(default_factory=list)


class ProposalReview(BaseModel):
    """Allocation review that can challenge, but not rewrite, a proposal."""

    model_config = ConfigDict(frozen=True)

    proposal_id: str
    proposal_version: int = Field(ge=1)
    decision: ReviewDecision
    summary: str
    objections: list[str] = Field(default_factory=list)


class PortfolioApprovalDecision(BaseModel):
    """Final approval decision referencing an existing deterministic proposal."""

    model_config = ConfigDict(frozen=True)

    proposal_id: str
    proposal_version: int = Field(ge=1)
    decision: ReviewDecision
    summary: str
    rationale: str


class FinalPortfolioResult(BaseModel):
    """Validated executable result; weights are copied from the proposal only."""

    model_config = ConfigDict(frozen=True)

    proposal_id: str
    proposal_version: int = Field(ge=1)
    approved: bool
    portfolio_action: str
    target_weights_by_symbol: dict[str, float]
    actions_by_symbol: dict[str, str]
    summary: str
    rationale: str


def validate_rebalance_proposal(
    proposal: RebalanceProposal,
    request: PortfolioRequest,
    *,
    tolerance: float = 1e-6,
) -> list[str]:
    """Return every executable invariant violation in ``proposal``."""

    errors: list[str] = []
    request_symbols = [position.instrument.symbol for position in request.positions]
    request_symbol_set = set(request_symbols)
    target_symbols = set(proposal.target_weights_by_symbol)
    component_symbols = [component.symbol for component in proposal.component_proposals]

    if len(component_symbols) != len(set(component_symbols)):
        errors.append("component symbols must be unique")
    if target_symbols != request_symbol_set:
        errors.append("target-weight symbols must exactly match portfolio symbols")
    if set(component_symbols) != request_symbol_set:
        errors.append("component symbols must exactly match portfolio symbols")

    weights = proposal.target_weights_by_symbol
    if any(
        not isinstance(weight, (int, float)) or weight < 0 or weight > 1
        for weight in weights.values()
    ):
        errors.append("every target weight must be between 0 and 1")
    if not isclose(sum(weights.values()), 1.0, abs_tol=tolerance):
        errors.append("target weights must sum to 1")

    positions = {position.instrument.symbol: position for position in request.positions}
    valid_actions = {"Buy", "Add", "Hold", "Trim", "Sell"}
    for component in proposal.component_proposals:
        position = positions.get(component.symbol)
        target = weights.get(component.symbol)
        if position is None or target is None:
            continue
        if not isclose(
            component.current_weight, float(position.current_weight), abs_tol=tolerance
        ):
            errors.append(f"{component.symbol}: current weight does not match request")
        if not isclose(component.target_weight, target, abs_tol=tolerance):
            errors.append(
                f"{component.symbol}: component target does not match target map"
            )
        if not isclose(
            component.weight_change,
            target - component.current_weight,
            abs_tol=tolerance,
        ):
            errors.append(f"{component.symbol}: weight change is inconsistent")
        action_tolerance = float(
            proposal.diagnostics.get("change_tolerance", tolerance)
        )
        expected_action = _expected_action(
            component.current_weight, target, action_tolerance
        )
        if component.action not in valid_actions or component.action != expected_action:
            errors.append(
                f"{component.symbol}: action is inconsistent with weight change"
            )
        if (
            position.target_min_weight is not None
            and target < float(position.target_min_weight) - tolerance
        ):
            errors.append(f"{component.symbol}: target is below target_min_weight")
        if (
            position.target_max_weight is not None
            and target > float(position.target_max_weight) + tolerance
        ):
            errors.append(f"{component.symbol}: target exceeds target_max_weight")

    constraints = request.constraints
    if (
        constraints.min_cash_weight is not None
        and weights.get("CASH", 0.0) < float(constraints.min_cash_weight) - tolerance
    ):
        errors.append("cash target is below min_cash_weight")
    if (
        constraints.max_cash_weight is not None
        and weights.get("CASH", 0.0) > float(constraints.max_cash_weight) + tolerance
    ):
        errors.append("cash target exceeds max_cash_weight")
    option_weight = sum(
        weights.get(position.instrument.symbol, 0.0)
        for position in request.positions
        if position.instrument.asset_type == AssetType.OPTION
    )
    if (
        constraints.max_options_weight is not None
        and option_weight > float(constraints.max_options_weight) + tolerance
    ):
        errors.append("option targets exceed max_options_weight")
    max_position = constraints.max_single_position_weight
    if max_position is not None and any(
        weight > float(max_position) + tolerance for weight in weights.values()
    ):
        errors.append("a target exceeds max_single_position_weight")
    return errors


def build_final_portfolio_result(
    proposal: RebalanceProposal,
    request: PortfolioRequest,
    approval: PortfolioApprovalDecision,
) -> FinalPortfolioResult:
    """Validate approval and build the only executable portfolio result shape."""

    errors = validate_rebalance_proposal(proposal, request)
    if (
        approval.proposal_id != proposal.proposal_id
        or approval.proposal_version != proposal.version
    ):
        errors.append("approval must reference the current deterministic proposal")
    approved = approval.decision == ReviewDecision.APPROVE and not errors
    summary = (
        approval.summary if not errors else f"Approval rejected: {'; '.join(errors)}"
    )
    return FinalPortfolioResult(
        proposal_id=proposal.proposal_id,
        proposal_version=proposal.version,
        approved=approved,
        portfolio_action=proposal.portfolio_action if approved else "Hold",
        target_weights_by_symbol=(
            dict(proposal.target_weights_by_symbol) if approved else {}
        ),
        actions_by_symbol=(
            {
                component.symbol: component.action
                for component in proposal.component_proposals
            }
            if approved
            else {}
        ),
        summary=summary,
        rationale=approval.rationale,
    )


def render_risk_validation(result: RiskValidationResult) -> str:
    return _render_review(
        "Risk Validation",
        result.decision,
        result.summary,
        result.violations,
        result.risk_controls,
    )


def render_proposal_review(result: ProposalReview) -> str:
    return _render_review(
        "Proposal Review", result.decision, result.summary, result.objections, []
    )


def render_final_portfolio_result(result: FinalPortfolioResult) -> str:
    parts = [
        f"**Decision**: {'Approved' if result.approved else 'Rejected'}",
        f"**Proposal**: `{result.proposal_id}` version {result.proposal_version}",
        f"**Portfolio Action**: {result.portfolio_action}",
        "",
        f"**Summary**: {result.summary}",
        f"**Rationale**: {result.rationale}",
    ]
    if result.approved:
        parts.extend(["", "| Symbol | Target Weight | Action |", "|---|---:|---|"])
        for symbol, weight in result.target_weights_by_symbol.items():
            parts.append(
                f"| {symbol} | {weight:.2%} | {result.actions_by_symbol[symbol]} |"
            )
    else:
        parts.extend(["", "No executable trade list was produced."])
    return "\n".join(parts)


def rejected_approval(
    proposal: RebalanceProposal, reason: str
) -> PortfolioApprovalDecision:
    """Create a safe rejection when structured approval output is unavailable."""

    return PortfolioApprovalDecision(
        proposal_id=proposal.proposal_id,
        proposal_version=proposal.version,
        decision=ReviewDecision.REJECT,
        summary="Portfolio approval could not be validated.",
        rationale=reason,
    )


def _expected_action(current: float, target: float, tolerance: float) -> str:
    change = target - current
    if abs(change) <= tolerance:
        return "Hold"
    if change > 0:
        return "Buy" if current <= tolerance else "Add"
    return "Sell" if target <= tolerance else "Trim"


def _render_review(
    title: str,
    decision: ReviewDecision,
    summary: str,
    issues: list[str],
    controls: list[str],
) -> str:
    parts = [f"**{title} Decision**: {decision.value.title()}", "", summary]
    if issues:
        parts.extend(["", "**Issues**", *[f"- {item}" for item in issues]])
    if controls:
        parts.extend(["", "**Risk Controls**", *[f"- {item}" for item in controls]])
    return "\n".join(parts)
