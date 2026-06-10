"""Portfolio-level allocation proposal reviewer node."""

from __future__ import annotations

from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.agents.utils.portfolio_prompting import (
    format_holding_evidence,
    format_portfolio_payload,
)
from tradingagents.agents.utils.structured import bind_structured
from tradingagents.portfolio.decision_protocol import (
    ProposalReview,
    ReviewDecision,
    render_proposal_review,
)


def create_portfolio_rebalancer(llm):
    """Create an LLM reviewer that challenges target weights without replacing them."""

    structured_llm = bind_structured(
        llm, ProposalReview, "Allocation Proposal Reviewer"
    )

    def portfolio_rebalancer_node(state) -> dict:
        proposal = state.get("rebalance_proposal")
        prompt = f"""As the Allocation Proposal Reviewer, review deterministic target weights before final approval.

The deterministic proposal is the numeric source of truth. Return a structured approve/reject review referencing proposal_id `{proposal.proposal_id}` and proposal_version `{proposal.version}`. You may challenge the proposal, but do not fabricate or return replacement weights or missing metrics.

**Single-Stock Decision Evidence**
```json
{format_holding_evidence(state.get('holdings', []))}
```
**Portfolio-Wide Market Context**
```json
{format_portfolio_payload(state.get('portfolio_market_context'))}
```
**Validated Market Regime and Allocation Overlay**
```json
{format_portfolio_payload(state.get('market_regime'))}
```
**Deterministic Portfolio Analytics**
```json
{format_portfolio_payload(state.get('portfolio_analytics'))}
```
**Deterministic Rebalance Proposal**
```json
{format_portfolio_payload(proposal)}
```
**Risk Validation**
```json
{format_portfolio_payload(state.get('risk_validation_result'))}
```
**Legacy Portfolio Risk Analyst Notes**
{state.get('portfolio_risk_analysis', '')}
Review the whole portfolio and identify any objections the final approver should consider.{get_language_instruction()}"""
        try:
            result = (
                structured_llm.invoke(prompt) if structured_llm is not None else None
            )
            if not isinstance(result, ProposalReview):
                raise TypeError("reviewer did not return ProposalReview")
            if (
                result.proposal_id != proposal.proposal_id
                or result.proposal_version != proposal.version
            ):
                raise ValueError("proposal review references a different proposal")
            markdown = render_proposal_review(result)
        except Exception as exc:
            response = llm.invoke(prompt)
            markdown = response.content
            result = ProposalReview(
                proposal_id=proposal.proposal_id,
                proposal_version=proposal.version,
                decision=ReviewDecision.REJECT,
                summary="Proposal review output could not be verified.",
                objections=[str(exc)],
            )
        return {"proposal_review": result, "portfolio_rebalance_review": markdown}

    return portfolio_rebalancer_node
