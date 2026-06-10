"""Portfolio-level risk controller node."""

from __future__ import annotations

from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.agents.utils.portfolio_prompting import (
    format_holding_evidence,
    format_portfolio_payload,
)
from tradingagents.agents.utils.structured import bind_structured
from tradingagents.portfolio.decision_protocol import (
    RiskValidationResult,
    ReviewDecision,
    render_risk_validation,
)


def create_portfolio_risk_analyst(llm):
    """Create an LLM node that validates a deterministic proposal without rewriting it."""

    structured_llm = bind_structured(
        llm, RiskValidationResult, "Portfolio Risk Controller"
    )

    def portfolio_risk_analyst_node(state) -> dict:
        proposal = state.get("rebalance_proposal")
        prompt = f"""As the Portfolio Risk Controller, validate the deterministic proposal using only the supplied deterministic metrics and evidence.

Return a structured approve/reject result referencing proposal_id `{proposal.proposal_id}` and proposal_version `{proposal.version}`. Do not invent or return replacement weights. Reject when a material risk or constraint violation makes this proposal unsafe.

**Portfolio Request**
```json
{format_portfolio_payload(state.get('portfolio_request'))}
```
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
Use only supplied deterministic metrics. Do not invent missing volatility, beta, correlation, liquidity, Greek, or concentration numbers.{get_language_instruction()}"""
        try:
            result = (
                structured_llm.invoke(prompt) if structured_llm is not None else None
            )
            if not isinstance(result, RiskValidationResult):
                raise TypeError("risk controller did not return RiskValidationResult")
            if (
                result.proposal_id != proposal.proposal_id
                or result.proposal_version != proposal.version
            ):
                raise ValueError("risk validation references a different proposal")
            markdown = render_risk_validation(result)
        except Exception as exc:
            response = llm.invoke(prompt)
            markdown = response.content
            result = RiskValidationResult(
                proposal_id=proposal.proposal_id,
                proposal_version=proposal.version,
                decision=ReviewDecision.REJECT,
                summary="Risk validation output could not be verified.",
                violations=[str(exc)],
            )
        return {"risk_validation_result": result, "portfolio_risk_analysis": markdown}

    return portfolio_risk_analyst_node
