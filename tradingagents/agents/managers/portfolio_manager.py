"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import (
    PortfolioDecision,
    render_pm_decision,
)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.portfolio_prompting import format_portfolio_payload
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.portfolio.decision_protocol import (
    PortfolioApprovalDecision,
    ReviewDecision,
    build_final_portfolio_result,
    rejected_approval,
    render_final_portfolio_result,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state[
                "current_aggressive_response"
            ],
            "current_conservative_response": risk_debate_state[
                "current_conservative_response"
            ],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node


def create_portfolio_allocation_manager(llm):
    """Create a final approver that can only approve or reject one proposal."""

    structured_llm = bind_structured(
        llm, PortfolioApprovalDecision, "Portfolio Decision Approver"
    )

    def portfolio_allocation_manager_node(state) -> dict:
        proposal = state["rebalance_proposal"]
        request = state["portfolio_request"]
        prompt = f"""As the Portfolio Decision Approver, approve or reject the supplied deterministic proposal.

Return a structured decision referencing proposal_id `{proposal.proposal_id}` and proposal_version `{proposal.version}`. You must not generate target weights, component recommendations, or an executable trade list. Reject if the risk controller or proposal reviewer rejects the proposal, or if the evidence does not support approval.

**Deterministic Rebalance Proposal**
```json
{format_portfolio_payload(proposal)}
```
**Risk Validation**
```json
{format_portfolio_payload(state.get('risk_validation_result'))}
```
**Proposal Review**
```json
{format_portfolio_payload(state.get('proposal_review'))}
```
**Portfolio-Wide Market Context**
```json
{format_portfolio_payload(state.get('portfolio_market_context'))}
```
Use the deterministic analytics and proposal as the numeric source of truth. The summary must be a portfolio-level recommendation.{get_language_instruction()}"""
        try:
            approval = (
                structured_llm.invoke(prompt) if structured_llm is not None else None
            )
            if not isinstance(approval, PortfolioApprovalDecision):
                raise TypeError("approver did not return PortfolioApprovalDecision")
        except Exception as exc:
            approval = rejected_approval(proposal, str(exc))
        if (
            state.get("risk_validation_result") is not None
            and state["risk_validation_result"].decision == ReviewDecision.REJECT
        ) or (
            state.get("proposal_review") is not None
            and state["proposal_review"].decision == ReviewDecision.REJECT
        ):
            approval = PortfolioApprovalDecision(
                proposal_id=proposal.proposal_id,
                proposal_version=proposal.version,
                decision=ReviewDecision.REJECT,
                summary="Portfolio review protocol rejected the proposal.",
                rationale="At least one required review rejected or could not validate the proposal.",
            )
        final_result = build_final_portfolio_result(proposal, request, approval)
        return {
            "portfolio_approval_decision": approval,
            "final_portfolio_result": final_result,
            "final_portfolio_decision": render_final_portfolio_result(final_result),
        }

    return portfolio_allocation_manager_node
