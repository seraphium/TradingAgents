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
    PortfolioAllocationDecision,
    PortfolioDecision,
    render_pm_decision,
    render_portfolio_allocation_decision,
)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.portfolio_prompting import (
    format_holding_evidence,
    format_portfolio_payload,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
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
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node


def create_portfolio_allocation_manager(llm):
    """Create a portfolio-level final decision node.

    This is separate from ``create_portfolio_manager`` so the existing
    single-instrument portfolio manager contract remains unchanged.
    """

    structured_llm = bind_structured(
        llm,
        PortfolioAllocationDecision,
        "Portfolio Allocation Manager",
    )

    def portfolio_allocation_manager_node(state) -> dict:
        analytics = state.get("portfolio_analytics")
        market_context = state.get("portfolio_market_context")
        rebalance_proposal = state.get("rebalance_proposal")
        portfolio_risk_analysis = state.get("portfolio_risk_analysis", "")
        portfolio_rebalance_review = state.get("portfolio_rebalance_review", "")
        holdings = state.get("holdings", [])

        prompt = f"""As the Portfolio Allocation Manager, deliver the final whole-portfolio decision.

Use the deterministic analytics and rebalance proposal as the numeric source of truth. You may critique the proposal, but do not invent new current weights, target weights, volatility, beta, correlation, Greek, liquidity, or concentration values. The final output must include one component recommendation for every supplied portfolio component.

Use the single-stock Portfolio Manager decisions and the aggressive/conservative/neutral debate evidence as the qualitative source of truth for each holding. Do not repeat one rebalance reason per stock. Instead, form one whole-portfolio thesis that reconciles all holding-level conclusions with sector, industry, or theme exposure; concentration; correlation; total volatility; cash; options exposure; and target-weight constraints.

**Single-Stock Decision Evidence**
```json
{format_holding_evidence(holdings)}
```

**Portfolio-Wide Market Context**
```json
{format_portfolio_payload(market_context)}
```

**Deterministic Portfolio Analytics**
```json
{format_portfolio_payload(analytics)}
```

**Deterministic Rebalance Proposal**
```json
{format_portfolio_payload(rebalance_proposal)}
```

**Portfolio Risk Analyst Notes**
{portfolio_risk_analysis}

**Portfolio Rebalancer Review**
{portfolio_rebalance_review}

Choose a portfolio action from Rebalance, Hold, De-risk, or Increase Risk. The summary must be a portfolio-level recommendation, for example reducing an over-concentrated industry/theme exposure, responding to broad market/news/sentiment/fundamental conditions, or lowering total volatility while preserving the strongest single-stock conclusions. For each component, provide current weight, target weight, weight change, action, a component summary based on the single-stock final conclusion, and a concise execution rationale.{get_language_instruction()}"""

        final_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_portfolio_allocation_decision,
            "Portfolio Allocation Manager",
        )

        return {"final_portfolio_decision": final_decision}

    return portfolio_allocation_manager_node
