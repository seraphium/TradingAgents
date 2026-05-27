"""Portfolio-level rebalancer review node."""

from __future__ import annotations

from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.agents.utils.portfolio_prompting import format_portfolio_payload


def create_portfolio_rebalancer(llm):
    """Create an LLM node that reviews deterministic target weights."""

    def portfolio_rebalancer_node(state) -> dict:
        analytics = state.get("portfolio_analytics")
        rebalance_proposal = state.get("rebalance_proposal")
        portfolio_risk_analysis = state.get("portfolio_risk_analysis", "")
        holdings = state.get("holdings", [])

        prompt = f"""As the Portfolio Rebalancer, review the deterministic target weights before final manager approval.

Use the supplied target weights and analytics as the numeric source of truth. Do not fabricate replacement weights or missing metrics. Explain whether each proposed add, trim, sell, buy, or hold action is consistent with the ratings, risk penalties, and constraints.

**Holding Analysis Summaries**
```json
{format_portfolio_payload(holdings)}
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

Return a concise Markdown rebalance review with portfolio-level tradeoffs and any objections the final manager should consider.{get_language_instruction()}"""

        response = llm.invoke(prompt)
        rebalance_review = response.content
        return {"portfolio_rebalance_review": rebalance_review}

    return portfolio_rebalancer_node
