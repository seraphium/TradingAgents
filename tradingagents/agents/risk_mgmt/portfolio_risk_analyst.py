"""Portfolio-level risk analyst node."""

from __future__ import annotations

from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.agents.utils.portfolio_prompting import format_portfolio_payload


def create_portfolio_risk_analyst(llm):
    """Create an LLM node that critiques deterministic portfolio risk metrics."""

    def portfolio_risk_analyst_node(state) -> dict:
        analytics = state.get("portfolio_analytics")
        rebalance_proposal = state.get("rebalance_proposal")
        portfolio_request = state.get("portfolio_request")

        prompt = f"""As the Portfolio Risk Analyst, assess portfolio-level risk using only the supplied deterministic metrics.

Do not invent missing volatility, beta, correlation, liquidity, Greek, or concentration numbers. If a metric is absent, say it is unavailable and explain the implication.

**Portfolio Request**
```json
{format_portfolio_payload(portfolio_request)}
```

**Deterministic Portfolio Analytics**
```json
{format_portfolio_payload(analytics)}
```

**Deterministic Rebalance Proposal**
```json
{format_portfolio_payload(rebalance_proposal)}
```

Focus on allocation concentration, correlation, volatility, beta, cash, option exposure, aggregate Greeks, and constraint flags. End with practical risk controls for the portfolio-level manager.{get_language_instruction()}"""

        response = llm.invoke(prompt)
        risk_analysis = response.content
        return {"portfolio_risk_analysis": risk_analysis}

    return portfolio_risk_analyst_node
