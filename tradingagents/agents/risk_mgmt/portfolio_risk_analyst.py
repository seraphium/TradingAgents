"""Portfolio-level risk analyst node."""

from __future__ import annotations

from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.agents.utils.portfolio_prompting import (
    format_holding_evidence,
    format_portfolio_payload,
)


def create_portfolio_risk_analyst(llm):
    """Create an LLM node that critiques deterministic portfolio risk metrics."""

    def portfolio_risk_analyst_node(state) -> dict:
        analytics = state.get("portfolio_analytics")
        rebalance_proposal = state.get("rebalance_proposal")
        portfolio_request = state.get("portfolio_request")
        holdings = state.get("holdings", [])

        prompt = f"""As the Portfolio Risk Analyst, assess portfolio-level risk using only the supplied deterministic metrics.

Do not invent missing volatility, beta, correlation, liquidity, Greek, or concentration numbers. If a metric is absent, say it is unavailable and explain the implication.

Use the single-stock decisions and debate evidence to understand each holding's conviction and risk disagreements. Do not re-run the single-stock debate and do not write a separate rebalance reason for each stock. Aggregate the conclusions into portfolio-level risk controls such as reducing an over-concentrated sector, industry, or theme exposure; lowering total volatility; controlling correlation; preserving cash; and limiting option exposure.

**Portfolio Request**
```json
{format_portfolio_payload(portfolio_request)}
```

**Single-Stock Decision Evidence**
```json
{format_holding_evidence(holdings)}
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
