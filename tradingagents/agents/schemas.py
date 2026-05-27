"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class ComponentAction(str, Enum):
    """Per-component portfolio allocation action."""

    BUY = "Buy"
    ADD = "Add"
    HOLD = "Hold"
    TRIM = "Trim"
    SELL = "Sell"


class PortfolioAllocationAction(str, Enum):
    """Whole-portfolio action for a multi-component allocation decision."""

    REBALANCE = "Rebalance"
    HOLD = "Hold"
    DE_RISK = "De-risk"
    INCREASE_RISK = "Increase Risk"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Allocation Decision
# ---------------------------------------------------------------------------


class ComponentRecommendation(BaseModel):
    """Structured recommendation for one portfolio component."""

    symbol: str = Field(
        description="Portfolio component symbol, e.g. AAPL, an option contract, or CASH.",
    )
    current_weight: float = Field(
        ge=0,
        le=1,
        description="Current portfolio allocation as a decimal, e.g. 0.25 for 25%.",
    )
    target_weight: float = Field(
        ge=0,
        le=1,
        description="Recommended target allocation as a decimal, e.g. 0.30 for 30%.",
    )
    weight_change: float = Field(
        ge=-1,
        le=1,
        description=(
            "Target minus current portfolio allocation as a decimal. Positive "
            "means add exposure; negative means reduce exposure."
        ),
    )
    action: ComponentAction = Field(
        description="Exactly one of Buy / Add / Hold / Trim / Sell.",
    )
    rationale: str = Field(
        description=(
            "Short rationale for the component action, grounded in the "
            "portfolio analytics and single-instrument analysis."
        ),
    )

    @model_validator(mode="after")
    def validate_weight_change(self) -> "ComponentRecommendation":
        expected_change = self.target_weight - self.current_weight
        if abs(self.weight_change - expected_change) > 0.0001:
            raise ValueError("weight_change must equal target_weight - current_weight")
        return self


class PortfolioAllocationDecision(BaseModel):
    """Structured output for a whole-portfolio allocation recommendation."""

    portfolio_action: PortfolioAllocationAction = Field(
        description="Exactly one of Rebalance / Hold / De-risk / Increase Risk.",
    )
    summary: str = Field(
        description=(
            "Concise portfolio-level recommendation summarizing the allocation "
            "change and why it is appropriate."
        ),
    )
    component_recommendations: list[ComponentRecommendation] = Field(
        min_length=1,
        description=(
            "One recommendation per portfolio component, including current "
            "weight, target weight, action, and rationale."
        ),
    )
    risk_notes: str = Field(
        description=(
            "Portfolio-level risk notes covering concentration, volatility, "
            "correlation, cash, option exposure, and constraint concerns."
        ),
    )


def render_portfolio_allocation_decision(
    decision: PortfolioAllocationDecision,
) -> str:
    """Render a portfolio allocation decision to Markdown."""

    parts = [
        f"**Portfolio Action**: {decision.portfolio_action.value}",
        "",
        f"**Summary**: {decision.summary}",
        "",
        "| Symbol | Current Weight | Target Weight | Change | Action | Rationale |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for recommendation in decision.component_recommendations:
        cells = [
            _escape_markdown_table_cell(recommendation.symbol),
            _format_weight(recommendation.current_weight),
            _format_weight(recommendation.target_weight),
            _format_signed_weight(recommendation.weight_change),
            recommendation.action.value,
            _escape_markdown_table_cell(recommendation.rationale),
        ]
        parts.append(f"| {' | '.join(cells)} |")
    parts.extend(["", f"**Risk Notes**: {decision.risk_notes}"])
    return "\n".join(parts)


def _format_weight(value: float) -> str:
    return f"{value:.2%}"


def _format_signed_weight(value: float) -> str:
    if abs(value) < 0.00005:
        value = 0.0
    return f"{value:+.2%}" if value > 0 else f"{value:.2%}"


def _escape_markdown_table_cell(value: str) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")
