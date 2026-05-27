"""Portfolio report rendering and filesystem output."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import json
from pathlib import Path
from typing import Any

from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.portfolio.analytics import portfolio_analytics_to_dict
from tradingagents.portfolio.rebalancing import (
    rebalance_proposal_to_dict,
    render_rebalance_proposal,
)


@dataclass(frozen=True)
class PortfolioReportPaths:
    """Paths written for a portfolio report."""

    root_dir: Path
    complete_report: Path
    holdings_dir: Path
    portfolio_dir: Path
    analytics_json: Path | None = None
    rebalance_json: Path | None = None
    rebalance_markdown: Path | None = None
    risk_markdown: Path | None = None
    rebalance_review_markdown: Path | None = None
    final_decision_markdown: Path | None = None


def default_portfolio_report_dir(
    base_dir: Path,
    *,
    timestamp: datetime.datetime | None = None,
) -> Path:
    """Return the default portfolio report directory path."""

    stamp = (timestamp or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    return base_dir / f"portfolio_{stamp}"


def save_portfolio_report_to_disk(
    portfolio_state: dict[str, Any],
    save_path: Path,
) -> PortfolioReportPaths:
    """Save a complete portfolio analysis report directory."""

    save_path.mkdir(parents=True, exist_ok=True)
    holdings_dir = save_path / "holdings"
    portfolio_dir = save_path / "portfolio"
    holdings_dir.mkdir(exist_ok=True)
    portfolio_dir.mkdir(exist_ok=True)

    for holding in portfolio_state.get("holdings", []):
        symbol = safe_ticker_component(holding.get("symbol", "UNKNOWN"))
        (holdings_dir / f"{symbol}.md").write_text(
            render_holding_report(holding),
            encoding="utf-8",
        )

    analytics_json = None
    analytics = portfolio_state.get("portfolio_analytics")
    if analytics is not None:
        analytics_json = portfolio_dir / "analytics.json"
        analytics_json.write_text(
            json.dumps(portfolio_analytics_to_dict(analytics), indent=2, default=str),
            encoding="utf-8",
        )

    rebalance_json = None
    rebalance_markdown = None
    proposal = portfolio_state.get("rebalance_proposal")
    if proposal is not None:
        rebalance_json = portfolio_dir / "rebalance.json"
        rebalance_json.write_text(
            json.dumps(rebalance_proposal_to_dict(proposal), indent=2, default=str),
            encoding="utf-8",
        )
        rebalance_markdown = portfolio_dir / "rebalance.md"
        rebalance_markdown.write_text(
            render_rebalance_proposal(proposal),
            encoding="utf-8",
        )

    risk_markdown = _write_optional_markdown(
        portfolio_dir,
        "risk.md",
        portfolio_state.get("portfolio_risk_analysis"),
    )
    rebalance_review_markdown = _write_optional_markdown(
        portfolio_dir,
        "rebalance_review.md",
        portfolio_state.get("portfolio_rebalance_review"),
    )
    final_decision_markdown = _write_optional_markdown(
        portfolio_dir,
        "final_decision.md",
        portfolio_state.get("final_portfolio_decision"),
    )

    complete_report = save_path / "complete_report.md"
    complete_report.write_text(
        render_complete_portfolio_report(portfolio_state),
        encoding="utf-8",
    )

    return PortfolioReportPaths(
        root_dir=save_path,
        complete_report=complete_report,
        holdings_dir=holdings_dir,
        portfolio_dir=portfolio_dir,
        analytics_json=analytics_json,
        rebalance_json=rebalance_json,
        rebalance_markdown=rebalance_markdown,
        risk_markdown=risk_markdown,
        rebalance_review_markdown=rebalance_review_markdown,
        final_decision_markdown=final_decision_markdown,
    )


def render_holding_report(holding: dict[str, Any]) -> str:
    """Render one portfolio holding report to Markdown."""

    parts = [
        f"# {holding.get('symbol', 'Unknown')}",
        "",
        f"- Asset type: {holding.get('asset_type', 'unknown')}",
        f"- Current weight: {float(holding.get('current_weight', 0)):.2%}",
        f"- Analysis status: {holding.get('analysis_status', 'unknown')}",
    ]
    if holding.get("market_value") is not None:
        parts.append(f"- Market value: {holding['market_value']}")
    if holding.get("quantity") is not None:
        parts.append(f"- Quantity: {holding['quantity']}")
    if holding.get("skip_reason"):
        parts.append(f"- Skip reason: {holding['skip_reason']}")

    analysis = holding.get("analysis")
    if analysis:
        reports = analysis.get("reports") or {}
        parts.extend(["", "## Signal", str(analysis.get("signal", ""))])
        if reports:
            parts.append("")
            parts.append("## Analyst Reports")
            for name, report in reports.items():
                if report:
                    parts.extend(["", f"### {name.title()}", str(report)])
        if analysis.get("trader_investment_plan"):
            parts.extend(["", "## Trader Plan", str(analysis["trader_investment_plan"])])
        if analysis.get("final_trade_decision"):
            parts.extend(["", "## Final Decision", str(analysis["final_trade_decision"])])

    return "\n".join(parts)


def render_complete_portfolio_report(portfolio_state: dict[str, Any]) -> str:
    """Assemble the complete portfolio report Markdown."""

    request = portfolio_state.get("portfolio_request")
    trade_date = portfolio_state.get("trade_date") or getattr(request, "trade_date", "")
    sections = [
        "# Portfolio Analysis Report",
        "",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Trade date: {trade_date}",
        f"Base currency: {portfolio_state.get('base_currency', getattr(request, 'base_currency', ''))}",
        "",
        "## Holdings",
    ]
    for holding in portfolio_state.get("holdings", []):
        sections.append(
            f"- {holding.get('symbol')}: {float(holding.get('current_weight', 0)):.2%} "
            f"({holding.get('analysis_status')})"
        )

    if portfolio_state.get("portfolio_analytics"):
        sections.extend(["", "## Deterministic Portfolio Analytics", "See `portfolio/analytics.json`."])
    if portfolio_state.get("rebalance_proposal"):
        sections.extend(
            [
                "",
                "## Deterministic Rebalance Proposal",
                render_rebalance_proposal(portfolio_state["rebalance_proposal"]),
            ]
        )
    if portfolio_state.get("portfolio_risk_analysis"):
        sections.extend(["", "## Portfolio Risk Analysis", portfolio_state["portfolio_risk_analysis"]])
    if portfolio_state.get("portfolio_rebalance_review"):
        sections.extend(["", "## Portfolio Rebalance Review", portfolio_state["portfolio_rebalance_review"]])
    if portfolio_state.get("final_portfolio_decision"):
        sections.extend(["", "## Final Portfolio Decision", portfolio_state["final_portfolio_decision"]])
    return "\n".join(sections)


def _write_optional_markdown(
    portfolio_dir: Path,
    filename: str,
    content: str | None,
) -> Path | None:
    if not content:
        return None
    path = portfolio_dir / filename
    path.write_text(content, encoding="utf-8")
    return path
