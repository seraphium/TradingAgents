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
    extract_portfolio_allocation_summary,
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
    market_context_json: Path | None = None
    market_regime_json: Path | None = None
    data_quality_json: Path | None = None
    instrument_proposals_json: Path | None = None
    rebalance_json: Path | None = None
    rebalance_markdown: Path | None = None
    risk_markdown: Path | None = None
    rebalance_review_markdown: Path | None = None
    final_decision_markdown: Path | None = None
    final_result_json: Path | None = None
    proposals_dir: Path | None = None
    risk_validation_json: Path | None = None
    proposal_review_json: Path | None = None
    approval_json: Path | None = None


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
    proposals_dir = portfolio_dir / "proposals"
    proposals_dir.mkdir(exist_ok=True)

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

    market_context_json = None
    market_context = portfolio_state.get("portfolio_market_context")
    if market_context is not None:
        market_context_json = portfolio_dir / "market_context.json"
        market_context_json.write_text(
            json.dumps(_json_payload(market_context), indent=2, default=str),
            encoding="utf-8",
        )

    market_regime_json = None
    market_regime = portfolio_state.get("market_regime")
    if market_regime is not None:
        market_regime_json = portfolio_dir / "market_regime.json"
        market_regime_json.write_text(
            json.dumps(_json_payload(market_regime), indent=2, default=str),
            encoding="utf-8",
        )

    instrument_proposals_json = None
    instrument_proposals = portfolio_state.get("instrument_proposals")
    if instrument_proposals is not None:
        instrument_proposals_json = portfolio_dir / "instrument_proposals.json"
        instrument_proposals_json.write_text(
            json.dumps(
                {
                    symbol: _json_payload(proposal)
                    for symbol, proposal in instrument_proposals.items()
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    data_quality_json = None
    data_quality = portfolio_state.get("data_quality_assessment")
    if data_quality is not None:
        data_quality_json = portfolio_dir / "data_quality.json"
        data_quality_json.write_text(
            json.dumps(_json_payload(data_quality), indent=2, default=str),
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
            render_rebalance_proposal(
                proposal,
                final_summary=extract_portfolio_allocation_summary(
                    portfolio_state.get("final_portfolio_decision")
                ),
            ),
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
    risk_validation_json = _write_optional_json(
        portfolio_dir,
        "risk_validation.json",
        {"narrative": portfolio_state.get("portfolio_risk_analysis")},
        enabled=bool(portfolio_state.get("portfolio_risk_analysis")),
    )
    proposal_review_json = _write_optional_json(
        portfolio_dir,
        "proposal_review.json",
        {"narrative": portfolio_state.get("portfolio_rebalance_review")},
        enabled=bool(portfolio_state.get("portfolio_rebalance_review")),
    )
    approval_json = _write_optional_json(
        portfolio_dir,
        "approval.json",
        {
            "status": (
                "recorded"
                if portfolio_state.get("final_portfolio_decision")
                else "pending"
            ),
            "narrative": portfolio_state.get("final_portfolio_decision"),
        },
        enabled=bool(portfolio_state.get("final_portfolio_decision")),
    )

    proposal_history = portfolio_state.get("proposal_history") or []
    for index, entry in enumerate(proposal_history, start=1):
        version = entry.get("version", index) if isinstance(entry, dict) else index
        (proposals_dir / f"proposal_v{version}.json").write_text(
            json.dumps(_proposal_history_payload(entry), indent=2, default=str),
            encoding="utf-8",
        )

    final_result_json = save_path / "final_result.json"
    final_result_json.write_text(
        json.dumps(build_final_result(portfolio_state), indent=2, default=str),
        encoding="utf-8",
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
        market_context_json=market_context_json,
        market_regime_json=market_regime_json,
        data_quality_json=data_quality_json,
        instrument_proposals_json=instrument_proposals_json,
        rebalance_json=rebalance_json,
        rebalance_markdown=rebalance_markdown,
        risk_markdown=risk_markdown,
        rebalance_review_markdown=rebalance_review_markdown,
        final_decision_markdown=final_decision_markdown,
        final_result_json=final_result_json,
        proposals_dir=proposals_dir,
        risk_validation_json=risk_validation_json,
        proposal_review_json=proposal_review_json,
        approval_json=approval_json,
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
            parts.extend(
                ["", "## Trader Plan", str(analysis["trader_investment_plan"])]
            )
        if analysis.get("final_trade_decision"):
            parts.extend(
                ["", "## Final Decision", str(analysis["final_trade_decision"])]
            )

    return "\n".join(parts)


def render_portfolio_decision_summary(portfolio_state: dict[str, Any]) -> str:
    """Render the decision-first sections suitable for the CLI."""

    return "\n".join(_decision_first_sections(portfolio_state))


def render_complete_portfolio_report(portfolio_state: dict[str, Any]) -> str:
    """Assemble a decision-first report with verbose narratives in the appendix."""

    request = portfolio_state.get("portfolio_request")
    trade_date = portfolio_state.get("trade_date") or getattr(request, "trade_date", "")
    sections = [
        "# Portfolio Decision Report",
        "",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Trade date: {trade_date}",
        f"Base currency: {portfolio_state.get('base_currency', getattr(request, 'base_currency', ''))}",
        "",
        *_decision_first_sections(portfolio_state),
        "",
        "## Appendix: Detailed Agent Narratives and Diagnostics",
        "",
        "Machine-readable diagnostics are stored under `portfolio/`; detailed holding reports are stored under `holdings/`.",
    ]
    for title, key in (
        ("Risk Controller Narrative", "portfolio_risk_analysis"),
        ("Allocation Proposal Reviewer Narrative", "portfolio_rebalance_review"),
        ("Portfolio Decision Approver Narrative", "final_portfolio_decision"),
    ):
        if portfolio_state.get(key):
            sections.extend(["", f"### {title}", str(portfolio_state[key])])
    return "\n".join(sections)


def build_final_result(portfolio_state: dict[str, Any]) -> dict[str, Any]:
    """Build the concise machine-readable final portfolio decision."""

    proposal = portfolio_state.get("rebalance_proposal")
    regime = portfolio_state.get("market_regime")
    quality = portfolio_state.get("data_quality_assessment")
    components = getattr(proposal, "component_proposals", []) if proposal else []
    return {
        "portfolio_action": getattr(proposal, "portfolio_action", "Unavailable"),
        "approval_status": (
            "recorded" if portfolio_state.get("final_portfolio_decision") else "pending"
        ),
        "decision_summary": extract_portfolio_allocation_summary(
            portfolio_state.get("final_portfolio_decision")
        ),
        "market_regime": _json_payload(regime) if regime is not None else None,
        "data_quality": _json_payload(quality) if quality is not None else None,
        "final_allocations": [
            {
                "symbol": component.symbol,
                "initial_rating": _initial_rating(
                    portfolio_state, component.symbol, component.rating
                ),
                "current_weight": component.current_weight,
                "target_weight": component.target_weight,
                "weight_change": component.weight_change,
                "action": component.action,
                "final_reason": component.rationale,
            }
            for component in components
        ],
        "proposal_history": [
            _proposal_history_payload(entry)
            for entry in portfolio_state.get("proposal_history", [])
        ],
        "warnings": list(portfolio_state.get("warnings", [])),
    }


def _decision_first_sections(portfolio_state: dict[str, Any]) -> list[str]:
    proposal = portfolio_state.get("rebalance_proposal")
    regime = portfolio_state.get("market_regime")
    quality = portfolio_state.get("data_quality_assessment")
    analytics = portfolio_state.get("portfolio_analytics")
    action = getattr(proposal, "portfolio_action", "Unavailable")
    summary = extract_portfolio_allocation_summary(
        portfolio_state.get("final_portfolio_decision")
    )
    approval = (
        "Recorded" if portfolio_state.get("final_portfolio_decision") else "Pending"
    )
    regime_label = getattr(regime, "label", "Unavailable")
    quality_status = getattr(quality, "status", "Unavailable")
    sections = [
        "## Executive Decision",
        f"- **Approval status:** {approval}",
        f"- **Portfolio action:** {action}",
        f"- **Market regime:** {regime_label}",
        f"- **Data confidence:** {quality_status}",
        f"- **Decision summary:** {summary or getattr(proposal, 'rebalance_reason', 'No final summary available.')}",
        "",
        "## Final Rebalance Table",
        "| Instrument | Initial Rating | Current | Target | Change | Action | Final Reason |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for component in getattr(proposal, "component_proposals", []):
        sections.append(
            "| "
            + " | ".join(
                [
                    _escape_cell(component.symbol),
                    _escape_cell(
                        _initial_rating(
                            portfolio_state, component.symbol, component.rating
                        )
                    ),
                    f"{component.current_weight:.2%}",
                    f"{component.target_weight:.2%}",
                    f"{component.weight_change:+.2%}",
                    _escape_cell(component.action),
                    _escape_cell(component.rationale),
                ]
            )
            + " |"
        )
    if not getattr(proposal, "component_proposals", []):
        sections.append("| No final allocation available | — | — | — | — | — | — |")

    sections.extend(["", "## Current Market Trend and News Impact"])
    if regime is not None:
        sections.append(
            f"- **{getattr(regime, 'benchmark_symbol', None) or 'Benchmark'} / {regime_label}:** "
            f"{getattr(regime, 'summary', '')}"
        )
        sections.append(
            f"- News sentiment score: {getattr(regime, 'news_sentiment_score', 0):+.2f}."
        )
        sections.extend(
            f"- Allocation implication: {item}"
            for item in getattr(regime, "allocation_implications", [])
        )
    else:
        sections.append("- Market-regime evidence was not available for this run.")

    sections.extend(["", "## Portfolio Risk and Exposure Summary"])
    if analytics is not None:
        sections.append(
            f"- Portfolio volatility: {_format_optional_percent(getattr(analytics, 'portfolio_volatility', None))}"
        )
        sections.append(
            f"- Asset-type exposure: {_format_weight_map(getattr(analytics, 'asset_type_exposure', {}))}"
        )
        sections.append(
            f"- Sector exposure: {_format_weight_map(getattr(analytics, 'sector_exposure', {}))}"
        )
        sections.append(
            f"- Benchmark beta by instrument: {_format_optional_number_map(getattr(analytics, 'beta_by_symbol', {}))}"
        )
        sections.append(
            f"- Risk contribution by instrument: {_format_optional_percent_map(getattr(analytics, 'risk_contribution_by_symbol', {}))}"
        )
        flags = getattr(analytics, "constraint_flags", [])
        sections.extend(
            f"- Constraint warning: {getattr(flag, 'message', str(flag))}"
            for flag in flags
        )
    else:
        sections.append(
            "- Deterministic portfolio analytics were not available for this run."
        )
    if quality is not None:
        sections.append(f"- Data quality: {quality_status}.")
        sections.append(
            f"- Weighted analysis coverage: {quality.weighted_analysis_coverage:.2%}"
        )
        metric_coverage = getattr(quality, "metric_coverage", {})
        if metric_coverage:
            sections.append(
                "- Metric coverage: "
                + ", ".join(
                    f"{name} {coverage:.2%}"
                    for name, coverage in metric_coverage.items()
                )
            )
        sections.extend(
            f"- Blocking issue: {item}"
            for item in getattr(quality, "blocking_issues", [])
        )
        sections.extend(
            f"- Warning: {item}" for item in getattr(quality, "warnings", [])
        )
        sections.extend(
            f"- Restriction: {item}" for item in getattr(quality, "restrictions", [])
        )

    sections.extend(["", "## Instrument Analysis Summaries"])
    proposals = portfolio_state.get("instrument_proposals") or {}
    for component in getattr(proposal, "component_proposals", []):
        initial = proposals.get(component.symbol)
        detail = (
            getattr(initial, "summary", "")
            or component.single_stock_summary
            or "No instrument summary available."
        )
        sections.append(
            f"- **{component.symbol}:** initial {_initial_rating(portfolio_state, component.symbol, component.rating)}; "
            f"final {component.action} to {component.target_weight:.2%}. {detail}"
        )

    sections.extend(["", "## Rebalance Rationale and Proposal History"])
    sections.append(
        f"- Final rationale: {getattr(proposal, 'rebalance_reason', 'Unavailable')}"
    )
    history = portfolio_state.get("proposal_history") or []
    if history:
        for index, entry in enumerate(history, start=1):
            version = entry.get("version", index)
            sections.append(
                f"- Proposal v{version}: {entry.get('reason', 'No reason recorded')}"
            )
    else:
        sections.append("- No versioned proposal history was recorded.")
    return sections


def _initial_rating(portfolio_state: dict[str, Any], symbol: str, fallback: str) -> str:
    proposal = (portfolio_state.get("instrument_proposals") or {}).get(symbol)
    return str(getattr(proposal, "rating", fallback))


def _proposal_history_payload(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return _json_payload(entry)
    return {key: _json_payload(value) for key, value in entry.items()}


def _format_optional_percent(value: Any) -> str:
    return "Unavailable" if value is None else f"{float(value):.2%}"


def _format_weight_map(values: dict[str, float]) -> str:
    if not values:
        return "Unavailable"
    return ", ".join(f"{name} {weight:.2%}" for name, weight in sorted(values.items()))


def _format_optional_number_map(values: dict[str, float | None]) -> str:
    present = {name: value for name, value in values.items() if value is not None}
    if not present:
        return "Unavailable"
    return ", ".join(
        f"{name} {float(value):.2f}" for name, value in sorted(present.items())
    )


def _format_optional_percent_map(values: dict[str, float | None]) -> str:
    present = {name: value for name, value in values.items() if value is not None}
    if not present:
        return "Unavailable"
    return ", ".join(
        f"{name} {float(value):.2%}" for name, value in sorted(present.items())
    )


def _escape_cell(value: Any) -> str:
    return " ".join(str(value).replace("|", "\\|").split())


def _json_payload(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


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


def _write_optional_json(
    portfolio_dir: Path,
    filename: str,
    payload: Any,
    *,
    enabled: bool,
) -> Path | None:
    if not enabled:
        return None
    path = portfolio_dir / filename
    path.write_text(
        json.dumps(_json_payload(payload), indent=2, default=str), encoding="utf-8"
    )
    return path
