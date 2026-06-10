"""Deterministic portfolio rebalancing.

The engine in this module converts single-instrument ratings and portfolio
analytics into a first-pass target allocation. LLM agents may critique or
explain this proposal later, but the numeric proposal itself is deterministic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
import re
from typing import TYPE_CHECKING, Any, Literal, Mapping

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows.config import get_config
from tradingagents.portfolio.analytics import PortfolioAnalytics
from tradingagents.portfolio.schemas import AssetType, PortfolioRequest

if TYPE_CHECKING:
    from tradingagents.portfolio.data_quality import DataQualityAssessment
    from tradingagents.portfolio.instrument_proposals import InstrumentProposal
    from tradingagents.portfolio.market_regime import MarketRegime


RATING_SCORES: dict[str, float] = {
    "Buy": 1.0,
    "Overweight": 0.5,
    "Hold": 0.0,
    "Underweight": -0.5,
    "Sell": -1.0,
}


@dataclass(frozen=True)
class RebalanceComponent:
    """Deterministic target allocation for one portfolio component."""

    symbol: str
    asset_type: str
    current_weight: float
    target_weight: float
    weight_change: float
    action: str
    rating: str
    score: float
    rationale: str
    single_stock_summary: str = ""
    constraints_applied: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RebalanceProposal:
    """A versioned deterministic portfolio-level target allocation proposal."""

    proposal_id: str
    version: int
    input_hash: str
    portfolio_action: str
    component_proposals: list[RebalanceComponent]
    target_weights_by_symbol: dict[str, float]
    score_by_symbol: dict[str, float]
    risk_notes: list[str]
    rebalance_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_rebalance_proposal(
    portfolio_request: PortfolioRequest,
    analytics: PortfolioAnalytics,
    *,
    ratings_by_symbol: Mapping[str, str] | None = None,
    instrument_proposals: Mapping[str, "InstrumentProposal"] | None = None,
    data_quality: "DataQualityAssessment" | None = None,
    market_regime: "MarketRegime" | None = None,
    sector_by_symbol: Mapping[str, str] | None = None,
    liquidity_by_symbol: Mapping[str, float] | None = None,
    holdings: list[Mapping[str, Any]] | None = None,
    optimizer: Literal["heuristic", "mean_variance"] | None = None,
    score_step: float = 0.05,
    change_tolerance: float = 0.005,
    proposal_version: int = 1,
) -> RebalanceProposal:
    """Generate deterministic target weights and action labels.

    Ratings can be canonical values or prose containing a parseable rating.
    Optional liquidity scores are expected on a 0-1 scale where lower values
    indicate weaker liquidity.
    """

    ratings = ratings_by_symbol or {}
    structured_proposals = instrument_proposals or {}
    liquidity = liquidity_by_symbol or {}
    holding_evidence = _holding_evidence_by_symbol(holdings or [])
    bounds = _component_bounds(portfolio_request)
    if data_quality is not None:
        for symbol in data_quality.restricted_symbols:
            if symbol in bounds:
                minimum, maximum = bounds[symbol]
                current = next(
                    float(position.current_weight)
                    for position in portfolio_request.positions
                    if position.instrument.symbol == symbol
                )
                bounds[symbol] = (min(minimum, current), min(maximum, current))
    positions_by_symbol = {
        position.instrument.symbol: position for position in portfolio_request.positions
    }
    current_weights = {
        symbol: float(position.current_weight)
        for symbol, position in positions_by_symbol.items()
    }
    scores = {
        symbol: (
            structured_proposals[symbol].allocation_score
            if symbol in structured_proposals
            else _score_for_symbol(symbol, ratings)
        )
        for symbol in positions_by_symbol
    }
    risk_adjustments, risk_notes = _risk_adjustments(
        analytics,
        portfolio_request,
        positions_by_symbol,
        liquidity,
    )

    optimizer_mode = _optimizer_mode(portfolio_request, optimizer)
    diagnostics: dict[str, Any] = {
        "score_step": score_step,
        "change_tolerance": change_tolerance,
        "optimizer": optimizer_mode,
        "instrument_proposal_sources": {
            symbol: proposal.source for symbol, proposal in structured_proposals.items()
        },
        "data_quality_status": (
            data_quality.status if data_quality is not None else None
        ),
        "market_regime": (
            market_regime.model_dump(mode="json") if market_regime is not None else None
        ),
    }
    targets = dict(current_weights)
    constraints_applied = {symbol: [] for symbol in positions_by_symbol}
    if data_quality is not None:
        for symbol in data_quality.restricted_symbols:
            if symbol in constraints_applied:
                constraints_applied[symbol].append("data_quality_no_increase")
    if optimizer_mode == "mean_variance":
        targets, optimizer_diagnostics = _mean_variance_targets(
            portfolio_request,
            analytics,
            current_weights,
            scores,
            risk_adjustments,
            bounds,
        )
        diagnostics.update(optimizer_diagnostics)
        for symbol, target in targets.items():
            if abs(target - current_weights[symbol]) > change_tolerance:
                constraints_applied[symbol].append("mean_variance_optimizer")
    else:
        for symbol, position in positions_by_symbol.items():
            if position.instrument.asset_type == AssetType.CASH:
                continue
            adjustment = (
                scores[symbol] + risk_adjustments.get(symbol, 0.0)
            ) * score_step
            unclamped = targets[symbol] + adjustment
            targets[symbol] = _clamp(unclamped, *bounds[symbol])
            if targets[symbol] != unclamped:
                constraints_applied[symbol].append("component_bounds")

    targets = _apply_market_overlay(
        portfolio_request,
        targets,
        bounds,
        constraints_applied,
        market_regime,
        sector_by_symbol or {},
    )
    targets = _apply_option_limit(
        portfolio_request, targets, bounds, constraints_applied
    )
    targets = _apply_cash_budget(
        portfolio_request, analytics, targets, bounds, constraints_applied
    )
    targets = _normalize_targets(
        targets, bounds, scores, portfolio_request, constraints_applied
    )

    components: list[RebalanceComponent] = []
    for symbol, position in positions_by_symbol.items():
        current_weight = current_weights[symbol]
        target_weight = _round_weight(targets[symbol])
        weight_change = _round_weight(target_weight - current_weight)
        action = _action_for_change(current_weight, target_weight, change_tolerance)
        rating = (
            structured_proposals[symbol].rating
            if symbol in structured_proposals
            else _rating_for_symbol(symbol, ratings)
        )
        if position.instrument.asset_type == AssetType.CASH:
            rationale_parts = [
                "Cash sleeve is treated as portfolio funding and liquidity reserve."
            ]
        else:
            rationale_parts = [
                f"Instrument proposal rating {rating} contributes confidence-weighted optimizer score {scores[symbol]:+.2f}."
            ]
        if optimizer_mode == "mean_variance":
            rationale_parts.append("Target set by mean-variance optimizer.")
        if risk_adjustments.get(symbol):
            rationale_parts.append(f"Risk adjustment {risk_adjustments[symbol]:+.2f}.")
        if (
            "market_regime_overlay" in constraints_applied[symbol]
            and market_regime is not None
        ):
            rationale_parts.append(
                f"Validated {market_regime.label.value} market overlay applied."
            )
        if constraints_applied[symbol]:
            rationale_parts.append(
                "Applied constraints: " + ", ".join(constraints_applied[symbol]) + "."
            )
        single_stock_summary = _single_stock_summary(
            symbol,
            position.instrument.asset_type,
            rating,
            holding_evidence,
        )
        components.append(
            RebalanceComponent(
                symbol=symbol,
                asset_type=position.instrument.asset_type.value,
                current_weight=current_weight,
                target_weight=target_weight,
                weight_change=weight_change,
                action=action,
                rating=rating,
                score=scores[symbol],
                rationale=" ".join(rationale_parts),
                single_stock_summary=single_stock_summary,
                constraints_applied=constraints_applied[symbol],
            )
        )

    target_weights_by_symbol = {
        component.symbol: component.target_weight for component in components
    }
    portfolio_action = _portfolio_action(
        components,
        analytics,
        portfolio_request,
        change_tolerance,
    )
    rebalance_reason = _portfolio_rebalance_reason(
        components,
        risk_notes,
        analytics,
        portfolio_request,
    )

    if proposal_version < 1:
        raise ValueError("proposal_version must be at least 1")
    input_hash = _proposal_input_hash(
        portfolio_request,
        analytics,
        ratings,
        structured_proposals,
        data_quality,
        market_regime,
        sector_by_symbol,
        liquidity,
        holdings,
        optimizer_mode,
        score_step,
        change_tolerance,
    )
    return RebalanceProposal(
        proposal_id=f"rp-{input_hash[:16]}-v{proposal_version}",
        version=proposal_version,
        input_hash=input_hash,
        portfolio_action=portfolio_action,
        component_proposals=components,
        target_weights_by_symbol=target_weights_by_symbol,
        score_by_symbol=scores,
        risk_notes=risk_notes,
        rebalance_reason=rebalance_reason,
        diagnostics={
            **diagnostics,
            "target_sum": sum(target_weights_by_symbol.values()),
        },
    )


def extract_ratings_from_portfolio_result(
    portfolio_result: Mapping[str, Any],
) -> dict[str, str]:
    """Extract per-holding ratings from a ``propagate_portfolio`` result."""

    ratings: dict[str, str] = {}
    for holding in portfolio_result.get("holdings", []):
        symbol = holding.get("symbol")
        if not symbol:
            continue
        analysis = holding.get("analysis") or {}
        text = analysis.get("final_trade_decision") or analysis.get("signal") or ""
        ratings[str(symbol)] = parse_rating(str(text))
    return ratings


def extract_holding_evidence_from_portfolio_result(
    portfolio_result: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Return holding evidence suitable for ``generate_rebalance_proposal``."""

    return list(portfolio_result.get("holdings", []))


def render_rebalance_proposal(
    proposal: RebalanceProposal,
    *,
    final_summary: str | None = None,
) -> str:
    """Render a deterministic rebalance proposal to Markdown."""

    if _render_language() == "Chinese":
        return _render_rebalance_proposal_zh(proposal, final_summary=final_summary)

    parts = [
        f"**Portfolio Action**: {proposal.portfolio_action}",
        "",
        "**Why Rebalance**:",
        final_summary or proposal.rebalance_reason,
    ]
    if final_summary and proposal.rebalance_reason:
        parts.extend(
            ["", "**Deterministic Rebalance Basis**:", proposal.rebalance_reason]
        )
    parts.extend(
        [
            "",
            "**Component Summary**:",
            *[
                f"- {component.symbol}: {_component_rebalance_summary(component)}"
                for component in proposal.component_proposals
            ],
            "",
            "| Symbol | Current Weight | Target Weight | Change | Action | Rating | Score | Rationale |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
        ]
    )
    for component in proposal.component_proposals:
        parts.append(
            "| "
            + " | ".join(
                [
                    _escape_cell(component.symbol),
                    _format_weight(component.current_weight),
                    _format_weight(component.target_weight),
                    _format_signed_weight(component.weight_change),
                    component.action,
                    component.rating,
                    f"{component.score:+.2f}",
                    _escape_cell(component.rationale),
                ]
            )
            + " |"
        )
    if proposal.risk_notes:
        parts.extend(["", "**Risk Notes**:"])
        parts.extend(f"- {note}" for note in proposal.risk_notes)
    return "\n".join(parts)


def _render_rebalance_proposal_zh(
    proposal: RebalanceProposal,
    *,
    final_summary: str | None = None,
) -> str:
    parts = [
        f"**组合操作**：{_localize_action(proposal.portfolio_action)}",
        "",
        "**再平衡原因**：",
        final_summary or _localize_rebalance_text(proposal.rebalance_reason),
    ]
    if final_summary and proposal.rebalance_reason:
        parts.extend(
            [
                "",
                "**量化再平衡依据**：",
                _localize_rebalance_text(proposal.rebalance_reason),
            ]
        )
    parts.extend(
        [
            "",
            "**成分摘要**：",
            *[
                f"- {component.symbol}: {_component_rebalance_summary(component, localize=True)}"
                for component in proposal.component_proposals
            ],
            "",
            "| 标的 | 当前权重 | 目标权重 | 变化 | 操作 | 评级 | 分数 | 依据 |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
        ]
    )
    for component in proposal.component_proposals:
        parts.append(
            "| "
            + " | ".join(
                [
                    _escape_cell(component.symbol),
                    _format_weight(component.current_weight),
                    _format_weight(component.target_weight),
                    _format_signed_weight(component.weight_change),
                    _localize_action(component.action),
                    _localize_rating(component.rating),
                    f"{component.score:+.2f}",
                    _escape_cell(_localize_rebalance_text(component.rationale)),
                ]
            )
            + " |"
        )
    if proposal.risk_notes:
        parts.extend(["", "**风险提示**："])
        parts.extend(
            f"- {_localize_rebalance_text(note)}" for note in proposal.risk_notes
        )
    return "\n".join(parts)


def extract_portfolio_allocation_summary(final_decision: Any) -> str:
    """Extract the portfolio-level summary from a rendered manager decision."""

    if final_decision is None:
        return ""
    text = str(final_decision).strip()
    if not text:
        return ""
    match = re.search(
        r"(?ims)^\s*\*\*Summary\*\*\s*:\s*(.+?)(?=^\s*\*\*|\Z)",
        text,
    )
    if match:
        return " ".join(match.group(1).strip().split())
    return " ".join(text.split())


def rebalance_proposal_to_dict(proposal: RebalanceProposal) -> dict[str, Any]:
    return proposal.to_dict()


def _rating_for_symbol(symbol: str, ratings_by_symbol: Mapping[str, str]) -> str:
    return parse_rating(str(ratings_by_symbol.get(symbol, "Hold")))


def _score_for_symbol(symbol: str, ratings_by_symbol: Mapping[str, str]) -> float:
    return RATING_SCORES[_rating_for_symbol(symbol, ratings_by_symbol)]


def _holding_evidence_by_symbol(
    holdings: list[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(holding.get("symbol")): holding
        for holding in holdings
        if holding.get("symbol")
    }


def _single_stock_summary(
    symbol: str,
    asset_type: AssetType,
    rating: str,
    holding_evidence: Mapping[str, Mapping[str, Any]],
) -> str:
    if asset_type == AssetType.CASH:
        return "Cash is not single-stock analyzed; it is adjusted as portfolio liquidity and risk reserve."

    holding = holding_evidence.get(symbol)
    if not holding:
        return f"No detailed single-stock conclusion was supplied; parsed holding rating is {rating}."

    if holding.get("analysis_status") == "failed":
        error = (
            holding.get("error") or holding.get("skip_reason") or "analysis unavailable"
        )
        return (
            f"Single-stock analysis failed for this component: {_truncate_text(error)}"
        )

    analysis = holding.get("analysis") or {}
    final_decision = _clean_decision_text(analysis.get("final_trade_decision"))
    trader_plan = _clean_decision_text(analysis.get("trader_investment_plan"))
    risk_decision = _clean_decision_text(
        (analysis.get("risk_debate_state") or {}).get("judge_decision")
    )
    pieces = []
    if final_decision:
        pieces.append(f"Single-stock final conclusion: {final_decision}")
    if trader_plan:
        pieces.append(f"Trader plan: {trader_plan}")
    if risk_decision:
        pieces.append(f"Risk debate conclusion: {risk_decision}")
    if pieces:
        return " ".join(pieces)
    return f"Single-stock analysis completed; parsed holding rating is {rating}."


def _clean_decision_text(value: Any) -> str:
    if value is None:
        return ""
    lines = []
    for raw_line in str(value).replace("**", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("rating:"):
            continue
        lines.append(line)
    text = " ".join(lines)
    return _truncate_text(" ".join(text.split()))


def _truncate_text(value: Any, limit: int = 360) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 15, 0)].rstrip() + "... [truncated]"


def _component_bounds(
    portfolio_request: PortfolioRequest,
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    constraints = portfolio_request.constraints
    for position in portfolio_request.positions:
        symbol = position.instrument.symbol
        minimum = (
            position.target_min_weight
            if position.target_min_weight is not None
            else 0.0
        )
        maximum = (
            position.target_max_weight
            if position.target_max_weight is not None
            else 1.0
        )
        if (
            position.instrument.asset_type != AssetType.CASH
            and constraints.max_single_position_weight is not None
        ):
            maximum = min(maximum, constraints.max_single_position_weight)
        if position.instrument.asset_type == AssetType.CASH:
            if constraints.min_cash_weight is not None:
                minimum = max(minimum, constraints.min_cash_weight)
            if constraints.max_cash_weight is not None:
                maximum = min(maximum, constraints.max_cash_weight)
        bounds[symbol] = (float(minimum), float(maximum))
    return bounds


def _risk_adjustments(
    analytics: PortfolioAnalytics,
    portfolio_request: PortfolioRequest,
    positions_by_symbol: Mapping[str, Any],
    liquidity_by_symbol: Mapping[str, float],
) -> tuple[dict[str, float], list[str]]:
    adjustments = {symbol: 0.0 for symbol in positions_by_symbol}
    notes: list[str] = []
    custom = portfolio_request.constraints.custom
    custom_thresholds = {
        "high_volatility_threshold": _positive_float(
            custom.get("high_volatility_threshold"),
            default=0.45,
        ),
        "high_correlation_threshold": _positive_float(
            custom.get("high_correlation_threshold"),
            default=0.80,
        ),
        "max_underlying_exposure": _positive_float(
            custom.get("max_underlying_exposure"),
            default=0.40,
        ),
        "max_sector_exposure": _positive_float(
            custom.get("max_sector_exposure"),
            default=0.45,
        ),
        "max_theme_exposure": _positive_float(
            custom.get("max_theme_exposure"),
            default=0.45,
        ),
        "min_liquidity_score": _positive_float(
            custom.get("min_liquidity_score"),
            default=0.35,
        ),
    }

    for symbol in positions_by_symbol:
        volatility = analytics.volatility_by_symbol.get(symbol)
        if (
            _is_number(volatility)
            and volatility > custom_thresholds["high_volatility_threshold"]
        ):
            adjustments[symbol] -= 0.5
            notes.append(f"{symbol} volatility is elevated at {volatility:.2%}.")

        average_correlation = _average_correlation(symbol, analytics.correlation_matrix)
        if (
            average_correlation is not None
            and average_correlation > custom_thresholds["high_correlation_threshold"]
        ):
            adjustments[symbol] -= 0.25
            notes.append(
                f"{symbol} has high average correlation at {average_correlation:.2f}."
            )

        liquidity_score = liquidity_by_symbol.get(symbol)
        if (
            _is_number(liquidity_score)
            and liquidity_score < custom_thresholds["min_liquidity_score"]
        ):
            adjustments[symbol] -= 0.5
            notes.append(f"{symbol} liquidity score is weak at {liquidity_score:.2f}.")

    for exposure_symbol, exposure in analytics.underlying_exposure.items():
        if exposure <= custom_thresholds["max_underlying_exposure"]:
            continue
        notes.append(f"{exposure_symbol} exposure is concentrated at {exposure:.2%}.")
        for symbol, position in positions_by_symbol.items():
            instrument = position.instrument
            underlying = (
                instrument.underlying
                if instrument.asset_type == AssetType.OPTION
                else instrument.symbol
            )
            if underlying == exposure_symbol:
                adjustments[symbol] -= 0.25

    classification_by_name = _classification_groups(portfolio_request)
    for sector, exposure in analytics.sector_exposure.items():
        if exposure <= custom_thresholds["max_sector_exposure"]:
            continue
        notes.append(f"{sector} sector exposure is elevated at {exposure:.2%}.")
        for symbol in _symbols_in_classification(
            positions_by_symbol,
            classification_by_name.get("sector", {}),
            sector,
        ):
            adjustments[symbol] -= 0.25

    for theme, symbol_map in classification_by_name.items():
        if theme == "sector":
            continue
        exposure_by_group = _classification_exposure(
            positions_by_symbol,
            symbol_map,
        )
        for group_name, exposure in exposure_by_group.items():
            if exposure <= custom_thresholds["max_theme_exposure"]:
                continue
            notes.append(
                f"{group_name} {theme} exposure is elevated at {exposure:.2%}."
            )
            for symbol in _symbols_in_classification(
                positions_by_symbol,
                symbol_map,
                group_name,
            ):
                adjustments[symbol] -= 0.25

    return adjustments, notes


def _classification_groups(
    portfolio_request: PortfolioRequest,
) -> dict[str, dict[str, str]]:
    custom = portfolio_request.constraints.custom
    configured = {
        "sector": custom.get("sector_by_symbol") or custom.get("sectors_by_symbol"),
        "industry": custom.get("industry_by_symbol")
        or custom.get("industries_by_symbol"),
        "theme": custom.get("theme_by_symbol") or custom.get("themes_by_symbol"),
    }
    groups: dict[str, dict[str, str]] = {}
    for classification_name, raw_mapping in configured.items():
        if not isinstance(raw_mapping, Mapping):
            continue
        symbol_map = {
            str(symbol).strip().upper(): str(group).strip()
            for symbol, group in raw_mapping.items()
            if str(symbol).strip() and str(group).strip()
        }
        if symbol_map:
            groups[classification_name] = symbol_map
    return groups


def _classification_exposure(
    positions_by_symbol: Mapping[str, Any],
    symbol_map: Mapping[str, str],
) -> dict[str, float]:
    exposure: dict[str, float] = {}
    for symbol, position in positions_by_symbol.items():
        lookup_symbol = _classification_lookup_symbol(position)
        group = symbol_map.get(lookup_symbol) or symbol_map.get(str(symbol).upper())
        if not group:
            continue
        exposure[group] = exposure.get(group, 0.0) + float(position.current_weight)
    return exposure


def _symbols_in_classification(
    positions_by_symbol: Mapping[str, Any],
    symbol_map: Mapping[str, str],
    group_name: str,
) -> list[str]:
    symbols = []
    for symbol, position in positions_by_symbol.items():
        lookup_symbol = _classification_lookup_symbol(position)
        group = symbol_map.get(lookup_symbol) or symbol_map.get(str(symbol).upper())
        if group == group_name:
            symbols.append(symbol)
    return symbols


def _classification_lookup_symbol(position: Any) -> str:
    instrument = position.instrument
    if instrument.asset_type == AssetType.OPTION and instrument.underlying:
        return str(instrument.underlying).upper()
    return str(instrument.symbol).upper()


def _apply_market_overlay(
    portfolio_request: PortfolioRequest,
    targets: dict[str, float],
    bounds: Mapping[str, tuple[float, float]],
    constraints_applied: dict[str, list[str]],
    market_regime: "MarketRegime" | None,
    sector_by_symbol: Mapping[str, str],
) -> dict[str, float]:
    """Apply a validated market overlay, then leave hard bounds to final normalization."""

    if market_regime is None or market_regime.confidence < 0.25:
        return targets
    overlay = market_regime.overlay
    cash_symbols = _cash_symbols(portfolio_request)
    risky_symbols = [
        position.instrument.symbol
        for position in portfolio_request.positions
        if position.instrument.asset_type != AssetType.CASH
    ]
    cash_shift = overlay.cash_weight_adjustment
    if cash_shift > 0 and cash_symbols:
        removed = _remove_delta(
            targets,
            cash_shift,
            bounds,
            risky_symbols,
            constraints_applied,
            "market_regime_overlay",
        )
        targets = _distribute_delta(
            targets,
            removed,
            bounds,
            cash_symbols,
            constraints_applied,
            "market_regime_overlay",
        )
    elif cash_shift < 0 and cash_symbols:
        removed = _remove_delta(
            targets,
            abs(cash_shift),
            bounds,
            cash_symbols,
            constraints_applied,
            "market_regime_overlay",
        )
        targets = _distribute_delta(
            targets,
            removed,
            bounds,
            risky_symbols,
            constraints_applied,
            "market_regime_overlay",
        )

    positions_by_symbol = {
        position.instrument.symbol: position for position in portfolio_request.positions
    }
    for sector, adjustment in overlay.sector_weight_adjustments.items():
        sector_symbols = _symbols_in_classification(
            positions_by_symbol, sector_by_symbol, sector
        )
        other_risky = [
            symbol for symbol in risky_symbols if symbol not in sector_symbols
        ]
        if not sector_symbols or not other_risky or abs(adjustment) < 1e-12:
            continue
        if adjustment > 0:
            removed = _remove_delta(
                targets,
                adjustment,
                bounds,
                other_risky,
                constraints_applied,
                "market_regime_overlay",
            )
            targets = _distribute_delta(
                targets,
                removed,
                bounds,
                sector_symbols,
                constraints_applied,
                "market_regime_overlay",
            )
        else:
            removed = _remove_delta(
                targets,
                abs(adjustment),
                bounds,
                sector_symbols,
                constraints_applied,
                "market_regime_overlay",
            )
            targets = _distribute_delta(
                targets,
                removed,
                bounds,
                other_risky,
                constraints_applied,
                "market_regime_overlay",
            )
    return targets


def _apply_option_limit(
    portfolio_request: PortfolioRequest,
    targets: dict[str, float],
    bounds: Mapping[str, tuple[float, float]],
    constraints_applied: dict[str, list[str]],
) -> dict[str, float]:
    max_options_weight = portfolio_request.constraints.max_options_weight
    if max_options_weight is None:
        return targets

    option_symbols = [
        position.instrument.symbol
        for position in portfolio_request.positions
        if position.instrument.asset_type == AssetType.OPTION
    ]
    option_total = sum(targets[symbol] for symbol in option_symbols)
    if option_total <= max_options_weight or option_total <= 0:
        return targets

    scale = max_options_weight / option_total
    freed = 0.0
    for symbol in option_symbols:
        old_target = targets[symbol]
        targets[symbol] = max(bounds[symbol][0], targets[symbol] * scale)
        freed += old_target - targets[symbol]
        constraints_applied[symbol].append("max_options_weight")
    return _distribute_delta(
        targets,
        freed,
        bounds,
        _cash_first_order(portfolio_request),
        constraints_applied,
        "max_options_weight",
    )


def _apply_cash_budget(
    portfolio_request: PortfolioRequest,
    analytics: PortfolioAnalytics,
    targets: dict[str, float],
    bounds: Mapping[str, tuple[float, float]],
    constraints_applied: dict[str, list[str]],
) -> dict[str, float]:
    max_portfolio_volatility = portfolio_request.constraints.custom.get(
        "max_portfolio_volatility"
    )
    if (
        not _is_number(max_portfolio_volatility)
        or analytics.portfolio_volatility is None
        or analytics.portfolio_volatility <= max_portfolio_volatility
    ):
        return targets

    cash_symbols = _cash_symbols(portfolio_request)
    if not cash_symbols:
        return targets

    reduction_budget = min(
        0.05,
        analytics.portfolio_volatility - float(max_portfolio_volatility),
    )
    risky_order = [
        position.instrument.symbol
        for position in portfolio_request.positions
        if position.instrument.asset_type != AssetType.CASH
    ]
    removed = _remove_delta(
        targets,
        reduction_budget,
        bounds,
        risky_order,
        constraints_applied,
        "risk_budget",
    )
    return _distribute_delta(
        targets, removed, bounds, cash_symbols, constraints_applied, "risk_budget"
    )


def _optimizer_mode(
    portfolio_request: PortfolioRequest,
    optimizer: Literal["heuristic", "mean_variance"] | None,
) -> Literal["heuristic", "mean_variance"]:
    configured = optimizer or portfolio_request.constraints.custom.get(
        "rebalance_optimizer",
        "heuristic",
    )
    normalized = str(configured).strip().lower().replace("-", "_")
    if normalized in {"mean_variance", "mean_variance_optimizer", "optimizer"}:
        return "mean_variance"
    return "heuristic"


def _mean_variance_targets(
    portfolio_request: PortfolioRequest,
    analytics: PortfolioAnalytics,
    current_weights: Mapping[str, float],
    scores: Mapping[str, float],
    risk_adjustments: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, float], dict[str, Any]]:
    symbols = [position.instrument.symbol for position in portfolio_request.positions]
    expected_returns = _optimizer_expected_returns(
        portfolio_request,
        symbols,
        scores,
        risk_adjustments,
    )
    covariance = _optimizer_covariance(portfolio_request, analytics, symbols)
    optimizer_config = portfolio_request.constraints.custom
    risk_aversion = _positive_float(
        optimizer_config.get("optimizer_risk_aversion"),
        default=1.0,
    )
    turnover_penalty = _positive_float(
        optimizer_config.get("optimizer_turnover_penalty"),
        default=0.25,
    )
    learning_rate = _positive_float(
        optimizer_config.get("optimizer_learning_rate"),
        default=0.20,
    )
    max_iterations = int(optimizer_config.get("optimizer_iterations", 250))
    max_iterations = max(1, min(max_iterations, 2000))
    tolerance = _positive_float(
        optimizer_config.get("optimizer_tolerance"),
        default=1e-9,
    )

    lows = [bounds[symbol][0] for symbol in symbols]
    highs = [bounds[symbol][1] for symbol in symbols]
    current = [current_weights[symbol] for symbol in symbols]
    weights = _project_to_bounded_simplex(current, lows, highs)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        gradient = _optimizer_gradient(
            weights,
            current,
            covariance,
            expected_returns,
            risk_aversion,
            turnover_penalty,
        )
        candidate = [
            weight - learning_rate * gradient_value
            for weight, gradient_value in zip(weights, gradient)
        ]
        projected = _project_to_bounded_simplex(candidate, lows, highs)
        if max(abs(new - old) for new, old in zip(projected, weights)) < tolerance:
            weights = projected
            break
        weights = projected

    targets = dict(zip(symbols, weights))
    return targets, {
        "optimizer_iterations": iterations,
        "optimizer_risk_aversion": risk_aversion,
        "optimizer_turnover_penalty": turnover_penalty,
        "optimizer_objective": _optimizer_objective(
            weights,
            current,
            covariance,
            expected_returns,
            risk_aversion,
            turnover_penalty,
        ),
    }


def _optimizer_expected_returns(
    portfolio_request: PortfolioRequest,
    symbols: list[str],
    scores: Mapping[str, float],
    risk_adjustments: Mapping[str, float],
) -> list[float]:
    scale = _positive_float(
        portfolio_request.constraints.custom.get("optimizer_score_scale"),
        default=0.05,
    )
    expected = []
    for symbol in symbols:
        if _position_asset_type(portfolio_request, symbol) == AssetType.CASH:
            expected.append(0.0)
        else:
            expected.append(
                (scores[symbol] + risk_adjustments.get(symbol, 0.0)) * scale
            )
    return expected


def _optimizer_covariance(
    portfolio_request: PortfolioRequest,
    analytics: PortfolioAnalytics,
    symbols: list[str],
) -> list[list[float]]:
    default_volatility = _positive_float(
        portfolio_request.constraints.custom.get("optimizer_default_volatility"),
        default=0.20,
    )
    default_option_volatility = _positive_float(
        portfolio_request.constraints.custom.get("optimizer_default_option_volatility"),
        default=0.60,
    )
    volatilities = []
    for symbol in symbols:
        volatility = analytics.volatility_by_symbol.get(symbol)
        if not _is_number(volatility):
            volatility = (
                0.0
                if _position_asset_type(portfolio_request, symbol) == AssetType.CASH
                else (
                    default_option_volatility
                    if _position_asset_type(portfolio_request, symbol)
                    == AssetType.OPTION
                    else default_volatility
                )
            )
        volatilities.append(float(volatility))

    covariance: list[list[float]] = []
    for left_index, left_symbol in enumerate(symbols):
        row = []
        for right_index, right_symbol in enumerate(symbols):
            if left_index == right_index:
                correlation = 1.0
            else:
                correlation = analytics.correlation_matrix.get(left_symbol, {}).get(
                    right_symbol,
                    0.0,
                )
                if not _is_number(correlation):
                    correlation = 0.0
            row.append(
                float(correlation)
                * volatilities[left_index]
                * volatilities[right_index]
            )
        covariance.append(row)
    return covariance


def _optimizer_gradient(
    weights: list[float],
    current_weights: list[float],
    covariance: list[list[float]],
    expected_returns: list[float],
    risk_aversion: float,
    turnover_penalty: float,
) -> list[float]:
    gradient = []
    for row_index, row in enumerate(covariance):
        risk_gradient = (
            2.0
            * risk_aversion
            * sum(
                covariance_value * weight
                for covariance_value, weight in zip(row, weights)
            )
        )
        return_gradient = -expected_returns[row_index]
        turnover_gradient = (
            2.0 * turnover_penalty * (weights[row_index] - current_weights[row_index])
        )
        gradient.append(risk_gradient + return_gradient + turnover_gradient)
    return gradient


def _optimizer_objective(
    weights: list[float],
    current_weights: list[float],
    covariance: list[list[float]],
    expected_returns: list[float],
    risk_aversion: float,
    turnover_penalty: float,
) -> float:
    variance = 0.0
    for row_index, row in enumerate(covariance):
        variance += weights[row_index] * sum(
            covariance_value * weight for covariance_value, weight in zip(row, weights)
        )
    expected_return = sum(
        weight * expected_return
        for weight, expected_return in zip(weights, expected_returns)
    )
    turnover = sum(
        (weight - current_weight) ** 2
        for weight, current_weight in zip(weights, current_weights)
    )
    return risk_aversion * variance - expected_return + turnover_penalty * turnover


def _project_to_bounded_simplex(
    values: list[float],
    lows: list[float],
    highs: list[float],
) -> list[float]:
    low_sum = sum(lows)
    high_sum = sum(highs)
    if low_sum > 1.0 + 1e-9 or high_sum < 1.0 - 1e-9:
        raise ValueError("portfolio target weight bounds are infeasible")

    lower_theta = min(value - high for value, high in zip(values, highs)) - 1.0
    upper_theta = max(value - low for value, low in zip(values, lows)) + 1.0
    projected = list(values)
    for _ in range(100):
        theta = (lower_theta + upper_theta) / 2.0
        projected = [
            _clamp(value - theta, low, high)
            for value, low, high in zip(values, lows, highs)
        ]
        total = sum(projected)
        if total > 1.0:
            lower_theta = theta
        else:
            upper_theta = theta
    return projected


def _position_asset_type(
    portfolio_request: PortfolioRequest,
    symbol: str,
) -> AssetType:
    for position in portfolio_request.positions:
        if position.instrument.symbol == symbol:
            return position.instrument.asset_type
    raise KeyError(symbol)


def _positive_float(value: Any, *, default: float) -> float:
    if not _is_number(value) or float(value) <= 0:
        return default
    return float(value)


def _normalize_targets(
    targets: dict[str, float],
    bounds: Mapping[str, tuple[float, float]],
    scores: Mapping[str, float],
    portfolio_request: PortfolioRequest,
    constraints_applied: dict[str, list[str]],
) -> dict[str, float]:
    total = sum(targets.values())
    difference = 1.0 - total
    if abs(difference) <= 1e-10:
        return targets
    if difference > 0:
        order = _cash_first_order(portfolio_request) + _positive_score_order(
            scores, portfolio_request
        )
        _distribute_delta(
            targets, difference, bounds, order, constraints_applied, "normalization"
        )
    else:
        order = _cash_symbols(portfolio_request) + _negative_score_order(
            scores, portfolio_request
        )
        _remove_delta(
            targets,
            abs(difference),
            bounds,
            order,
            constraints_applied,
            "normalization",
        )
    return targets


def _distribute_delta(
    targets: dict[str, float],
    delta: float,
    bounds: Mapping[str, tuple[float, float]],
    order: list[str],
    constraints_applied: dict[str, list[str]],
    reason: str,
) -> dict[str, float]:
    remaining = delta
    for symbol in _dedupe(order):
        if remaining <= 1e-10:
            break
        capacity = bounds[symbol][1] - targets[symbol]
        if capacity <= 0:
            continue
        addition = min(capacity, remaining)
        targets[symbol] += addition
        remaining -= addition
        constraints_applied[symbol].append(reason)
    return targets


def _remove_delta(
    targets: dict[str, float],
    delta: float,
    bounds: Mapping[str, tuple[float, float]],
    order: list[str],
    constraints_applied: dict[str, list[str]],
    reason: str,
) -> float:
    remaining = delta
    removed = 0.0
    for symbol in _dedupe(order):
        if remaining <= 1e-10:
            break
        reducible = targets[symbol] - bounds[symbol][0]
        if reducible <= 0:
            continue
        reduction = min(reducible, remaining)
        targets[symbol] -= reduction
        remaining -= reduction
        removed += reduction
        constraints_applied[symbol].append(reason)
    return removed


def _portfolio_action(
    components: list[RebalanceComponent],
    analytics: PortfolioAnalytics,
    portfolio_request: PortfolioRequest,
    change_tolerance: float,
) -> str:
    if all(
        abs(component.weight_change) <= change_tolerance for component in components
    ):
        return "Hold"

    max_portfolio_volatility = portfolio_request.constraints.custom.get(
        "max_portfolio_volatility"
    )
    if (
        _is_number(max_portfolio_volatility)
        and analytics.portfolio_volatility is not None
        and analytics.portfolio_volatility > max_portfolio_volatility
    ):
        return "De-risk"

    current_cash = sum(
        component.current_weight
        for component in components
        if component.asset_type == AssetType.CASH.value
    )
    target_cash = next(
        (
            component.target_weight
            for component in components
            if component.asset_type == AssetType.CASH.value
        ),
        current_cash,
    )
    if target_cash < current_cash - change_tolerance:
        return "Increase Risk"
    return "Rebalance"


def _portfolio_rebalance_reason(
    components: list[RebalanceComponent],
    risk_notes: list[str],
    analytics: PortfolioAnalytics,
    portfolio_request: PortfolioRequest,
) -> str:
    changed = [component for component in components if component.action != "Hold"]
    if not changed:
        return "All components remain within tolerance, so no material allocation change is proposed."

    risk_reductions = [
        component
        for component in changed
        if component.weight_change < 0 and component.asset_type != AssetType.CASH.value
    ]
    risk_adds = [
        component
        for component in changed
        if component.weight_change > 0 and component.asset_type != AssetType.CASH.value
    ]
    cash_changes = [
        component
        for component in changed
        if component.asset_type == AssetType.CASH.value
    ]

    clauses = []
    portfolio_risk_context = _portfolio_risk_context(
        analytics,
        portfolio_request,
        risk_notes,
    )
    if portfolio_risk_context:
        clauses.append("Portfolio-level risk view: " + " ".join(portfolio_risk_context))
    signal_context = _aggregate_signal_context(components)
    if signal_context:
        clauses.append(signal_context)
    if risk_reductions:
        clauses.append(
            "The optimizer reduces the sleeves that add the most unwanted risk or have weaker aggregate conviction."
        )
    if risk_adds:
        clauses.append(
            "Freed capital is redeployed into components that better fit the portfolio risk budget and single-stock conviction set."
        )
    if cash_changes:
        clauses.append(
            "Cash is adjusted as the funding and liquidity reserve after the risky-asset targets are set."
        )
    return " ".join(clauses)


def _portfolio_risk_context(
    analytics: PortfolioAnalytics,
    portfolio_request: PortfolioRequest,
    risk_notes: list[str],
) -> list[str]:
    context: list[str] = []
    max_portfolio_volatility = portfolio_request.constraints.custom.get(
        "max_portfolio_volatility"
    )
    if analytics.portfolio_volatility is not None:
        if _is_number(max_portfolio_volatility):
            relation = (
                "above"
                if analytics.portfolio_volatility > float(max_portfolio_volatility)
                else "inside"
            )
            context.append(
                f"total volatility is {analytics.portfolio_volatility:.2%}, "
                f"{relation} the configured {float(max_portfolio_volatility):.2%} budget."
            )
        else:
            context.append(f"total volatility is {analytics.portfolio_volatility:.2%}.")

    exposure_context = _exposure_context(analytics, portfolio_request)
    if exposure_context:
        context.extend(exposure_context)

    if analytics.aggregate_option_greeks:
        greek_parts = [
            f"{name} {value:.2f}"
            for name, value in analytics.aggregate_option_greeks.items()
            if _is_number(value) and abs(value) > 1e-9
        ]
        if greek_parts:
            context.append(
                "aggregate option Greeks are " + ", ".join(greek_parts) + "."
            )

    risk_flag_notes = [
        note
        for note in risk_notes
        if "sector exposure" not in note.lower()
        and "theme exposure" not in note.lower()
    ]
    if risk_flag_notes:
        context.append(
            "Risk flags: "
            + " ".join(note.rstrip(".") + "." for note in risk_flag_notes)
        )
    return context


def _exposure_context(
    analytics: PortfolioAnalytics,
    portfolio_request: PortfolioRequest,
) -> list[str]:
    custom = portfolio_request.constraints.custom
    sector_threshold = _positive_float(custom.get("max_sector_exposure"), default=0.45)
    theme_threshold = _positive_float(custom.get("max_theme_exposure"), default=0.45)
    context: list[str] = []

    elevated_sectors = [
        f"{sector} {exposure:.2%}"
        for sector, exposure in sorted(
            analytics.sector_exposure.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if exposure > sector_threshold
    ]
    if elevated_sectors:
        context.append(
            "sector exposure is concentrated in " + ", ".join(elevated_sectors) + "."
        )

    positions_by_symbol = {
        position.instrument.symbol: position for position in portfolio_request.positions
    }
    for classification_name, symbol_map in _classification_groups(
        portfolio_request
    ).items():
        if classification_name == "sector":
            continue
        elevated_groups = [
            f"{group_name} {exposure:.2%}"
            for group_name, exposure in sorted(
                _classification_exposure(positions_by_symbol, symbol_map).items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if exposure > theme_threshold
        ]
        if elevated_groups:
            context.append(
                f"{classification_name} exposure is concentrated in "
                + ", ".join(elevated_groups)
                + "."
            )

    option_exposure = analytics.asset_type_exposure.get(AssetType.OPTION.value, 0.0)
    if option_exposure > 0:
        context.append(f"options exposure is {option_exposure:.2%}.")
    return context


def _aggregate_signal_context(components: list[RebalanceComponent]) -> str:
    non_cash = [
        component
        for component in components
        if component.asset_type != AssetType.CASH.value
    ]
    if not non_cash:
        return ""
    positive = sum(
        1 for component in non_cash if component.rating in {"Buy", "Overweight"}
    )
    negative = sum(
        1 for component in non_cash if component.rating in {"Sell", "Underweight"}
    )
    neutral = len(non_cash) - positive - negative
    added_weight = sum(
        component.weight_change for component in non_cash if component.weight_change > 0
    )
    reduced_weight = -sum(
        component.weight_change for component in non_cash if component.weight_change < 0
    )
    return (
        "Aggregate single-stock signals are "
        f"{positive} positive, {neutral} neutral, and {negative} negative; "
        f"the proposed trade set adds {_format_weight(added_weight)} to selected risky assets "
        f"and trims {_format_weight(reduced_weight)} from riskier or lower-conviction sleeves."
    )


def _component_rebalance_summary(
    component: RebalanceComponent,
    *,
    localize: bool = False,
) -> str:
    if localize:
        return (
            f"{_localize_action(component.action)}："
            f"{_format_weight(component.current_weight)} -> "
            f"{_format_weight(component.target_weight)} "
            f"({_format_signed_weight(component.weight_change)})；"
            f"{_localize_rebalance_text(component.single_stock_summary)}"
        )
    return (
        f"{component.action} from {_format_weight(component.current_weight)} "
        f"to {_format_weight(component.target_weight)} "
        f"({_format_signed_weight(component.weight_change)}); "
        f"{component.single_stock_summary}"
    )


def _render_language() -> str:
    return str(get_config().get("output_language", "English")).strip()


def _localize_action(action: str) -> str:
    return {
        "Buy": "买入",
        "Sell": "卖出",
        "Add": "增持",
        "Trim": "减持",
        "Hold": "持有",
        "Rebalance": "再平衡",
        "De-risk": "降低风险",
        "Increase Risk": "提高风险",
    }.get(action, action)


def _localize_rating(rating: str) -> str:
    return {
        "Buy": "买入",
        "Overweight": "超配",
        "Hold": "持有",
        "Underweight": "低配",
        "Sell": "卖出",
    }.get(rating, rating)


def _localize_rebalance_text(text: str) -> str:
    replacements = [
        (
            "All components remain within tolerance, so no material allocation change is proposed.",
            "所有成分仍在容忍区间内，因此不建议进行实质性配置调整。",
        ),
        ("Portfolio-level risk view: ", "组合层面风险视角："),
        ("total volatility is ", "总波动率为 "),
        (", above the configured ", "，高于设定的 "),
        (", inside the configured ", "，处于设定的 "),
        (" budget.", " 风险预算内。"),
        ("sector exposure is concentrated in ", "行业敞口集中在 "),
        (" exposure is concentrated in ", " 敞口集中在 "),
        ("options exposure is ", "期权敞口为 "),
        ("aggregate option Greeks are ", "汇总期权 Greeks 为 "),
        ("Risk flags: ", "风险标记："),
        ("Aggregate single-stock signals are ", "汇总个股信号为 "),
        (" positive, ", " 个正面、"),
        (" neutral, and ", " 个中性、"),
        (
            " negative; the proposed trade set adds ",
            " 个负面；拟议交易组合向选定风险资产增加 ",
        ),
        (
            " to selected risky assets and trims ",
            "，并从风险较高或低确信度仓位削减 ",
        ),
        (" from riskier or lower-conviction sleeves.", "。"),
        (
            "The optimizer reduces the sleeves that add the most unwanted risk or have weaker aggregate conviction.",
            "优化器会降低带来较多非预期风险或整体确信度较弱的仓位。",
        ),
        (
            "Freed capital is redeployed into components that better fit the portfolio risk budget and single-stock conviction set.",
            "释放出的资金会重新配置到更符合组合风险预算和个股确信度的成分。",
        ),
        (
            "Cash is adjusted as the funding and liquidity reserve after the risky-asset targets are set.",
            "在风险资产目标确定后，现金作为资金来源和流动性储备进行调整。",
        ),
        (
            "Cash sleeve is treated as portfolio funding and liquidity reserve.",
            "现金仓位被视为组合资金来源和流动性储备。",
        ),
        ("Parsed single-stock rating ", "解析出的个股评级为 "),
        (" contributes optimizer score ", "，贡献优化器分数 "),
        ("Target set by mean-variance optimizer.", "目标权重由均值-方差优化器设定。"),
        ("Risk adjustment ", "风险调整 "),
        ("Applied constraints: ", "已应用约束："),
        (
            "Cash is not single-stock analyzed; it is adjusted as portfolio liquidity and risk reserve.",
            "现金不进行个股分析；它作为组合流动性和风险储备进行调整。",
        ),
        (
            "No detailed single-stock conclusion was supplied; parsed holding rating is ",
            "未提供详细个股结论；解析出的持仓评级为 ",
        ),
        (
            "Single-stock analysis failed for this component: ",
            "该成分的个股分析失败：",
        ),
        ("Single-stock final conclusion: ", "个股最终结论："),
        ("Trader plan: ", "交易员计划："),
        ("Risk debate conclusion: ", "风险辩论结论："),
        (
            "Single-stock analysis completed; parsed holding rating is ",
            "个股分析已完成；解析出的持仓评级为 ",
        ),
    ]
    localized = text
    for source, target in replacements:
        localized = localized.replace(source, target)
    return localized


def _action_for_change(
    current_weight: float, target_weight: float, tolerance: float
) -> str:
    change = target_weight - current_weight
    if current_weight > tolerance and target_weight <= tolerance:
        return "Sell"
    if current_weight <= tolerance and target_weight > tolerance:
        return "Buy"
    if change > tolerance:
        return "Add"
    if change < -tolerance:
        return "Trim"
    return "Hold"


def _cash_symbols(portfolio_request: PortfolioRequest) -> list[str]:
    return [
        position.instrument.symbol
        for position in portfolio_request.positions
        if position.instrument.asset_type == AssetType.CASH
    ]


def _cash_first_order(portfolio_request: PortfolioRequest) -> list[str]:
    cash = _cash_symbols(portfolio_request)
    non_cash = [
        position.instrument.symbol
        for position in portfolio_request.positions
        if position.instrument.asset_type != AssetType.CASH
    ]
    return cash + non_cash


def _cash_last_order(portfolio_request: PortfolioRequest) -> list[str]:
    non_cash = [
        position.instrument.symbol
        for position in portfolio_request.positions
        if position.instrument.asset_type != AssetType.CASH
    ]
    return non_cash + _cash_symbols(portfolio_request)


def _positive_score_order(
    scores: Mapping[str, float],
    portfolio_request: PortfolioRequest,
) -> list[str]:
    non_cash = [
        position.instrument.symbol
        for position in portfolio_request.positions
        if position.instrument.asset_type != AssetType.CASH
    ]
    return sorted(non_cash, key=lambda symbol: (-scores[symbol], symbol))


def _negative_score_order(
    scores: Mapping[str, float],
    portfolio_request: PortfolioRequest,
) -> list[str]:
    non_cash = [
        position.instrument.symbol
        for position in portfolio_request.positions
        if position.instrument.asset_type != AssetType.CASH
    ]
    return sorted(non_cash, key=lambda symbol: (scores[symbol], symbol))


def _average_correlation(
    symbol: str,
    correlation_matrix: Mapping[str, Mapping[str, float | None]],
) -> float | None:
    row = correlation_matrix.get(symbol)
    if not row:
        return None
    values = [
        value
        for other_symbol, value in row.items()
        if other_symbol != symbol and _is_number(value)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _round_weight(value: float) -> float:
    if abs(value) < 0.0000005:
        return 0.0
    return round(value, 6)


def _format_weight(value: float) -> str:
    return f"{value:.2%}"


def _format_signed_weight(value: float) -> str:
    if abs(value) < 0.00005:
        value = 0.0
    return f"{value:+.2%}" if value > 0 else f"{value:.2%}"


def _escape_cell(value: str) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _dedupe(symbols: list[str]) -> list[str]:
    seen = set()
    result = []
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
    return result


def _is_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(numeric)


def _proposal_input_hash(*values: Any) -> str:
    """Return a stable hash of the inputs that produced a proposal."""

    def normalize(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "model_dump"):
            return normalize(value.model_dump(mode="json"))
        if hasattr(value, "to_dict"):
            return normalize(value.to_dict())
        if hasattr(value, "__dataclass_fields__"):
            return normalize(asdict(value))
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple, set)):
            return [normalize(item) for item in value]
        return str(value)

    payload = json.dumps(
        normalize(values), sort_keys=True, separators=(",", ":"), default=str
    )
    return sha256(payload.encode("utf-8")).hexdigest()
