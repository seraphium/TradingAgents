"""Deterministic portfolio rebalancing.

The engine in this module converts single-instrument ratings and portfolio
analytics into a first-pass target allocation. LLM agents may critique or
explain this proposal later, but the numeric proposal itself is deterministic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Literal, Mapping

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.portfolio.analytics import PortfolioAnalytics
from tradingagents.portfolio.schemas import AssetType, PortfolioRequest


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
    constraints_applied: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RebalanceProposal:
    """A deterministic portfolio-level target allocation proposal."""

    portfolio_action: str
    component_proposals: list[RebalanceComponent]
    target_weights_by_symbol: dict[str, float]
    score_by_symbol: dict[str, float]
    risk_notes: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_rebalance_proposal(
    portfolio_request: PortfolioRequest,
    analytics: PortfolioAnalytics,
    *,
    ratings_by_symbol: Mapping[str, str] | None = None,
    liquidity_by_symbol: Mapping[str, float] | None = None,
    optimizer: Literal["heuristic", "mean_variance"] | None = None,
    score_step: float = 0.05,
    change_tolerance: float = 0.005,
) -> RebalanceProposal:
    """Generate deterministic target weights and action labels.

    Ratings can be canonical values or prose containing a parseable rating.
    Optional liquidity scores are expected on a 0-1 scale where lower values
    indicate weaker liquidity.
    """

    ratings = ratings_by_symbol or {}
    liquidity = liquidity_by_symbol or {}
    bounds = _component_bounds(portfolio_request)
    positions_by_symbol = {
        position.instrument.symbol: position for position in portfolio_request.positions
    }
    current_weights = {
        symbol: float(position.current_weight)
        for symbol, position in positions_by_symbol.items()
    }
    scores = {
        symbol: _score_for_symbol(symbol, ratings)
        for symbol in positions_by_symbol
    }
    risk_adjustments, risk_notes = _risk_adjustments(
        analytics,
        positions_by_symbol,
        liquidity,
    )

    optimizer_mode = _optimizer_mode(portfolio_request, optimizer)
    diagnostics: dict[str, Any] = {
        "score_step": score_step,
        "change_tolerance": change_tolerance,
        "optimizer": optimizer_mode,
    }
    targets = dict(current_weights)
    constraints_applied = {symbol: [] for symbol in positions_by_symbol}
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
            adjustment = (scores[symbol] + risk_adjustments.get(symbol, 0.0)) * score_step
            unclamped = targets[symbol] + adjustment
            targets[symbol] = _clamp(unclamped, *bounds[symbol])
            if targets[symbol] != unclamped:
                constraints_applied[symbol].append("component_bounds")

    targets = _apply_option_limit(portfolio_request, targets, bounds, constraints_applied)
    targets = _apply_cash_budget(portfolio_request, analytics, targets, bounds, constraints_applied)
    targets = _normalize_targets(targets, bounds, scores, portfolio_request, constraints_applied)

    components: list[RebalanceComponent] = []
    for symbol, position in positions_by_symbol.items():
        current_weight = current_weights[symbol]
        target_weight = _round_weight(targets[symbol])
        weight_change = _round_weight(target_weight - current_weight)
        action = _action_for_change(current_weight, target_weight, change_tolerance)
        rating = _rating_for_symbol(symbol, ratings)
        rationale_parts = [f"{rating} rating score {scores[symbol]:+.2f}."]
        if optimizer_mode == "mean_variance":
            rationale_parts.append("Target set by mean-variance optimizer.")
        if risk_adjustments.get(symbol):
            rationale_parts.append(f"Risk adjustment {risk_adjustments[symbol]:+.2f}.")
        if constraints_applied[symbol]:
            rationale_parts.append("Applied constraints: " + ", ".join(constraints_applied[symbol]) + ".")
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

    return RebalanceProposal(
        portfolio_action=portfolio_action,
        component_proposals=components,
        target_weights_by_symbol=target_weights_by_symbol,
        score_by_symbol=scores,
        risk_notes=risk_notes,
        diagnostics={**diagnostics, "target_sum": sum(target_weights_by_symbol.values())},
    )


def extract_ratings_from_portfolio_result(portfolio_result: Mapping[str, Any]) -> dict[str, str]:
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


def render_rebalance_proposal(proposal: RebalanceProposal) -> str:
    """Render a deterministic rebalance proposal to Markdown."""

    parts = [
        f"**Portfolio Action**: {proposal.portfolio_action}",
        "",
        "| Symbol | Current Weight | Target Weight | Change | Action | Rating | Score | Rationale |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
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


def rebalance_proposal_to_dict(proposal: RebalanceProposal) -> dict[str, Any]:
    return proposal.to_dict()


def _rating_for_symbol(symbol: str, ratings_by_symbol: Mapping[str, str]) -> str:
    return parse_rating(str(ratings_by_symbol.get(symbol, "Hold")))


def _score_for_symbol(symbol: str, ratings_by_symbol: Mapping[str, str]) -> float:
    return RATING_SCORES[_rating_for_symbol(symbol, ratings_by_symbol)]


def _component_bounds(portfolio_request: PortfolioRequest) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    constraints = portfolio_request.constraints
    for position in portfolio_request.positions:
        symbol = position.instrument.symbol
        minimum = position.target_min_weight if position.target_min_weight is not None else 0.0
        maximum = position.target_max_weight if position.target_max_weight is not None else 1.0
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
    positions_by_symbol: Mapping[str, Any],
    liquidity_by_symbol: Mapping[str, float],
) -> tuple[dict[str, float], list[str]]:
    adjustments = {symbol: 0.0 for symbol in positions_by_symbol}
    notes: list[str] = []
    custom_thresholds = {
        "high_volatility_threshold": 0.45,
        "high_correlation_threshold": 0.80,
        "max_underlying_exposure": 0.40,
        "min_liquidity_score": 0.35,
    }

    for symbol in positions_by_symbol:
        volatility = analytics.volatility_by_symbol.get(symbol)
        if _is_number(volatility) and volatility > custom_thresholds["high_volatility_threshold"]:
            adjustments[symbol] -= 0.5
            notes.append(f"{symbol} volatility is elevated at {volatility:.2%}.")

        average_correlation = _average_correlation(symbol, analytics.correlation_matrix)
        if (
            average_correlation is not None
            and average_correlation > custom_thresholds["high_correlation_threshold"]
        ):
            adjustments[symbol] -= 0.25
            notes.append(f"{symbol} has high average correlation at {average_correlation:.2f}.")

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

    return adjustments, notes


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
    return _distribute_delta(targets, freed, bounds, _cash_first_order(portfolio_request), constraints_applied, "max_options_weight")


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
    return _distribute_delta(targets, removed, bounds, cash_symbols, constraints_applied, "risk_budget")


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
            expected.append((scores[symbol] + risk_adjustments.get(symbol, 0.0)) * scale)
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
                else default_option_volatility
                if _position_asset_type(portfolio_request, symbol) == AssetType.OPTION
                else default_volatility
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
            row.append(float(correlation) * volatilities[left_index] * volatilities[right_index])
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
        risk_gradient = 2.0 * risk_aversion * sum(
            covariance_value * weight
            for covariance_value, weight in zip(row, weights)
        )
        return_gradient = -expected_returns[row_index]
        turnover_gradient = 2.0 * turnover_penalty * (
            weights[row_index] - current_weights[row_index]
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
            covariance_value * weight
            for covariance_value, weight in zip(row, weights)
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
        order = _cash_first_order(portfolio_request) + _positive_score_order(scores, portfolio_request)
        _distribute_delta(targets, difference, bounds, order, constraints_applied, "normalization")
    else:
        order = _cash_symbols(portfolio_request) + _negative_score_order(scores, portfolio_request)
        _remove_delta(targets, abs(difference), bounds, order, constraints_applied, "normalization")
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
    if all(abs(component.weight_change) <= change_tolerance for component in components):
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


def _action_for_change(current_weight: float, target_weight: float, tolerance: float) -> str:
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
