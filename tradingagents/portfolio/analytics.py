"""Deterministic portfolio analytics.

This module is intentionally provider-agnostic. Callers can pass price history,
benchmark history, sector labels, and option Greeks from any data source; the
calculations here stay pure and offline-testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import pandas as pd

from tradingagents.portfolio.schemas import AssetType, PortfolioRequest


NumberSeries = Sequence[float] | pd.Series
GREEK_FIELDS = ("delta", "gamma", "theta", "vega", "rho")


@dataclass(frozen=True)
class OptionGreeks:
    """Option Greeks for one contract or position.

    Values are accepted as supplied by the provider. Portfolio aggregation in
    this module weight-adjusts them by current portfolio weight.
    """

    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None


@dataclass(frozen=True)
class ConstraintFlag:
    """A deterministic allocation or risk constraint observation."""

    scope: str
    kind: str
    message: str
    symbol: str | None = None
    current_weight: float | None = None
    limit: float | None = None


@dataclass(frozen=True)
class PortfolioAnalytics:
    """Computed analytics for a validated portfolio request."""

    total_market_value: float | None
    weights_by_symbol: dict[str, float]
    computed_weights_by_symbol: dict[str, float]
    market_values_by_symbol: dict[str, float | None]
    historical_returns_by_symbol: dict[str, list[float]]
    volatility_by_symbol: dict[str, float | None]
    correlation_matrix: dict[str, dict[str, float | None]]
    beta_by_symbol: dict[str, float | None]
    portfolio_volatility: float | None
    risk_contribution_by_symbol: dict[str, float | None]
    asset_type_exposure: dict[str, float]
    underlying_exposure: dict[str, float]
    sector_exposure: dict[str, float]
    delta_adjusted_exposure_by_symbol: dict[str, float | None]
    portfolio_delta_adjusted_exposure: float
    option_greeks_by_symbol: dict[str, dict[str, float | None]]
    aggregate_option_greeks: dict[str, float]
    constraint_flags: list[ConstraintFlag] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


def calculate_portfolio_analytics(
    portfolio_request: PortfolioRequest,
    *,
    historical_prices: Mapping[str, NumberSeries] | None = None,
    benchmark_prices: NumberSeries | None = None,
    option_greeks: Mapping[str, OptionGreeks | Mapping[str, float | None]] | None = None,
    sector_by_symbol: Mapping[str, str] | None = None,
    total_market_value: float | None = None,
    annualization_periods: int = 252,
) -> PortfolioAnalytics:
    """Compute deterministic analytics for a portfolio.

    Historical price inputs are optional. Metrics that require unavailable data
    are returned as ``None`` for the affected symbol instead of raising.
    """

    weights_by_symbol = {
        position.instrument.symbol: float(position.current_weight)
        for position in portfolio_request.positions
    }
    resolved_total_market_value = _resolve_total_market_value(
        portfolio_request,
        total_market_value=total_market_value,
    )
    market_values_by_symbol = _market_values_by_symbol(
        portfolio_request,
        total_market_value=resolved_total_market_value,
    )
    computed_weights_by_symbol = _computed_weights_by_symbol(
        weights_by_symbol,
        market_values_by_symbol,
        resolved_total_market_value,
    )

    returns_frame = _returns_frame(historical_prices or {})
    historical_returns_by_symbol = _historical_returns_by_symbol(
        returns_frame,
        weights_by_symbol,
    )
    volatility_by_symbol = _volatility_by_symbol(
        returns_frame,
        weights_by_symbol,
        annualization_periods,
    )
    correlation_matrix = _correlation_matrix(returns_frame, weights_by_symbol)
    beta_by_symbol = _beta_by_symbol(
        returns_frame,
        weights_by_symbol,
        benchmark_prices,
    )
    portfolio_volatility, risk_contribution_by_symbol = _portfolio_risk(
        returns_frame,
        weights_by_symbol,
        annualization_periods,
    )

    normalized_option_greeks = {
        symbol: _normalize_greeks(greeks)
        for symbol, greeks in (option_greeks or {}).items()
    }
    asset_type_exposure = _asset_type_exposure(portfolio_request)
    underlying_exposure = _underlying_exposure(portfolio_request)
    sector_exposure = _sector_exposure(portfolio_request, sector_by_symbol or {})
    (
        delta_adjusted_exposure_by_symbol,
        portfolio_delta_adjusted_exposure,
        option_greeks_by_symbol,
        aggregate_option_greeks,
    ) = _option_exposures(portfolio_request, normalized_option_greeks)

    constraint_flags = _constraint_flags(portfolio_request)

    return PortfolioAnalytics(
        total_market_value=resolved_total_market_value,
        weights_by_symbol=weights_by_symbol,
        computed_weights_by_symbol=computed_weights_by_symbol,
        market_values_by_symbol=market_values_by_symbol,
        historical_returns_by_symbol=historical_returns_by_symbol,
        volatility_by_symbol=volatility_by_symbol,
        correlation_matrix=correlation_matrix,
        beta_by_symbol=beta_by_symbol,
        portfolio_volatility=portfolio_volatility,
        risk_contribution_by_symbol=risk_contribution_by_symbol,
        asset_type_exposure=asset_type_exposure,
        underlying_exposure=underlying_exposure,
        sector_exposure=sector_exposure,
        delta_adjusted_exposure_by_symbol=delta_adjusted_exposure_by_symbol,
        portfolio_delta_adjusted_exposure=portfolio_delta_adjusted_exposure,
        option_greeks_by_symbol=option_greeks_by_symbol,
        aggregate_option_greeks=aggregate_option_greeks,
        constraint_flags=constraint_flags,
    )


def portfolio_analytics_to_dict(analytics: PortfolioAnalytics) -> dict[str, Any]:
    """Convert analytics to a JSON-serializable dictionary."""

    return analytics.to_dict()


def _resolve_total_market_value(
    portfolio_request: PortfolioRequest,
    *,
    total_market_value: float | None,
) -> float | None:
    if total_market_value is not None:
        return float(total_market_value)

    explicit_positions = [
        position
        for position in portfolio_request.positions
        if position.market_value is not None
    ]
    if not explicit_positions:
        return None

    explicit_value = sum(float(position.market_value) for position in explicit_positions)
    explicit_weight = sum(position.current_weight for position in explicit_positions)
    if explicit_weight <= 0:
        return None
    return explicit_value / explicit_weight


def _market_values_by_symbol(
    portfolio_request: PortfolioRequest,
    *,
    total_market_value: float | None,
) -> dict[str, float | None]:
    market_values: dict[str, float | None] = {}
    for position in portfolio_request.positions:
        symbol = position.instrument.symbol
        if position.market_value is not None:
            market_values[symbol] = float(position.market_value)
        elif total_market_value is not None:
            market_values[symbol] = float(position.current_weight * total_market_value)
        else:
            market_values[symbol] = None
    return market_values


def _computed_weights_by_symbol(
    weights_by_symbol: Mapping[str, float],
    market_values_by_symbol: Mapping[str, float | None],
    total_market_value: float | None,
) -> dict[str, float]:
    if total_market_value is None or total_market_value <= 0:
        return dict(weights_by_symbol)

    computed = {}
    for symbol, current_weight in weights_by_symbol.items():
        market_value = market_values_by_symbol.get(symbol)
        if market_value is None:
            computed[symbol] = current_weight
        else:
            computed[symbol] = float(market_value / total_market_value)
    return computed


def _returns_frame(historical_prices: Mapping[str, NumberSeries]) -> pd.DataFrame:
    price_series: dict[str, pd.Series] = {}
    for symbol, values in historical_prices.items():
        series = _numeric_series(values)
        if len(series) >= 2:
            price_series[symbol.upper()] = series
    if not price_series:
        return pd.DataFrame()
    prices = pd.DataFrame(price_series)
    return prices.pct_change(fill_method=None).replace([float("inf"), float("-inf")], pd.NA)


def _numeric_series(values: NumberSeries) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(list(values))
    return pd.to_numeric(series, errors="coerce").dropna()


def _historical_returns_by_symbol(
    returns_frame: pd.DataFrame,
    weights_by_symbol: Mapping[str, float],
) -> dict[str, list[float]]:
    returns_by_symbol: dict[str, list[float]] = {}
    for symbol in weights_by_symbol:
        if symbol not in returns_frame:
            returns_by_symbol[symbol] = []
            continue
        returns_by_symbol[symbol] = [
            float(value)
            for value in returns_frame[symbol].dropna().tolist()
            if _is_real_number(value)
        ]
    return returns_by_symbol


def _volatility_by_symbol(
    returns_frame: pd.DataFrame,
    weights_by_symbol: Mapping[str, float],
    annualization_periods: int,
) -> dict[str, float | None]:
    volatility: dict[str, float | None] = {}
    for symbol in weights_by_symbol:
        if symbol not in returns_frame or returns_frame[symbol].dropna().shape[0] < 2:
            volatility[symbol] = None
            continue
        value = returns_frame[symbol].std(skipna=True, ddof=1) * sqrt(annualization_periods)
        volatility[symbol] = _clean_float(value)
    return volatility


def _correlation_matrix(
    returns_frame: pd.DataFrame,
    weights_by_symbol: Mapping[str, float],
) -> dict[str, dict[str, float | None]]:
    symbols = [symbol for symbol in weights_by_symbol if symbol in returns_frame]
    if not symbols:
        return {}

    correlation = returns_frame[symbols].corr()
    matrix: dict[str, dict[str, float | None]] = {}
    for row_symbol in symbols:
        matrix[row_symbol] = {}
        for column_symbol in symbols:
            matrix[row_symbol][column_symbol] = _clean_float(
                correlation.loc[row_symbol, column_symbol]
            )
    return matrix


def _beta_by_symbol(
    returns_frame: pd.DataFrame,
    weights_by_symbol: Mapping[str, float],
    benchmark_prices: NumberSeries | None,
) -> dict[str, float | None]:
    beta = {symbol: None for symbol in weights_by_symbol}
    if benchmark_prices is None or returns_frame.empty:
        return beta

    benchmark_returns = _numeric_series(benchmark_prices).pct_change(fill_method=None).dropna()
    if benchmark_returns.shape[0] < 2:
        return beta
    benchmark_variance = benchmark_returns.var(ddof=1)
    if not _is_real_number(benchmark_variance) or benchmark_variance == 0:
        return beta

    for symbol in weights_by_symbol:
        if symbol not in returns_frame:
            continue
        aligned = pd.concat(
            [returns_frame[symbol], benchmark_returns],
            axis=1,
            join="inner",
        ).dropna()
        if aligned.shape[0] < 2:
            continue
        covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
        beta[symbol] = _clean_float(covariance / benchmark_variance)
    return beta


def _portfolio_risk(
    returns_frame: pd.DataFrame,
    weights_by_symbol: Mapping[str, float],
    annualization_periods: int,
) -> tuple[float | None, dict[str, float | None]]:
    risk_contribution = {symbol: None for symbol in weights_by_symbol}
    symbols = [
        symbol
        for symbol in weights_by_symbol
        if symbol in returns_frame and returns_frame[symbol].dropna().shape[0] >= 2
    ]
    if not symbols:
        return None, risk_contribution

    aligned_returns = returns_frame[symbols].dropna()
    if aligned_returns.shape[0] < 2:
        return None, risk_contribution

    covariance = aligned_returns.cov() * annualization_periods
    weights = pd.Series({symbol: weights_by_symbol[symbol] for symbol in symbols})
    variance = float(weights.T @ covariance @ weights)
    if variance < 0 and abs(variance) < 1e-12:
        variance = 0.0
    if variance <= 0:
        return 0.0, {symbol: 0.0 for symbol in weights_by_symbol}

    portfolio_volatility = sqrt(variance)
    covariance_weight = covariance @ weights
    for symbol in symbols:
        contribution = weights[symbol] * covariance_weight[symbol] / variance
        risk_contribution[symbol] = _clean_float(contribution)
    for symbol in weights_by_symbol:
        if risk_contribution[symbol] is None:
            risk_contribution[symbol] = 0.0
    return portfolio_volatility, risk_contribution


def _asset_type_exposure(portfolio_request: PortfolioRequest) -> dict[str, float]:
    exposure = {asset_type.value: 0.0 for asset_type in AssetType}
    for position in portfolio_request.positions:
        exposure[position.instrument.asset_type.value] += float(position.current_weight)
    return {key: value for key, value in exposure.items() if value != 0}


def _underlying_exposure(portfolio_request: PortfolioRequest) -> dict[str, float]:
    exposure: dict[str, float] = {}
    for position in portfolio_request.positions:
        instrument = position.instrument
        if instrument.asset_type == AssetType.OPTION:
            exposure_symbol = instrument.underlying
        else:
            exposure_symbol = instrument.symbol
        if exposure_symbol is None:
            continue
        exposure[exposure_symbol] = exposure.get(exposure_symbol, 0.0) + float(
            position.current_weight
        )
    return exposure


def _sector_exposure(
    portfolio_request: PortfolioRequest,
    sector_by_symbol: Mapping[str, str],
) -> dict[str, float]:
    exposure: dict[str, float] = {}
    for position in portfolio_request.positions:
        instrument = position.instrument
        lookup_symbol = (
            instrument.underlying
            if instrument.asset_type == AssetType.OPTION
            else instrument.symbol
        )
        sector = sector_by_symbol.get(lookup_symbol or "")
        if sector is None:
            continue
        exposure[sector] = exposure.get(sector, 0.0) + float(position.current_weight)
    return exposure


def _option_exposures(
    portfolio_request: PortfolioRequest,
    option_greeks: Mapping[str, OptionGreeks],
) -> tuple[
    dict[str, float | None],
    float,
    dict[str, dict[str, float | None]],
    dict[str, float],
]:
    delta_adjusted: dict[str, float | None] = {}
    option_greeks_by_symbol: dict[str, dict[str, float | None]] = {}
    aggregate_greeks = {field_name: 0.0 for field_name in GREEK_FIELDS}
    portfolio_delta_adjusted = 0.0

    for position in portfolio_request.positions:
        instrument = position.instrument
        symbol = instrument.symbol
        weight = float(position.current_weight)

        if instrument.asset_type == AssetType.STOCK:
            delta_adjusted[symbol] = weight
            portfolio_delta_adjusted += weight
            continue

        if instrument.asset_type == AssetType.CASH:
            delta_adjusted[symbol] = 0.0
            continue

        greeks = option_greeks.get(symbol)
        greek_values = _greeks_to_dict(greeks)
        option_greeks_by_symbol[symbol] = greek_values

        delta = greek_values["delta"]
        if delta is None:
            delta_adjusted[symbol] = None
        else:
            delta_adjusted[symbol] = weight * delta
            portfolio_delta_adjusted += delta_adjusted[symbol]

        for field_name, value in greek_values.items():
            if value is not None:
                aggregate_greeks[field_name] += weight * value

    return (
        delta_adjusted,
        portfolio_delta_adjusted,
        option_greeks_by_symbol,
        aggregate_greeks,
    )


def _constraint_flags(portfolio_request: PortfolioRequest) -> list[ConstraintFlag]:
    flags: list[ConstraintFlag] = []
    constraints = portfolio_request.constraints

    for position in portfolio_request.positions:
        symbol = position.instrument.symbol
        weight = float(position.current_weight)
        if (
            position.target_min_weight is not None
            and weight < position.target_min_weight
        ):
            flags.append(
                ConstraintFlag(
                    scope="position",
                    kind="underweight",
                    symbol=symbol,
                    current_weight=weight,
                    limit=float(position.target_min_weight),
                    message=(
                        f"{symbol} is below target_min_weight "
                        f"{position.target_min_weight:.4f}"
                    ),
                )
            )
        if (
            position.target_max_weight is not None
            and weight > position.target_max_weight
        ):
            flags.append(
                ConstraintFlag(
                    scope="position",
                    kind="overweight",
                    symbol=symbol,
                    current_weight=weight,
                    limit=float(position.target_max_weight),
                    message=(
                        f"{symbol} exceeds target_max_weight "
                        f"{position.target_max_weight:.4f}"
                    ),
                )
            )
        if (
            constraints.max_single_position_weight is not None
            and weight > constraints.max_single_position_weight
        ):
            flags.append(
                ConstraintFlag(
                    scope="portfolio",
                    kind="max_single_position_weight",
                    symbol=symbol,
                    current_weight=weight,
                    limit=float(constraints.max_single_position_weight),
                    message=(
                        f"{symbol} exceeds max_single_position_weight "
                        f"{constraints.max_single_position_weight:.4f}"
                    ),
                )
            )

    options_weight = sum(
        position.current_weight
        for position in portfolio_request.positions
        if position.instrument.asset_type == AssetType.OPTION
    )
    if (
        constraints.max_options_weight is not None
        and options_weight > constraints.max_options_weight
    ):
        flags.append(
            ConstraintFlag(
                scope="portfolio",
                kind="max_options_weight",
                current_weight=float(options_weight),
                limit=float(constraints.max_options_weight),
                message=(
                    "options exposure exceeds max_options_weight "
                    f"{constraints.max_options_weight:.4f}"
                ),
            )
        )

    cash_weight = sum(
        position.current_weight
        for position in portfolio_request.positions
        if position.instrument.asset_type == AssetType.CASH
    )
    if constraints.min_cash_weight is not None and cash_weight < constraints.min_cash_weight:
        flags.append(
            ConstraintFlag(
                scope="portfolio",
                kind="min_cash_weight",
                current_weight=float(cash_weight),
                limit=float(constraints.min_cash_weight),
                message=(
                    "cash exposure is below min_cash_weight "
                    f"{constraints.min_cash_weight:.4f}"
                ),
            )
        )
    if constraints.max_cash_weight is not None and cash_weight > constraints.max_cash_weight:
        flags.append(
            ConstraintFlag(
                scope="portfolio",
                kind="max_cash_weight",
                current_weight=float(cash_weight),
                limit=float(constraints.max_cash_weight),
                message=(
                    "cash exposure exceeds max_cash_weight "
                    f"{constraints.max_cash_weight:.4f}"
                ),
            )
        )

    return flags


def _normalize_greeks(greeks: OptionGreeks | Mapping[str, float | None]) -> OptionGreeks:
    if isinstance(greeks, OptionGreeks):
        return greeks
    return OptionGreeks(
        **{
            field_name: _clean_float(greeks.get(field_name))
            for field_name in GREEK_FIELDS
        }
    )


def _greeks_to_dict(greeks: OptionGreeks | None) -> dict[str, float | None]:
    if greeks is None:
        return {field_name: None for field_name in GREEK_FIELDS}
    return {
        field_name: _clean_float(getattr(greeks, field_name))
        for field_name in GREEK_FIELDS
    }


def _clean_float(value: Any) -> float | None:
    if not _is_real_number(value):
        return None
    return float(value)


def _is_real_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(numeric)
