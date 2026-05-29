"""Best-effort market inputs for deterministic portfolio analytics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Any

import pandas as pd

from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio.schemas import AssetType, PortfolioRequest


PRICE_COLUMNS = (
    "adjusted_close",
    "Adj Close",
    "adj close",
    "close",
    "Close",
)


@dataclass(frozen=True)
class PortfolioAnalyticsInputs:
    """Provider-derived inputs used by ``calculate_portfolio_analytics``."""

    historical_prices: dict[str, list[float]] = field(default_factory=dict)
    benchmark_prices: list[float] | None = None
    sector_by_symbol: dict[str, str] = field(default_factory=dict)
    benchmark_symbol: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_portfolio_analytics_inputs(
    portfolio_request: PortfolioRequest,
    *,
    lookback_days: int = 365,
    config: dict[str, Any] | None = None,
) -> PortfolioAnalyticsInputs:
    """Fetch optional analytics inputs for a portfolio.

    Price, benchmark, and sector data improve correlation/risk/sector checks,
    but provider gaps should not block the portfolio analysis. This function
    records warnings and returns whatever could be collected.
    """

    effective_config = config or DEFAULT_CONFIG
    trade_date = _coerce_date(portfolio_request.trade_date)
    start_date = trade_date - timedelta(days=lookback_days)
    end = trade_date.isoformat()
    start = start_date.isoformat()

    historical_prices: dict[str, list[float]] = {}
    sector_by_symbol: dict[str, str] = {}
    warnings: list[str] = []
    price_cache: dict[str, list[float]] = {}

    symbols = _analysis_symbols(portfolio_request)
    for position_symbol, lookup_symbol in symbols:
        prices = price_cache.get(lookup_symbol)
        if prices is None:
            prices = _fetch_price_history(
                lookup_symbol,
                start_date=start,
                end_date=end,
                warnings=warnings,
                label="holding",
            )
            if prices:
                price_cache[lookup_symbol] = prices
        if prices:
            historical_prices[position_symbol] = prices

    for lookup_symbol in sorted({lookup_symbol for _, lookup_symbol in symbols}):
        sector = _fetch_sector_label(lookup_symbol, trade_date=end, warnings=warnings)
        if sector:
            sector_by_symbol[lookup_symbol] = sector

    benchmark_symbol = _resolve_benchmark_symbol(portfolio_request, effective_config)
    benchmark_prices = None
    if benchmark_symbol:
        benchmark_prices = _fetch_price_history(
            benchmark_symbol,
            start_date=start,
            end_date=end,
            warnings=warnings,
            label="benchmark",
        )

    return PortfolioAnalyticsInputs(
        historical_prices=historical_prices,
        benchmark_prices=benchmark_prices,
        sector_by_symbol=sector_by_symbol,
        benchmark_symbol=benchmark_symbol,
        warnings=warnings,
    )


def _analysis_symbols(portfolio_request: PortfolioRequest) -> list[tuple[str, str]]:
    symbols: list[tuple[str, str]] = []
    for position in portfolio_request.positions:
        instrument = position.instrument
        if instrument.asset_type == AssetType.CASH:
            continue
        lookup_symbol = (
            instrument.underlying
            if instrument.asset_type == AssetType.OPTION
            else instrument.symbol
        )
        if lookup_symbol:
            symbols.append((instrument.symbol, lookup_symbol))
    return symbols


def _fetch_price_history(
    symbol: str,
    *,
    start_date: str,
    end_date: str,
    warnings: list[str],
    label: str,
) -> list[float] | None:
    try:
        response = route_to_vendor("get_stock_data", symbol, start_date, end_date)
    except Exception as exc:
        warnings.append(f"{label} price history unavailable for {symbol}: {exc}")
        return None

    prices = _parse_price_history(response)
    if len(prices) < 2:
        warnings.append(f"{label} price history unavailable for {symbol}: insufficient data")
        return None
    return prices


def _parse_price_history(response: Any) -> list[float]:
    if isinstance(response, pd.DataFrame):
        return _prices_from_frame(response)
    if not isinstance(response, str):
        return []

    stripped = response.strip()
    if not stripped:
        return []
    if stripped.startswith("{"):
        return _prices_from_json(stripped)

    rows = [
        line
        for line in stripped.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        return []

    try:
        frame = pd.read_csv(StringIO("\n".join(rows)))
    except Exception:
        return []
    return _prices_from_frame(frame)


def _prices_from_json(payload: str) -> list[float]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    if any(key in data for key in ("Information", "Error Message", "Note")):
        return []

    series = None
    for key, value in data.items():
        if "Time Series" in key and isinstance(value, dict):
            series = value
            break
    if not isinstance(series, dict):
        return []

    prices: list[tuple[str, float]] = []
    for timestamp, values in series.items():
        if not isinstance(values, dict):
            continue
        close = (
            values.get("5. adjusted close")
            or values.get("4. close")
            or values.get("adjusted_close")
            or values.get("close")
        )
        try:
            prices.append((timestamp, float(close)))
        except (TypeError, ValueError):
            continue
    return [price for _, price in sorted(prices)]


def _prices_from_frame(frame: pd.DataFrame) -> list[float]:
    if frame.empty:
        return []
    for column in PRICE_COLUMNS:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            return [float(value) for value in values.tolist()]
    lower_columns = {str(column).lower(): column for column in frame.columns}
    for column in PRICE_COLUMNS:
        matched = lower_columns.get(column.lower())
        if matched is not None:
            values = pd.to_numeric(frame[matched], errors="coerce").dropna()
            return [float(value) for value in values.tolist()]
    return []


def _fetch_sector_label(
    symbol: str,
    *,
    trade_date: str,
    warnings: list[str],
) -> str | None:
    try:
        response = route_to_vendor("get_fundamentals", symbol, trade_date)
    except Exception as exc:
        warnings.append(f"sector label unavailable for {symbol}: {exc}")
        return None

    sector, industry = _parse_sector_and_industry(response)
    if sector and industry:
        return f"{sector} / {industry}"
    if sector:
        return sector
    if industry:
        return industry
    return None


def _parse_sector_and_industry(response: Any) -> tuple[str | None, str | None]:
    if isinstance(response, str):
        stripped = response.strip()
        if not stripped:
            return None, None
        if stripped.startswith("{"):
            try:
                response = json.loads(stripped)
            except json.JSONDecodeError:
                return _parse_sector_from_text(stripped)
        else:
            return _parse_sector_from_text(stripped)

    if isinstance(response, dict):
        sector = _clean_label(response.get("Sector") or response.get("sector"))
        industry = _clean_label(response.get("Industry") or response.get("industry"))
        return sector, industry

    return None, None


def _parse_sector_from_text(text: str) -> tuple[str | None, str | None]:
    sector = None
    industry = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("sector:"):
            sector = _clean_label(line.split(":", 1)[1])
        elif line.lower().startswith("industry:"):
            industry = _clean_label(line.split(":", 1)[1])
    return sector, industry


def _clean_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    if not label or label.lower() in {"none", "nan", "null"}:
        return None
    return label


def _resolve_benchmark_symbol(
    portfolio_request: PortfolioRequest,
    config: dict[str, Any],
) -> str | None:
    explicit = config.get("benchmark_ticker")
    if explicit:
        return str(explicit).strip().upper()

    benchmark_map = config.get("benchmark_map") or {}
    first_symbol = next(
        (
            lookup_symbol
            for _, lookup_symbol in _analysis_symbols(portfolio_request)
            if lookup_symbol
        ),
        None,
    )
    if first_symbol is None:
        return benchmark_map.get("", "SPY")

    symbol_upper = first_symbol.upper()
    for suffix, benchmark in benchmark_map.items():
        if suffix and symbol_upper.endswith(str(suffix).upper()):
            return str(benchmark).strip().upper()
    default = benchmark_map.get("", "SPY")
    return str(default).strip().upper() if default else None


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
