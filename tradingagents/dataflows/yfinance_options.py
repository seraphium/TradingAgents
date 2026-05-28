"""Options data adapter backed by yfinance.

The functions in this module return JSON-serializable dictionaries so they can
be used both by LangChain tools and deterministic portfolio analytics.
"""

from __future__ import annotations

from datetime import date, datetime
import json
from math import erf, exp, isfinite, log, pi, sqrt
from typing import Annotated, Any

import pandas as pd
import yfinance as yf

from .stockstats_utils import yf_retry


GREEK_FIELDS = ("delta", "gamma", "theta", "vega", "rho")


def get_options_chain(
    underlying: Annotated[str, "underlying ticker symbol"],
    expiry: Annotated[str | None, "option expiry in YYYY-MM-DD format"] = None,
    curr_date: Annotated[str | None, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Return the options chain for an underlying as compact JSON."""

    chain = retrieve_options_chain(underlying, expiry=expiry, curr_date=curr_date)
    return json.dumps(chain, indent=2, default=str)


def get_option_contract(
    underlying: Annotated[str, "underlying ticker symbol"],
    expiry: Annotated[str, "option expiry in YYYY-MM-DD format"],
    strike: Annotated[float, "option strike price"],
    right: Annotated[str, "C/call or P/put"],
    curr_date: Annotated[str | None, "current date in YYYY-MM-DD format"] = None,
    contract_symbol: Annotated[str | None, "optional vendor contract symbol"] = None,
) -> str:
    """Return one option contract lookup as compact JSON."""

    contract = lookup_option_contract(
        underlying,
        expiry,
        strike,
        right,
        curr_date=curr_date,
        contract_symbol=contract_symbol,
    )
    return json.dumps(contract, indent=2, default=str)


def get_option_greeks(
    underlying: Annotated[str, "underlying ticker symbol"],
    expiry: Annotated[str, "option expiry in YYYY-MM-DD format"],
    strike: Annotated[float, "option strike price"],
    right: Annotated[str, "C/call or P/put"],
    curr_date: Annotated[str | None, "current date in YYYY-MM-DD format"] = None,
    contract_symbol: Annotated[str | None, "optional vendor contract symbol"] = None,
    underlying_price: Annotated[float | None, "optional current underlying price"] = None,
    implied_volatility: Annotated[float | None, "optional implied volatility decimal"] = None,
    risk_free_rate: Annotated[float, "annualized risk-free rate as decimal"] = 0.05,
    dividend_yield: Annotated[float, "annualized dividend yield as decimal"] = 0.0,
) -> str:
    """Return provider Greeks when available, with Black-Scholes fallback."""

    greeks = retrieve_option_greeks(
        underlying,
        expiry,
        strike,
        right,
        curr_date=curr_date,
        contract_symbol=contract_symbol,
        underlying_price=underlying_price,
        implied_volatility=implied_volatility,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    return json.dumps(greeks, indent=2, default=str)


def retrieve_options_chain(
    underlying: str,
    *,
    expiry: str | None = None,
    curr_date: str | None = None,
) -> dict[str, Any]:
    """Fetch one yfinance options chain and enrich rows with deterministic fields."""

    symbol = _clean_symbol(underlying)
    ticker = yf.Ticker(symbol)
    expirations = list(yf_retry(lambda: ticker.options) or [])
    selected_expiry = _select_expiry(expirations, expiry)
    option_chain = yf_retry(lambda: ticker.option_chain(selected_expiry))
    underlying_price = _resolve_underlying_price(ticker)
    as_of_date = _parse_optional_date(curr_date) or date.today()

    return {
        "underlying": symbol,
        "underlying_price": underlying_price,
        "expiry": selected_expiry,
        "available_expirations": expirations,
        "days_to_expiry": days_to_expiry(selected_expiry, as_of_date),
        "calls": _records_from_chain(
            option_chain.calls,
            "C",
            selected_expiry,
            as_of_date,
            underlying_price,
        ),
        "puts": _records_from_chain(
            option_chain.puts,
            "P",
            selected_expiry,
            as_of_date,
            underlying_price,
        ),
    }


def lookup_option_contract(
    underlying: str,
    expiry: str,
    strike: float,
    right: str,
    *,
    curr_date: str | None = None,
    contract_symbol: str | None = None,
) -> dict[str, Any]:
    """Find a single contract by symbol or nearest strike in the option chain."""

    normalized_right = normalize_option_right(right)
    chain = retrieve_options_chain(underlying, expiry=expiry, curr_date=curr_date)
    rows = chain["calls"] if normalized_right == "C" else chain["puts"]
    if not rows:
        raise ValueError(f"No {normalized_right} contracts found for {underlying} {expiry}")

    normalized_contract_symbol = (
        _clean_symbol(contract_symbol) if contract_symbol else None
    )
    if normalized_contract_symbol:
        matches = [
            row
            for row in rows
            if _clean_symbol(str(row.get("contractSymbol", "")))
            == normalized_contract_symbol
        ]
        if matches:
            return _contract_payload(chain, matches[0], normalized_right, strike)

    target_strike = float(strike)
    match = min(rows, key=lambda row: abs(float(row.get("strike", 0)) - target_strike))
    return _contract_payload(chain, match, normalized_right, target_strike)


def retrieve_option_greeks(
    underlying: str,
    expiry: str,
    strike: float,
    right: str,
    *,
    curr_date: str | None = None,
    contract_symbol: str | None = None,
    underlying_price: float | None = None,
    implied_volatility: float | None = None,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> dict[str, Any]:
    """Return complete option Greek fields, using fallback math when needed."""

    contract: dict[str, Any] | None = None
    if underlying_price is None or implied_volatility is None or contract_symbol:
        contract = lookup_option_contract(
            underlying,
            expiry,
            strike,
            right,
            curr_date=curr_date,
            contract_symbol=contract_symbol,
        )

    price = _first_number(
        underlying_price,
        (contract or {}).get("underlying_price"),
    )
    iv = _first_number(
        implied_volatility,
        (contract or {}).get("impliedVolatility"),
    )
    as_of_date = _parse_optional_date(curr_date) or date.today()
    dte = days_to_expiry(expiry, as_of_date)
    fallback = black_scholes_greeks(
        underlying_price=price,
        strike=float(strike),
        days_to_expiry=dte,
        implied_volatility=iv,
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )

    greeks: dict[str, float | None] = {}
    sources: dict[str, str] = {}
    provider_fields = contract or {}
    for field in GREEK_FIELDS:
        provider_value = _to_float(provider_fields.get(field))
        if provider_value is not None:
            greeks[field] = provider_value
            sources[field] = "provider"
        elif fallback.get(field) is not None:
            greeks[field] = fallback[field]
            sources[field] = "fallback_black_scholes"
        else:
            greeks[field] = None
            sources[field] = "unavailable"

    return {
        "symbol": (contract or {}).get("symbol") or contract_symbol,
        "underlying": _clean_symbol(underlying),
        "underlying_price": price,
        "expiry": _parse_date(expiry).isoformat(),
        "strike": float(strike),
        "right": normalize_option_right(right),
        "implied_volatility": iv,
        "days_to_expiry": dte,
        "moneyness": calculate_moneyness(price, float(strike)),
        "greeks": greeks,
        "greek_sources": sources,
    }


def black_scholes_greeks(
    *,
    underlying_price: float | None,
    strike: float,
    days_to_expiry: int,
    implied_volatility: float | None,
    right: str,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> dict[str, float | None]:
    """Compute Black-Scholes Greeks for fallback use.

    Vega and rho are returned per one percentage-point change; theta is per day.
    """

    price = _to_float(underlying_price)
    volatility = _to_float(implied_volatility)
    if (
        price is None
        or price <= 0
        or strike <= 0
        or volatility is None
        or volatility <= 0
        or days_to_expiry <= 0
    ):
        return {field: None for field in GREEK_FIELDS}

    option_right = normalize_option_right(right)
    time_to_expiry = days_to_expiry / 365.0
    sqrt_time = sqrt(time_to_expiry)
    d1 = (
        log(price / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt_time)
    d2 = d1 - volatility * sqrt_time
    discount_dividend = exp(-dividend_yield * time_to_expiry)
    discount_rate = exp(-risk_free_rate * time_to_expiry)

    if option_right == "C":
        delta = discount_dividend * _norm_cdf(d1)
        theta = (
            -price * discount_dividend * _norm_pdf(d1) * volatility / (2 * sqrt_time)
            - risk_free_rate * strike * discount_rate * _norm_cdf(d2)
            + dividend_yield * price * discount_dividend * _norm_cdf(d1)
        ) / 365.0
        rho = strike * time_to_expiry * discount_rate * _norm_cdf(d2) / 100.0
    else:
        delta = discount_dividend * (_norm_cdf(d1) - 1)
        theta = (
            -price * discount_dividend * _norm_pdf(d1) * volatility / (2 * sqrt_time)
            + risk_free_rate * strike * discount_rate * _norm_cdf(-d2)
            - dividend_yield * price * discount_dividend * _norm_cdf(-d1)
        ) / 365.0
        rho = -strike * time_to_expiry * discount_rate * _norm_cdf(-d2) / 100.0

    gamma = discount_dividend * _norm_pdf(d1) / (price * volatility * sqrt_time)
    vega = price * discount_dividend * _norm_pdf(d1) * sqrt_time / 100.0
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }


def days_to_expiry(expiry: str | date, curr_date: str | date | None = None) -> int:
    """Return non-negative calendar days between current date and expiry."""

    expiry_date = _parse_date(expiry)
    as_of_date = _parse_optional_date(curr_date) or date.today()
    return max((expiry_date - as_of_date).days, 0)


def calculate_moneyness(
    underlying_price: float | None,
    strike: float,
) -> float | None:
    """Return underlying price divided by strike when both values are usable."""

    price = _to_float(underlying_price)
    if price is None or strike <= 0:
        return None
    return price / float(strike)


def normalize_option_right(right: str) -> str:
    normalized = str(right).strip().upper()
    if normalized == "CALL":
        return "C"
    if normalized == "PUT":
        return "P"
    if normalized in {"C", "P"}:
        return normalized
    raise ValueError("right must be C, P, call, or put")


def _records_from_chain(
    frame: pd.DataFrame,
    right: str,
    expiry: str,
    as_of_date: date,
    underlying_price: float | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        record = {key: _json_safe_value(value) for key, value in row.items()}
        record["right"] = right
        record["expiry"] = expiry
        record["days_to_expiry"] = days_to_expiry(expiry, as_of_date)
        record["moneyness"] = calculate_moneyness(
            underlying_price,
            float(record["strike"]),
        )
        records.append(record)
    return records


def _contract_payload(
    chain: dict[str, Any],
    row: dict[str, Any],
    right: str,
    requested_strike: float,
) -> dict[str, Any]:
    payload = dict(row)
    payload.update(
        {
            "symbol": payload.get("contractSymbol"),
            "underlying": chain["underlying"],
            "underlying_price": chain["underlying_price"],
            "right": right,
            "expiry": chain["expiry"],
            "requested_strike": float(requested_strike),
            "strike_match_delta": abs(
                float(payload.get("strike", requested_strike)) - float(requested_strike)
            ),
        }
    )
    return payload


def _select_expiry(expirations: list[str], expiry: str | None) -> str:
    if not expirations:
        raise ValueError("No option expirations available")
    if expiry is None:
        return expirations[0]
    selected = _parse_date(expiry).isoformat()
    if selected not in expirations:
        raise ValueError(
            f"Expiry {selected} is not available. Available expiries: "
            + ", ".join(expirations)
        )
    return selected


def _resolve_underlying_price(ticker) -> float | None:
    candidates: list[Any] = []
    try:
        fast_info = yf_retry(lambda: ticker.fast_info)
        candidates.extend(
            [
                getattr(fast_info, "last_price", None),
                fast_info.get("last_price") if hasattr(fast_info, "get") else None,
                fast_info.get("lastPrice") if hasattr(fast_info, "get") else None,
            ]
        )
    except Exception:
        pass
    try:
        history = yf_retry(lambda: ticker.history(period="5d"))
        if not history.empty:
            candidates.append(history["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return _first_number(*candidates)


def _first_number(*values: Any) -> float | None:
    for value in values:
        numeric = _to_float(value)
        if numeric is not None:
            return numeric
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric):
        return None
    return numeric


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _parse_date(value: str | date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _parse_optional_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    return _parse_date(value)


def _clean_symbol(value: str) -> str:
    symbol = str(value).strip().upper()
    if not symbol:
        raise ValueError("symbol cannot be empty")
    return symbol


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _norm_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)
