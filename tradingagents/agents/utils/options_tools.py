from langchain_core.tools import tool
import json
from typing import Annotated

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_options_chain(
    underlying: Annotated[str, "underlying ticker symbol"],
    expiry: Annotated[str | None, "option expiry in yyyy-mm-dd format"] = None,
    curr_date: Annotated[str | None, "current date in yyyy-mm-dd format"] = None,
) -> str:
    """Retrieve calls and puts for an underlying option expiry."""

    try:
        return route_to_vendor("get_options_chain", underlying, expiry, curr_date)
    except Exception as exc:
        return _options_unavailable_payload(
            "get_options_chain",
            underlying,
            exc,
            expiry=expiry,
        )


@tool
def get_option_contract(
    underlying: Annotated[str, "underlying ticker symbol"],
    expiry: Annotated[str, "option expiry in yyyy-mm-dd format"],
    strike: Annotated[float, "option strike price"],
    right: Annotated[str, "C/call or P/put"],
    curr_date: Annotated[str | None, "current date in yyyy-mm-dd format"] = None,
    contract_symbol: Annotated[str | None, "optional vendor contract symbol"] = None,
) -> str:
    """Lookup one option contract by terms and optional contract symbol."""

    try:
        return route_to_vendor(
            "get_option_contract",
            underlying,
            expiry,
            strike,
            right,
            curr_date,
            contract_symbol,
        )
    except Exception as exc:
        return _options_unavailable_payload(
            "get_option_contract",
            underlying,
            exc,
            expiry=expiry,
            strike=strike,
            right=right,
            contract_symbol=contract_symbol,
        )


@tool
def get_option_greeks(
    underlying: Annotated[str, "underlying ticker symbol"],
    expiry: Annotated[str, "option expiry in yyyy-mm-dd format"],
    strike: Annotated[float, "option strike price"],
    right: Annotated[str, "C/call or P/put"],
    curr_date: Annotated[str | None, "current date in yyyy-mm-dd format"] = None,
    contract_symbol: Annotated[str | None, "optional vendor contract symbol"] = None,
    underlying_price: Annotated[float | None, "optional current underlying price"] = None,
    implied_volatility: Annotated[float | None, "optional implied volatility decimal"] = None,
    risk_free_rate: Annotated[float, "annualized risk-free rate as decimal"] = 0.05,
    dividend_yield: Annotated[float, "annualized dividend yield as decimal"] = 0.0,
) -> str:
    """Retrieve option Greeks, using a fallback model when provider Greeks are missing."""

    try:
        return route_to_vendor(
            "get_option_greeks",
            underlying,
            expiry,
            strike,
            right,
            curr_date,
            contract_symbol,
            underlying_price,
            implied_volatility,
            risk_free_rate,
            dividend_yield,
        )
    except Exception as exc:
        return _options_unavailable_payload(
            "get_option_greeks",
            underlying,
            exc,
            expiry=expiry,
            strike=strike,
            right=right,
            contract_symbol=contract_symbol,
        )


def _options_unavailable_payload(
    tool_name: str,
    underlying: str,
    exc: Exception,
    **context,
) -> str:
    payload = {
        "status": "unavailable",
        "tool": tool_name,
        "underlying": str(underlying).strip().upper(),
        "reason": str(exc),
        "message": (
            "Options data is unavailable from the configured vendor. "
            "Continue the portfolio analysis using stock and cash holdings; "
            "treat option-specific chain and Greek fields as missing."
        ),
    }
    payload.update({key: value for key, value in context.items() if value is not None})
    return json.dumps(payload, indent=2, default=str)
