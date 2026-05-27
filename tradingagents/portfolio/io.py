"""Portfolio file parsing utilities."""

from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import ValidationError

from tradingagents.portfolio.schemas import (
    AssetType,
    PortfolioConstraints,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
)


class PortfolioParseError(ValueError):
    """Raised when a portfolio input file cannot be parsed."""


_OPTION_SYMBOL_RE = re.compile(
    r"^(?P<underlying>[A-Z0-9.\-]+?)(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$"
)


def load_portfolio_file(
    path: str | Path,
    *,
    trade_date: date | str | None = None,
    base_currency: str = "USD",
    constraints: PortfolioConstraints | Mapping[str, Any] | None = None,
) -> PortfolioRequest:
    """Load a portfolio request from a CSV or JSON file."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return parse_portfolio_csv(
            file_path,
            trade_date=trade_date,
            base_currency=base_currency,
            constraints=constraints,
        )
    if suffix == ".json":
        return parse_portfolio_json(
            file_path,
            trade_date=trade_date,
            base_currency=base_currency,
            constraints=constraints,
        )
    raise PortfolioParseError(f"unsupported portfolio file type: {file_path.suffix}")


def parse_portfolio_csv(
    path: str | Path,
    *,
    trade_date: date | str | None,
    base_currency: str = "USD",
    constraints: PortfolioConstraints | Mapping[str, Any] | None = None,
) -> PortfolioRequest:
    """Parse a flattened CSV portfolio file into a validated request."""

    if trade_date is None:
        raise PortfolioParseError("trade_date is required when parsing portfolio CSV files")

    file_path = Path(path)
    try:
        with file_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise PortfolioParseError("portfolio CSV must include a header row")
            positions = [
                _position_from_flat_row(_normalize_mapping(row), row_number=index)
                for index, row in enumerate(reader, start=2)
            ]
    except OSError as exc:
        raise PortfolioParseError(f"could not read portfolio CSV: {file_path}") from exc

    if not positions:
        raise PortfolioParseError("portfolio CSV must include at least one position")

    return _build_request(
        positions=positions,
        trade_date=trade_date,
        base_currency=base_currency,
        constraints=constraints,
    )


def parse_portfolio_json(
    path: str | Path,
    *,
    trade_date: date | str | None = None,
    base_currency: str = "USD",
    constraints: PortfolioConstraints | Mapping[str, Any] | None = None,
) -> PortfolioRequest:
    """Parse a portfolio JSON file into a validated request.

    The JSON may be a full ``PortfolioRequest`` payload, a list of flattened
    position rows, or an object with flattened rows under ``positions``.
    """

    file_path = Path(path)
    try:
        with file_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise PortfolioParseError(f"invalid portfolio JSON: {file_path}") from exc
    except OSError as exc:
        raise PortfolioParseError(f"could not read portfolio JSON: {file_path}") from exc

    return parse_portfolio_json_payload(
        payload,
        trade_date=trade_date,
        base_currency=base_currency,
        constraints=constraints,
    )


def parse_portfolio_json_payload(
    payload: Any,
    *,
    trade_date: date | str | None = None,
    base_currency: str = "USD",
    constraints: PortfolioConstraints | Mapping[str, Any] | None = None,
) -> PortfolioRequest:
    """Parse an in-memory JSON-compatible portfolio payload."""

    if isinstance(payload, list):
        if trade_date is None:
            raise PortfolioParseError("trade_date is required for JSON position lists")
        positions = _positions_from_flat_rows(payload)
        return _build_request(
            positions=positions,
            trade_date=trade_date,
            base_currency=base_currency,
            constraints=constraints,
        )

    if not isinstance(payload, Mapping):
        raise PortfolioParseError("portfolio JSON must be an object or a list of positions")

    data = dict(payload)
    raw_positions = data.get("positions")
    if raw_positions is None:
        raise PortfolioParseError("portfolio JSON must include positions")

    request_trade_date = trade_date if trade_date is not None else data.get("trade_date")
    if request_trade_date is None:
        raise PortfolioParseError("portfolio JSON must include trade_date")

    request_base_currency = base_currency
    if base_currency == "USD" and data.get("base_currency") is not None:
        request_base_currency = data["base_currency"]

    request_constraints = constraints
    if request_constraints is None and data.get("constraints") is not None:
        request_constraints = data["constraints"]

    positions = _positions_from_json_positions(raw_positions)
    return _build_request(
        positions=positions,
        trade_date=request_trade_date,
        base_currency=request_base_currency,
        constraints=request_constraints,
    )


def normalize_portfolio_symbol(symbol: str) -> str:
    """Normalize portfolio ticker input while preserving exchange suffixes."""

    if not isinstance(symbol, str):
        raise PortfolioParseError("symbol must be a string")
    normalized = symbol.strip().upper()
    if not normalized:
        raise PortfolioParseError("symbol cannot be empty")
    return normalized


def _positions_from_json_positions(raw_positions: Any) -> list[PortfolioPosition]:
    if not isinstance(raw_positions, list):
        raise PortfolioParseError("positions must be a list")

    positions: list[PortfolioPosition] = []
    for index, row in enumerate(raw_positions, start=1):
        if not isinstance(row, Mapping):
            raise PortfolioParseError(f"position {index} must be an object")
        normalized_row = _normalize_mapping(row)
        if "instrument" in normalized_row:
            positions.append(_position_from_structured_row(normalized_row, index=index))
        else:
            positions.append(_position_from_flat_row(normalized_row, row_number=index))
    return positions


def _positions_from_flat_rows(rows: Iterable[Any]) -> list[PortfolioPosition]:
    positions: list[PortfolioPosition] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise PortfolioParseError(f"position {index} must be an object")
        positions.append(_position_from_flat_row(_normalize_mapping(row), row_number=index))
    return positions


def _position_from_structured_row(row: Mapping[str, Any], *, index: int) -> PortfolioPosition:
    try:
        return PortfolioPosition(**_none_if_blank_dict(row))
    except ValidationError as exc:
        raise PortfolioParseError(f"invalid portfolio position {index}: {exc}") from exc


def _position_from_flat_row(row: Mapping[str, Any], *, row_number: int) -> PortfolioPosition:
    try:
        symbol = normalize_portfolio_symbol(_required(row, "symbol", row_number))
        asset_type = str(_required(row, "asset_type", row_number)).strip().lower()

        option_defaults = _infer_option_terms(symbol) if asset_type == AssetType.OPTION.value else {}
        instrument = PortfolioInstrument(
            symbol=symbol,
            asset_type=asset_type,
            underlying=_string_value(row.get("underlying")) or option_defaults.get("underlying"),
            expiry=_string_value(row.get("expiry")) or option_defaults.get("expiry"),
            strike=_float_value(row.get("strike"), "strike", row_number)
            if _has_value(row.get("strike"))
            else option_defaults.get("strike"),
            right=_string_value(row.get("right")) or option_defaults.get("right"),
            contract_symbol=_string_value(row.get("contract_symbol")),
        )

        return PortfolioPosition(
            instrument=instrument,
            current_weight=_float_value(row.get("current_weight"), "current_weight", row_number),
            quantity=_float_value(row.get("quantity"), "quantity", row_number),
            market_value=_float_value(row.get("market_value"), "market_value", row_number),
            cost_basis=_float_value(row.get("cost_basis"), "cost_basis", row_number),
            target_min_weight=_float_value(
                row.get("target_min_weight"),
                "target_min_weight",
                row_number,
            ),
            target_max_weight=_float_value(
                row.get("target_max_weight"),
                "target_max_weight",
                row_number,
            ),
        )
    except ValidationError as exc:
        raise PortfolioParseError(f"invalid portfolio row {row_number}: {exc}") from exc


def _build_request(
    *,
    positions: list[PortfolioPosition],
    trade_date: date | str,
    base_currency: str,
    constraints: PortfolioConstraints | Mapping[str, Any] | None,
) -> PortfolioRequest:
    try:
        request_constraints = constraints if constraints is not None else PortfolioConstraints()
        return PortfolioRequest(
            positions=positions,
            trade_date=trade_date,
            base_currency=base_currency,
            constraints=request_constraints,
        )
    except ValidationError as exc:
        raise PortfolioParseError(f"invalid portfolio request: {exc}") from exc


def _infer_option_terms(symbol: str) -> dict[str, Any]:
    match = _OPTION_SYMBOL_RE.match(symbol)
    if not match:
        return {}

    expiry_part = match.group("expiry")
    strike_part = match.group("strike")
    return {
        "underlying": match.group("underlying"),
        "expiry": f"20{expiry_part[:2]}-{expiry_part[2:4]}-{expiry_part[4:]}",
        "right": match.group("right"),
        "strike": int(strike_part) / 1000,
    }


def _normalize_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key).strip().lower(): value
        for key, value in row.items()
        if key is not None
    }


def _none_if_blank_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _none_if_blank(value) for key, value in row.items()}


def _none_if_blank(value: Any) -> Any:
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _required(row: Mapping[str, Any], field_name: str, row_number: int) -> Any:
    value = row.get(field_name)
    if not _has_value(value):
        raise PortfolioParseError(f"portfolio row {row_number} missing required {field_name}")
    return value


def _has_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value.strip() == "")


def _string_value(value: Any) -> str | None:
    if not _has_value(value):
        return None
    if not isinstance(value, str):
        return str(value)
    return value.strip()


def _float_value(value: Any, field_name: str, row_number: int) -> float | None:
    if not _has_value(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            try:
                return float(stripped[:-1].strip()) / 100
            except ValueError as exc:
                raise PortfolioParseError(
                    f"portfolio row {row_number} has invalid {field_name}: {value!r}"
                ) from exc
        value = stripped
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PortfolioParseError(
            f"portfolio row {row_number} has invalid {field_name}: {value!r}"
        ) from exc
