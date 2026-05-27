"""Typed portfolio input models.

These models are intentionally independent from the LangGraph state used by
the single-instrument pipeline. Portfolio orchestration can validate and carry
multi-holding inputs here, then reuse the existing graph per holding in later
phases.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AssetType(str, Enum):
    """Supported portfolio component types."""

    STOCK = "stock"
    OPTION = "option"
    CASH = "cash"


class OptionRight(str, Enum):
    """Option contract side."""

    CALL = "C"
    PUT = "P"


def _clean_symbol(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("symbol fields must be strings")
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError("symbol cannot be empty")
    return symbol


def _clean_asset_type(value: AssetType | str) -> AssetType | str:
    if isinstance(value, AssetType):
        return value
    if not isinstance(value, str):
        raise ValueError("asset_type must be stock, option, or cash")
    return value.strip().lower()


def _clean_option_right(value: OptionRight | str | None) -> OptionRight | str | None:
    if value is None or isinstance(value, OptionRight):
        return value
    if not isinstance(value, str):
        raise ValueError("right must be C, P, call, or put")
    normalized = value.strip().upper()
    if normalized == "CALL":
        return "C"
    if normalized == "PUT":
        return "P"
    return normalized


class PortfolioInstrument(BaseModel):
    """A tradable instrument or explicit cash component in a portfolio."""

    model_config = ConfigDict(use_enum_values=False)

    symbol: str = Field(description="Ticker, option contract symbol, or CASH.")
    asset_type: AssetType = Field(description="Instrument type: stock, option, or cash.")
    underlying: Optional[str] = Field(
        default=None,
        description="Required for options; underlying ticker symbol.",
    )
    expiry: Optional[date] = Field(
        default=None,
        description="Required for options; contract expiration date.",
    )
    strike: Optional[float] = Field(
        default=None,
        gt=0,
        description="Required for options; strike price.",
    )
    right: Optional[OptionRight] = Field(
        default=None,
        description="Required for options; C for call or P for put.",
    )
    contract_symbol: Optional[str] = Field(
        default=None,
        description="Optional vendor-specific option contract symbol.",
    )

    @field_validator("asset_type", mode="before")
    @classmethod
    def normalize_asset_type(cls, value: AssetType | str) -> AssetType | str:
        return _clean_asset_type(value)

    @field_validator("symbol", "underlying", "contract_symbol", mode="before")
    @classmethod
    def normalize_symbol_fields(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _clean_symbol(value)

    @field_validator("right", mode="before")
    @classmethod
    def normalize_option_right(
        cls, value: OptionRight | str | None
    ) -> OptionRight | str | None:
        return _clean_option_right(value)

    @model_validator(mode="after")
    def validate_by_asset_type(self) -> "PortfolioInstrument":
        if self.asset_type == AssetType.STOCK:
            self._reject_option_fields("stock")
        elif self.asset_type == AssetType.CASH:
            if self.symbol != "CASH":
                raise ValueError("cash instruments must use symbol CASH")
            self._reject_option_fields("cash")
        elif self.asset_type == AssetType.OPTION:
            missing = [
                field_name
                for field_name in ("underlying", "expiry", "strike", "right")
                if getattr(self, field_name) is None
            ]
            if missing:
                raise ValueError(
                    "option instruments require: " + ", ".join(missing)
                )
            if self.underlying == self.symbol:
                raise ValueError("option underlying must differ from option symbol")
        return self

    def _reject_option_fields(self, asset_label: str) -> None:
        supplied = [
            field_name
            for field_name in ("underlying", "expiry", "strike", "right", "contract_symbol")
            if getattr(self, field_name) is not None
        ]
        if supplied:
            raise ValueError(
                f"{asset_label} instruments cannot include option fields: "
                + ", ".join(supplied)
            )


class PortfolioPosition(BaseModel):
    """A weighted component of a portfolio."""

    instrument: PortfolioInstrument
    current_weight: float = Field(
        ge=0,
        le=1,
        description="Current portfolio allocation as a decimal, e.g. 0.25 for 25%.",
    )
    quantity: Optional[float] = Field(
        default=None,
        ge=0,
        description="Optional held quantity or contract count.",
    )
    market_value: Optional[float] = Field(
        default=None,
        ge=0,
        description="Optional current market value in the portfolio base currency.",
    )
    cost_basis: Optional[float] = Field(
        default=None,
        ge=0,
        description="Optional total cost basis in the portfolio base currency.",
    )
    target_min_weight: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional minimum target allocation.",
    )
    target_max_weight: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional maximum target allocation.",
    )

    @model_validator(mode="after")
    def validate_position(self) -> "PortfolioPosition":
        if (
            self.target_min_weight is not None
            and self.target_max_weight is not None
            and self.target_min_weight > self.target_max_weight
        ):
            raise ValueError("target_min_weight cannot exceed target_max_weight")

        if self.instrument.asset_type == AssetType.CASH and self.quantity is not None:
            raise ValueError("cash positions should use weight or market_value, not quantity")

        if self.instrument.asset_type != AssetType.CASH and (
            self.quantity is None and self.market_value is None
        ):
            raise ValueError("non-cash positions require quantity or market_value")

        return self


class PortfolioConstraints(BaseModel):
    """Portfolio-level risk and allocation constraints."""

    max_single_position_weight: Optional[float] = Field(default=None, gt=0, le=1)
    max_options_weight: Optional[float] = Field(default=None, ge=0, le=1)
    min_cash_weight: Optional[float] = Field(default=None, ge=0, le=1)
    max_cash_weight: Optional[float] = Field(default=None, ge=0, le=1)
    weight_tolerance: float = Field(
        default=0.005,
        ge=0,
        le=0.05,
        description="Allowed absolute difference from a total portfolio weight of 1.0.",
    )
    custom: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_constraint_ranges(self) -> "PortfolioConstraints":
        if (
            self.min_cash_weight is not None
            and self.max_cash_weight is not None
            and self.min_cash_weight > self.max_cash_weight
        ):
            raise ValueError("min_cash_weight cannot exceed max_cash_weight")
        return self


class PortfolioRequest(BaseModel):
    """Validated request for a whole-portfolio analysis run."""

    positions: List[PortfolioPosition] = Field(min_length=1)
    trade_date: date
    base_currency: str = Field(default="USD", min_length=3, max_length=3)
    constraints: PortfolioConstraints = Field(default_factory=PortfolioConstraints)

    @field_validator("base_currency", mode="before")
    @classmethod
    def normalize_base_currency(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("base_currency must be a 3-letter currency code")
        currency = value.strip().upper()
        if len(currency) != 3 or not currency.isalpha():
            raise ValueError("base_currency must be a 3-letter currency code")
        return currency

    @model_validator(mode="after")
    def validate_portfolio(self) -> "PortfolioRequest":
        total_weight = sum(position.current_weight for position in self.positions)
        if abs(total_weight - 1.0) > self.constraints.weight_tolerance:
            raise ValueError(
                f"portfolio weights must sum to 1.0 within tolerance; got {total_weight:.6f}"
            )

        symbols = [position.instrument.symbol for position in self.positions]
        duplicates = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
        if duplicates:
            raise ValueError("duplicate portfolio symbols: " + ", ".join(duplicates))

        if self.constraints.max_single_position_weight is not None:
            oversized = [
                position.instrument.symbol
                for position in self.positions
                if position.current_weight > self.constraints.max_single_position_weight
            ]
            if oversized:
                raise ValueError(
                    "positions exceed max_single_position_weight: "
                    + ", ".join(oversized)
                )

        options_weight = sum(
            position.current_weight
            for position in self.positions
            if position.instrument.asset_type == AssetType.OPTION
        )
        if (
            self.constraints.max_options_weight is not None
            and options_weight > self.constraints.max_options_weight
        ):
            raise ValueError(
                f"options weight exceeds max_options_weight: {options_weight:.6f}"
            )

        cash_weight = sum(
            position.current_weight
            for position in self.positions
            if position.instrument.asset_type == AssetType.CASH
        )
        if (
            self.constraints.min_cash_weight is not None
            and cash_weight < self.constraints.min_cash_weight
        ):
            raise ValueError(f"cash weight is below min_cash_weight: {cash_weight:.6f}")
        if (
            self.constraints.max_cash_weight is not None
            and cash_weight > self.constraints.max_cash_weight
        ):
            raise ValueError(f"cash weight exceeds max_cash_weight: {cash_weight:.6f}")

        return self
