"""Bounded, deterministic constraint adjustments requested by review agents."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradingagents.portfolio.schemas import AssetType, PortfolioRequest


class ConstraintAdjustmentRequest(BaseModel):
    """A review agent's request to tighten deterministic optimizer constraints."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=500)
    min_cash_weight: float | None = Field(default=None, ge=0, le=1)
    max_single_position_weight: float | None = Field(default=None, gt=0, le=1)
    max_options_weight: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_adjustment(self) -> "ConstraintAdjustmentRequest":
        if all(
            value is None
            for value in (
                self.min_cash_weight,
                self.max_single_position_weight,
                self.max_options_weight,
            )
        ):
            raise ValueError("at least one constraint adjustment is required")
        return self

    def adjustments(self) -> dict[str, float]:
        """Return only requested numeric adjustments."""

        return {
            key: float(value)
            for key, value in self.model_dump(exclude={"reason"}).items()
            if value is not None
        }


class ValidatedConstraintAdjustment(BaseModel):
    """A validated, feasible set of tighter constraints for one rerun."""

    model_config = ConfigDict(extra="forbid")

    adjustments: dict[str, float]
    reasons: list[str]
    rejected_requests: list[str] = Field(default_factory=list)


_JSON_FENCE = re.compile(
    r"```(?:json|constraint_adjustments)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)


def extract_constraint_adjustment_request(
    text: str,
) -> ConstraintAdjustmentRequest | None:
    """Extract an optional request from an agent response without trusting prose."""

    if not text or "constraint" not in text.lower():
        return None
    candidates = [match.group(1) for match in _JSON_FENCE.finditer(text)]
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        nested = (
            payload.get("constraint_adjustments") if isinstance(payload, dict) else None
        )
        if isinstance(nested, dict):
            payload = nested
        if not isinstance(payload, dict):
            continue
        try:
            return ConstraintAdjustmentRequest.model_validate(payload)
        except ValueError:
            continue
    return None


def validate_constraint_adjustment_requests(
    portfolio_request: PortfolioRequest,
    requests: list[tuple[str, ConstraintAdjustmentRequest]],
    *,
    max_adjustment: float = 0.10,
) -> ValidatedConstraintAdjustment | None:
    """Validate requests, accepting only bounded risk-tightening adjustments."""

    accepted: list[tuple[str, ConstraintAdjustmentRequest]] = []
    rejected: list[str] = []
    for source, request in requests:
        error = _request_error(
            portfolio_request, request, max_adjustment=max_adjustment
        )
        if error:
            rejected.append(f"{source}: {error}")
        else:
            accepted.append((source, request))
    if not accepted:
        return None

    adjustments: dict[str, float] = {}
    for key in ("max_single_position_weight", "max_options_weight"):
        values = [
            getattr(request, key)
            for _, request in accepted
            if getattr(request, key) is not None
        ]
        if values:
            adjustments[key] = float(min(values))
    cash_values = [
        request.min_cash_weight
        for _, request in accepted
        if request.min_cash_weight is not None
    ]
    if cash_values:
        adjustments["min_cash_weight"] = float(max(cash_values))

    feasibility_error = _feasibility_error(portfolio_request, adjustments)
    if feasibility_error:
        rejected.extend(f"{source}: {feasibility_error}" for source, _ in accepted)
        return None
    return ValidatedConstraintAdjustment(
        adjustments=adjustments,
        reasons=[f"{source}: {request.reason}" for source, request in accepted],
        rejected_requests=rejected,
    )


def _request_error(
    portfolio_request: PortfolioRequest,
    request: ConstraintAdjustmentRequest,
    *,
    max_adjustment: float,
) -> str | None:
    constraints = portfolio_request.constraints
    cash_weight = sum(
        position.current_weight
        for position in portfolio_request.positions
        if position.instrument.asset_type == AssetType.CASH
    )
    risky_weights = [
        position.current_weight
        for position in portfolio_request.positions
        if position.instrument.asset_type != AssetType.CASH
    ]
    options_weight = sum(
        position.current_weight
        for position in portfolio_request.positions
        if position.instrument.asset_type == AssetType.OPTION
    )
    baselines = {
        "min_cash_weight": max(constraints.min_cash_weight or 0.0, cash_weight),
        "max_single_position_weight": min(
            constraints.max_single_position_weight or 1.0,
            max(risky_weights, default=0.0),
        ),
        "max_options_weight": min(
            constraints.max_options_weight or 1.0, options_weight
        ),
    }
    if request.min_cash_weight is not None:
        if not any(
            position.instrument.asset_type == AssetType.CASH
            for position in portfolio_request.positions
        ):
            return "min_cash_weight requires an explicit cash sleeve"
        baseline = baselines["min_cash_weight"]
        if request.min_cash_weight <= baseline:
            return "min_cash_weight must tighten the current allocation and user constraint"
        if request.min_cash_weight - baseline > max_adjustment + 1e-12:
            return f"min_cash_weight change exceeds the {max_adjustment:.0%} bound"
    for key in ("max_single_position_weight", "max_options_weight"):
        requested = getattr(request, key)
        if requested is None:
            continue
        baseline = baselines[key]
        if requested >= baseline:
            return f"{key} must tighten the current allocation and user constraint"
        if baseline - requested > max_adjustment + 1e-12:
            return f"{key} change exceeds the {max_adjustment:.0%} bound"
    return None


def _feasibility_error(
    portfolio_request: PortfolioRequest, adjustments: Mapping[str, float]
) -> str | None:
    constraints = portfolio_request.constraints
    cash_minimum = adjustments.get(
        "min_cash_weight", constraints.min_cash_weight or 0.0
    )
    max_single = adjustments.get(
        "max_single_position_weight", constraints.max_single_position_weight or 1.0
    )
    minimum_total = 0.0
    maximum_total = 0.0
    for position in portfolio_request.positions:
        minimum = position.target_min_weight or 0.0
        maximum = (
            position.target_max_weight
            if position.target_max_weight is not None
            else 1.0
        )
        if position.instrument.asset_type == AssetType.CASH:
            minimum = max(minimum, cash_minimum)
            if constraints.max_cash_weight is not None:
                maximum = min(maximum, constraints.max_cash_weight)
        else:
            maximum = min(maximum, max_single)
        minimum_total += minimum
        maximum_total += maximum
    tolerance = constraints.weight_tolerance
    if minimum_total > 1.0 + tolerance or maximum_total < 1.0 - tolerance:
        return "requested constraints cannot support a fully allocated portfolio"
    return None
