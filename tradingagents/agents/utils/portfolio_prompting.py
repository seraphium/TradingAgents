"""Helpers for portfolio-level agent prompts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any

from pydantic import BaseModel


def format_portfolio_payload(value: Any) -> str:
    """Render portfolio analytics/proposal objects as compact JSON for prompts."""

    normalized = _normalize_payload(value)
    return json.dumps(normalized, indent=2, sort_keys=True, default=str)


def _normalize_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _normalize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(item) for item in value]
    return value
