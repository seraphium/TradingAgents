"""Helpers for portfolio-level agent prompts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any

from pydantic import BaseModel


DEFAULT_EVIDENCE_LIMIT = 900


def format_portfolio_payload(value: Any) -> str:
    """Render portfolio analytics/proposal objects as compact JSON for prompts."""

    normalized = _normalize_payload(value)
    return json.dumps(normalized, indent=2, sort_keys=True, default=str)


def format_holding_evidence(
    holdings: list[dict[str, Any]],
    *,
    text_limit: int = DEFAULT_EVIDENCE_LIMIT,
) -> str:
    """Render compact single-holding decisions and debate evidence for prompts.

    Portfolio-level agents need the old single-stock Portfolio Manager output
    and debate conclusions, but not every full analyst report. This keeps
    prompts focused while preserving the decision evidence that drives target
    weights and final allocation review.
    """

    evidence = [_holding_evidence(holding, text_limit=text_limit) for holding in holdings]
    return format_portfolio_payload(evidence)


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


def _holding_evidence(
    holding: dict[str, Any],
    *,
    text_limit: int,
) -> dict[str, Any]:
    analysis = holding.get("analysis") or {}
    investment_debate = analysis.get("investment_debate_state") or {}
    risk_debate = analysis.get("risk_debate_state") or {}
    return {
        "symbol": holding.get("symbol"),
        "asset_type": holding.get("asset_type"),
        "current_weight": holding.get("current_weight"),
        "analysis_status": holding.get("analysis_status"),
        "analysis_symbol": holding.get("analysis_symbol"),
        "skip_reason": holding.get("skip_reason"),
        "error": holding.get("error"),
        "signal": _truncate_text(analysis.get("signal"), text_limit),
        "single_stock_portfolio_decision": _truncate_text(
            analysis.get("final_trade_decision"),
            text_limit,
        ),
        "trader_plan": _truncate_text(
            analysis.get("trader_investment_plan"),
            text_limit,
        ),
        "research_manager_decision": _truncate_text(
            investment_debate.get("judge_decision"),
            text_limit,
        ),
        "risk_manager_decision": _truncate_text(
            risk_debate.get("judge_decision"),
            text_limit,
        ),
        "risk_debate": {
            "aggressive": _truncate_text(
                risk_debate.get("aggressive_history"),
                text_limit,
            ),
            "conservative": _truncate_text(
                risk_debate.get("conservative_history"),
                text_limit,
            ),
            "neutral": _truncate_text(
                risk_debate.get("neutral_history"),
                text_limit,
            ),
        },
    }


def _truncate_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 15, 0)].rstrip() + "... [truncated]"
