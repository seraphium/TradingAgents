"""Structured per-instrument proposals used by portfolio allocation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.portfolio.rebalancing import RATING_SCORES


class ProposalSource(str, Enum):
    """How an instrument proposal was produced."""

    STRUCTURED = "structured"
    TEXT_FALLBACK = "text_fallback"
    UNSUPPORTED = "unsupported"


class InstrumentProposal(BaseModel):
    """Validated allocation input produced for one portfolio instrument."""

    model_config = ConfigDict(use_enum_values=True)

    symbol: str
    rating: str = "Hold"
    conviction: float = Field(ge=-1, le=1)
    horizon: str = "unspecified"
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    summary: str = ""
    source: ProposalSource = ProposalSource.STRUCTURED

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: Any) -> str:
        symbol = str(value).strip().upper()
        if not symbol:
            raise ValueError("proposal symbol cannot be empty")
        return symbol

    @field_validator("rating", mode="before")
    @classmethod
    def normalize_rating(cls, value: Any) -> str:
        return parse_rating(str(value))

    @property
    def allocation_score(self) -> float:
        """Return the confidence-weighted deterministic optimizer score."""

        return self.conviction * self.confidence


def extract_instrument_proposals(
    portfolio_result: Mapping[str, Any],
) -> dict[str, InstrumentProposal]:
    """Extract structured proposals, retaining Markdown rating parsing as fallback."""

    proposals: dict[str, InstrumentProposal] = {}
    for holding in portfolio_result.get("holdings", []):
        symbol = str(holding.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        structured = holding.get("instrument_proposal")
        if structured is not None:
            try:
                proposal = InstrumentProposal.model_validate(structured)
                if proposal.symbol != symbol:
                    raise ValueError("proposal symbol does not match holding")
                proposals[symbol] = proposal
                continue
            except (TypeError, ValueError):
                pass
        proposals[symbol] = _fallback_proposal(holding, symbol)
    return proposals


def _fallback_proposal(holding: Mapping[str, Any], symbol: str) -> InstrumentProposal:
    status = str(holding.get("analysis_status", "unknown"))
    asset_type = str(holding.get("asset_type", ""))
    analysis = holding.get("analysis") or {}
    final_decision = str(analysis.get("final_trade_decision") or "")
    rating = parse_rating(final_decision or "Hold")
    risk_flags: list[str] = []

    if status in {"failed", "skipped", "unknown"} and asset_type != "cash":
        return InstrumentProposal(
            symbol=symbol,
            rating="Hold",
            conviction=0,
            confidence=0,
            risk_flags=["analysis_unavailable"],
            summary=str(
                holding.get("error")
                or holding.get("skip_reason")
                or "Analysis unavailable."
            ),
            source=ProposalSource.UNSUPPORTED,
        )

    confidence = 1.0
    if asset_type == "option" and not holding.get("contract_analysis"):
        risk_flags.append("contract_level_evidence_unavailable")
        confidence = 0.5

    return InstrumentProposal(
        symbol=symbol,
        rating=rating,
        conviction=RATING_SCORES[rating],
        confidence=confidence,
        risk_flags=risk_flags,
        summary=final_decision,
        source=ProposalSource.TEXT_FALLBACK,
    )
