"""Deterministic portfolio data-quality assessment and allocation restrictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from tradingagents.portfolio.analytics import PortfolioAnalytics
from tradingagents.portfolio.instrument_proposals import InstrumentProposal
from tradingagents.portfolio.schemas import AssetType, PortfolioRequest


@dataclass(frozen=True)
class DataQualityPolicy:
    """Configurable thresholds for deciding whether risk increases are supported."""

    minimum_analysis_coverage: float = 0.8
    insufficient_analysis_coverage: float = 0.5
    minimum_proposal_confidence: float = 0.25
    restrict_options_without_contract_evidence: bool = True

    def __post_init__(self) -> None:
        for name in (
            "minimum_analysis_coverage",
            "insufficient_analysis_coverage",
            "minimum_proposal_confidence",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.insufficient_analysis_coverage > self.minimum_analysis_coverage:
            raise ValueError(
                "insufficient_analysis_coverage cannot exceed minimum_analysis_coverage"
            )

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "DataQualityPolicy":
        values = dict((config or {}).get("portfolio_data_quality", {}))
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass(frozen=True)
class DataQualityAssessment:
    """Evidence coverage and restrictions applied before deterministic allocation."""

    status: str
    weighted_analysis_coverage: float
    metric_coverage: dict[str, float]
    restricted_symbols: list[str] = field(default_factory=list)
    restrictions: list[str] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_data_quality(
    request: PortfolioRequest,
    analytics: PortfolioAnalytics,
    holdings: list[Mapping[str, Any]],
    proposals: Mapping[str, InstrumentProposal],
    *,
    policy: DataQualityPolicy | None = None,
) -> DataQualityAssessment:
    """Assess weighted evidence coverage and identify symbols that cannot increase."""

    effective_policy = policy or DataQualityPolicy()
    holdings_by_symbol = {str(item.get("symbol")): item for item in holdings}
    non_cash = [
        p for p in request.positions if p.instrument.asset_type != AssetType.CASH
    ]
    total_weight = sum(float(p.current_weight) for p in non_cash)
    analyzed_weight = 0.0
    restricted: set[str] = set()
    restrictions: list[str] = []
    warnings: list[str] = []

    for position in non_cash:
        symbol = position.instrument.symbol
        holding = holdings_by_symbol.get(symbol, {})
        status = holding.get("analysis_status")
        proposal = proposals.get(symbol)
        supported = (
            status in {"analyzed", "underlying_analyzed"} and proposal is not None
        )
        if supported:
            analyzed_weight += float(position.current_weight)
        if (
            not supported
            or proposal is None
            or proposal.confidence < effective_policy.minimum_proposal_confidence
        ):
            restricted.add(symbol)
            restrictions.append(
                f"{symbol} cannot increase because analysis is failed, unsupported, or low-confidence."
            )
        if (
            effective_policy.restrict_options_without_contract_evidence
            and position.instrument.asset_type == AssetType.OPTION
            and not holding.get("contract_analysis")
        ):
            restricted.add(symbol)
            restrictions.append(
                f"{symbol} cannot increase without contract-level option evidence."
            )

    analysis_coverage = analyzed_weight / total_weight if total_weight else 1.0
    metric_coverage = _metric_coverage(request, analytics)
    blocking: list[str] = []
    if analysis_coverage < effective_policy.insufficient_analysis_coverage:
        status = "insufficient"
        blocking.append(
            "Weighted holding-analysis coverage is below the insufficient-data threshold."
        )
        restricted.update(p.instrument.symbol for p in non_cash)
    elif analysis_coverage < effective_policy.minimum_analysis_coverage or restricted:
        status = "restricted"
    else:
        status = "pass"

    if metric_coverage["volatility"] < 1:
        warnings.append(
            "Volatility coverage is incomplete; volatility-driven risk increases are unsupported."
        )
    if metric_coverage["benchmark_beta"] < 1:
        warnings.append("Benchmark beta coverage is incomplete.")

    return DataQualityAssessment(
        status=status,
        weighted_analysis_coverage=round(analysis_coverage, 6),
        metric_coverage=metric_coverage,
        restricted_symbols=sorted(restricted),
        restrictions=restrictions,
        blocking_issues=blocking,
        warnings=warnings,
    )


def _metric_coverage(
    request: PortfolioRequest, analytics: PortfolioAnalytics
) -> dict[str, float]:
    non_cash = [
        p for p in request.positions if p.instrument.asset_type != AssetType.CASH
    ]
    total_weight = sum(float(p.current_weight) for p in non_cash)

    def coverage(values: Mapping[str, Any]) -> float:
        if not total_weight:
            return 1.0
        present = sum(
            float(position.current_weight)
            for position in non_cash
            if values.get(position.instrument.symbol) is not None
        )
        return round(present / total_weight, 6)

    sector_exposure = getattr(analytics, "sector_exposure", {})
    return {
        "volatility": coverage(getattr(analytics, "volatility_by_symbol", {})),
        "benchmark_beta": coverage(getattr(analytics, "beta_by_symbol", {})),
        "risk_contribution": coverage(
            getattr(analytics, "risk_contribution_by_symbol", {})
        ),
        "sector": (
            round(min(sum(sector_exposure.values()) / total_weight, 1.0), 6)
            if total_weight
            else 1.0
        ),
    }
