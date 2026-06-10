"""Validated deterministic market-regime interpretation and allocation overlays."""

from __future__ import annotations

from enum import Enum
from math import isfinite
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingagents.portfolio.market_context import PortfolioMarketContext


class MarketRegimeLabel(str, Enum):
    """Broad market posture used by the deterministic allocation engine."""

    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


class BenchmarkTrendFeatures(BaseModel):
    """Deterministic trend measurements derived from benchmark closes."""

    observations: int = Field(default=0, ge=0)
    return_20d: float | None = None
    return_60d: float | None = None
    price_vs_50d_sma: float | None = None
    price_vs_200d_sma: float | None = None
    trend_score: float = Field(default=0.0, ge=-1.0, le=1.0)

    @field_validator(
        "return_20d", "return_60d", "price_vs_50d_sma", "price_vs_200d_sma"
    )
    @classmethod
    def finite_optional_metric(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("benchmark trend metrics must be finite")
        return value


class MarketOverlay(BaseModel):
    """Bounded deterministic adjustments applied before hard constraints."""

    model_config = ConfigDict(extra="forbid")

    risk_adjustment: float = Field(default=0.0, ge=-1.0, le=1.0)
    cash_weight_adjustment: float = Field(default=0.0, ge=-0.05, le=0.05)
    sector_weight_adjustments: dict[str, float] = Field(default_factory=dict)

    @field_validator("sector_weight_adjustments")
    @classmethod
    def validate_sector_adjustments(cls, value: dict[str, float]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for sector, adjustment in value.items():
            name = str(sector).strip()
            numeric = float(adjustment)
            if not name:
                raise ValueError("sector overlay names cannot be empty")
            if not isfinite(numeric) or not -0.03 <= numeric <= 0.03:
                raise ValueError(
                    "sector weight adjustments must be between -0.03 and 0.03"
                )
            cleaned[name] = numeric
        if abs(sum(cleaned.values())) > 0.05 + 1e-9:
            raise ValueError("aggregate sector overlay must be bounded to 0.05")
        return cleaned


class MarketRegime(BaseModel):
    """Structured market interpretation consumed by portfolio allocation."""

    model_config = ConfigDict(extra="forbid")

    label: MarketRegimeLabel = MarketRegimeLabel.NEUTRAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    benchmark_symbol: str | None = None
    benchmark_trend: BenchmarkTrendFeatures = Field(
        default_factory=BenchmarkTrendFeatures
    )
    news_sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    summary: str
    allocation_implications: list[str] = Field(default_factory=list)
    overlay: MarketOverlay = Field(default_factory=MarketOverlay)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def reduce_low_confidence_overlay(self) -> "MarketRegime":
        """Guarantee low-confidence interpretations cannot materially allocate."""
        if self.confidence < 0.25 and (
            abs(self.overlay.cash_weight_adjustment) > 1e-12
            or abs(self.overlay.risk_adjustment) > 1e-12
            or self.overlay.sector_weight_adjustments
        ):
            raise ValueError(
                "market overlays must be zero when confidence is below 0.25"
            )
        return self


def derive_benchmark_trend_features(
    prices: list[float] | None,
) -> BenchmarkTrendFeatures:
    """Derive reproducible benchmark trend features from ordered closes."""

    clean = [float(price) for price in prices or [] if _positive_finite(price)]
    if len(clean) < 2:
        return BenchmarkTrendFeatures(observations=len(clean))
    return_20d = _period_return(clean, 20)
    return_60d = _period_return(clean, 60)
    price_vs_50d = _sma_gap(clean, 50)
    price_vs_200d = _sma_gap(clean, 200)
    signals = [
        _scaled(return_20d, 0.08),
        _scaled(return_60d, 0.16),
        _scaled(price_vs_50d, 0.08),
        _scaled(price_vs_200d, 0.16),
    ]
    available = [signal for signal in signals if signal is not None]
    score = sum(available) / len(available) if available else 0.0
    return BenchmarkTrendFeatures(
        observations=len(clean),
        return_20d=return_20d,
        return_60d=return_60d,
        price_vs_50d_sma=price_vs_50d,
        price_vs_200d_sma=price_vs_200d,
        trend_score=_clamp(score, -1.0, 1.0),
    )


def create_market_regime_agent(*, max_cash_shift: float = 0.05):
    """Create a deterministic market-regime node with a LangGraph-like contract."""

    bounded_cash_shift = _clamp(float(max_cash_shift), 0.0, 0.05)

    def market_regime_node(state: dict[str, Any]) -> dict[str, MarketRegime]:
        inputs = state.get("portfolio_analytics_inputs")
        context = state.get("portfolio_market_context")
        benchmark_prices = getattr(inputs, "benchmark_prices", None)
        regime = derive_market_regime(
            context,
            benchmark_prices=benchmark_prices,
            max_cash_shift=bounded_cash_shift,
        )
        return {"market_regime": regime}

    return market_regime_node


def derive_market_regime(
    context: PortfolioMarketContext | None,
    *,
    benchmark_prices: list[float] | None,
    max_cash_shift: float = 0.05,
) -> MarketRegime:
    """Convert benchmark trend and recent news into a validated bounded overlay."""

    trend = derive_benchmark_trend_features(benchmark_prices)
    news_score, news_evidence = _news_sentiment(context)
    trend_confidence = min(trend.observations / 60.0, 1.0)
    news_confidence = 0.35 if news_evidence else 0.0
    confidence = _clamp(0.75 * trend_confidence + news_confidence, 0.0, 1.0)
    combined = _clamp(0.75 * trend.trend_score + 0.25 * news_score, -1.0, 1.0)

    if combined >= 0.20:
        label = MarketRegimeLabel.RISK_ON
    elif combined <= -0.20:
        label = MarketRegimeLabel.RISK_OFF
    else:
        label = MarketRegimeLabel.NEUTRAL

    effective_score = combined * confidence if confidence >= 0.25 else 0.0
    cash_adjustment = -effective_score * _clamp(float(max_cash_shift), 0.0, 0.05)
    cash_adjustment = _clamp(cash_adjustment, -0.05, 0.05)
    implications = []
    if cash_adjustment > 0.001:
        implications.append(
            f"Increase cash target by up to {cash_adjustment:.2%} before hard constraints."
        )
    elif cash_adjustment < -0.001:
        implications.append(
            f"Permit cash target to fall by up to {abs(cash_adjustment):.2%} before hard constraints."
        )
    else:
        implications.append("No material market-driven allocation adjustment.")

    evidence = []
    if trend.observations:
        evidence.append(
            f"Benchmark trend score {trend.trend_score:+.2f} from {trend.observations} closes."
        )
    evidence.extend(news_evidence)
    benchmark_symbol = getattr(context, "benchmark_symbol", None) if context else None
    return MarketRegime(
        label=label,
        confidence=confidence,
        benchmark_symbol=benchmark_symbol,
        benchmark_trend=trend,
        news_sentiment_score=news_score,
        summary=f"{label.value.replace('_', '-').title()} market regime with {confidence:.0%} confidence.",
        allocation_implications=implications,
        overlay=MarketOverlay(
            risk_adjustment=effective_score,
            cash_weight_adjustment=cash_adjustment,
        ),
        evidence=evidence,
    )


def _news_sentiment(context: PortfolioMarketContext | None) -> tuple[float, list[str]]:
    if context is None:
        return 0.0, []
    text = " ".join(
        str(value or "")
        for value in (
            getattr(context, "global_news", ""),
            getattr(context, "benchmark_news", ""),
        )
    ).lower()
    if not text.strip():
        return 0.0, []
    positive = len(
        re.findall(
            r"\b(rally|rallies|growth|bullish|optimis\w*|strong|upgrade|beat)\b", text
        )
    )
    negative = len(
        re.findall(
            r"\b(crash|selloff|recession|bearish|fear|weak|downgrade|miss|war|crisis)\b",
            text,
        )
    )
    total = positive + negative
    if not total:
        return 0.0, []
    score = _clamp((positive - negative) / total, -1.0, 1.0)
    return score, [
        f"Recent market/news directional score {score:+.2f} from {total} recognized terms."
    ]


def _period_return(prices: list[float], periods: int) -> float | None:
    if len(prices) <= periods:
        return None
    return prices[-1] / prices[-(periods + 1)] - 1.0


def _sma_gap(prices: list[float], periods: int) -> float | None:
    if len(prices) < periods:
        return None
    average = sum(prices[-periods:]) / periods
    return prices[-1] / average - 1.0 if average else None


def _scaled(value: float | None, scale: float) -> float | None:
    return None if value is None else _clamp(value / scale, -1.0, 1.0)


def _positive_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric > 0 and isfinite(numeric)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
