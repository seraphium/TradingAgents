from datetime import date

import pytest
from pydantic import ValidationError

from tradingagents.portfolio.schemas import (
    AssetType,
    OptionRight,
    PortfolioConstraints,
    PortfolioInstrument,
    PortfolioPosition,
    PortfolioRequest,
)


def stock_position(symbol: str, weight: float) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol=symbol, asset_type=AssetType.STOCK),
        current_weight=weight,
        quantity=10,
    )


def cash_position(weight: float) -> PortfolioPosition:
    return PortfolioPosition(
        instrument=PortfolioInstrument(symbol="CASH", asset_type=AssetType.CASH),
        current_weight=weight,
        market_value=1000,
    )


@pytest.mark.unit
class TestPortfolioInstrument:
    def test_stock_symbol_is_normalized(self):
        instrument = PortfolioInstrument(symbol=" aapl ", asset_type="Stock")
        assert instrument.symbol == "AAPL"
        assert instrument.asset_type == AssetType.STOCK

    def test_stock_rejects_option_fields(self):
        with pytest.raises(ValidationError, match="stock instruments cannot include option fields"):
            PortfolioInstrument(
                symbol="AAPL",
                asset_type="stock",
                underlying="AAPL",
            )

    def test_option_requires_contract_terms(self):
        with pytest.raises(ValidationError, match="option instruments require"):
            PortfolioInstrument(symbol="AAPL250620C00200000", asset_type="option")

    def test_option_accepts_required_contract_terms(self):
        instrument = PortfolioInstrument(
            symbol="aapl250620c00200000",
            asset_type=AssetType.OPTION,
            underlying="aapl",
            expiry="2025-06-20",
            strike=200,
            right="call",
        )

        assert instrument.symbol == "AAPL250620C00200000"
        assert instrument.underlying == "AAPL"
        assert instrument.expiry == date(2025, 6, 20)
        assert instrument.strike == 200
        assert instrument.right == OptionRight.CALL

    def test_cash_must_use_cash_symbol(self):
        with pytest.raises(ValidationError, match="cash instruments must use symbol CASH"):
            PortfolioInstrument(symbol="USD", asset_type="cash")


@pytest.mark.unit
class TestPortfolioPosition:
    def test_non_cash_position_requires_quantity_or_market_value(self):
        with pytest.raises(ValidationError, match="non-cash positions require"):
            PortfolioPosition(
                instrument=PortfolioInstrument(symbol="MSFT", asset_type="stock"),
                current_weight=0.25,
            )

    def test_target_min_cannot_exceed_target_max(self):
        with pytest.raises(ValidationError, match="target_min_weight cannot exceed"):
            PortfolioPosition(
                instrument=PortfolioInstrument(symbol="MSFT", asset_type="stock"),
                current_weight=0.25,
                quantity=5,
                target_min_weight=0.3,
                target_max_weight=0.2,
            )

    def test_cash_rejects_quantity(self):
        with pytest.raises(ValidationError, match="cash positions should use"):
            PortfolioPosition(
                instrument=PortfolioInstrument(symbol="CASH", asset_type="cash"),
                current_weight=0.10,
                quantity=1,
            )


@pytest.mark.unit
class TestPortfolioRequest:
    def test_valid_portfolio_request(self):
        request = PortfolioRequest(
            positions=[
                stock_position("AAPL", 0.30),
                stock_position("MSFT", 0.25),
                stock_position("NVDA", 0.25),
                cash_position(0.20),
            ],
            trade_date="2026-05-27",
            base_currency="usd",
        )

        assert request.trade_date == date(2026, 5, 27)
        assert request.base_currency == "USD"
        assert sum(position.current_weight for position in request.positions) == pytest.approx(1.0)

    def test_rejects_weights_outside_tolerance(self):
        with pytest.raises(ValidationError, match="portfolio weights must sum to 1.0"):
            PortfolioRequest(
                positions=[
                    stock_position("AAPL", 0.50),
                    stock_position("MSFT", 0.40),
                ],
                trade_date="2026-05-27",
            )

    def test_rejects_duplicate_symbols(self):
        with pytest.raises(ValidationError, match="duplicate portfolio symbols"):
            PortfolioRequest(
                positions=[
                    stock_position("AAPL", 0.50),
                    stock_position("AAPL", 0.50),
                ],
                trade_date="2026-05-27",
            )

    def test_enforces_max_single_position_weight(self):
        with pytest.raises(ValidationError, match="max_single_position_weight"):
            PortfolioRequest(
                positions=[
                    stock_position("AAPL", 0.70),
                    stock_position("MSFT", 0.20),
                    cash_position(0.10),
                ],
                trade_date="2026-05-27",
                constraints=PortfolioConstraints(max_single_position_weight=0.60),
            )

    def test_enforces_options_weight_limit(self):
        option_position = PortfolioPosition(
            instrument=PortfolioInstrument(
                symbol="AAPL250620C00200000",
                asset_type="option",
                underlying="AAPL",
                expiry="2025-06-20",
                strike=200,
                right="C",
            ),
            current_weight=0.20,
            quantity=2,
        )

        with pytest.raises(ValidationError, match="options weight exceeds"):
            PortfolioRequest(
                positions=[
                    stock_position("AAPL", 0.70),
                    option_position,
                    cash_position(0.10),
                ],
                trade_date="2026-05-27",
                constraints=PortfolioConstraints(max_options_weight=0.10),
            )

    def test_enforces_cash_limits(self):
        with pytest.raises(ValidationError, match="cash weight is below"):
            PortfolioRequest(
                positions=[
                    stock_position("AAPL", 0.95),
                    cash_position(0.05),
                ],
                trade_date="2026-05-27",
                constraints=PortfolioConstraints(min_cash_weight=0.10),
            )
