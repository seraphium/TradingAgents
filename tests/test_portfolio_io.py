import json
from datetime import date

import pytest

from tradingagents.portfolio import (
    AssetType,
    OptionRight,
    PortfolioParseError,
    load_portfolio_file,
    normalize_portfolio_symbol,
    parse_portfolio_csv,
    parse_portfolio_json,
    parse_portfolio_json_payload,
)


@pytest.mark.unit
def test_csv_portfolio_parses_flat_positions(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text(
        "\n".join(
            [
                "symbol,asset_type,current_weight,quantity,expiry,strike,right",
                " aapl ,stock,0.30,25,,,",
                "MSFT,stock,0.25,12,,,",
                "NVDA,stock,0.20,10,,,",
                "AAPL250620C00200000,option,0.05,2,2025-06-20,200,C",
                "CASH,cash,0.20,,,,",
            ]
        ),
        encoding="utf-8",
    )

    request = parse_portfolio_csv(path, trade_date="2026-05-27")

    assert request.trade_date == date(2026, 5, 27)
    assert request.positions[0].instrument.symbol == "AAPL"
    assert request.positions[0].instrument.asset_type == AssetType.STOCK
    assert request.positions[3].instrument.asset_type == AssetType.OPTION
    assert request.positions[3].instrument.underlying == "AAPL"
    assert request.positions[3].instrument.right == OptionRight.CALL
    assert request.positions[4].instrument.asset_type == AssetType.CASH
    assert sum(position.current_weight for position in request.positions) == pytest.approx(1.0)


@pytest.mark.unit
def test_csv_infers_occ_option_terms_when_fields_are_missing(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text(
        "\n".join(
            [
                "symbol,asset_type,current_weight,quantity",
                "AAPL250620C00200000,option,0.10,2",
                "CASH,cash,0.90,",
            ]
        ),
        encoding="utf-8",
    )

    request = parse_portfolio_csv(path, trade_date="2026-05-27")
    option = request.positions[0].instrument

    assert option.underlying == "AAPL"
    assert option.expiry == date(2025, 6, 20)
    assert option.strike == 200
    assert option.right == OptionRight.CALL


@pytest.mark.unit
def test_csv_requires_trade_date(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text("symbol,asset_type,current_weight\nCASH,cash,1.0\n", encoding="utf-8")

    with pytest.raises(PortfolioParseError, match="trade_date is required"):
        parse_portfolio_csv(path, trade_date=None)


@pytest.mark.unit
def test_csv_invalid_weight_reports_row_number(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text(
        "symbol,asset_type,current_weight,quantity\nAAPL,stock,not-a-number,10\n",
        encoding="utf-8",
    )

    with pytest.raises(PortfolioParseError, match="row 2"):
        parse_portfolio_csv(path, trade_date="2026-05-27")


@pytest.mark.unit
def test_json_portfolio_parses_full_request_with_flat_positions(tmp_path):
    path = tmp_path / "portfolio.json"
    path.write_text(
        json.dumps(
            {
                "trade_date": "2026-05-27",
                "base_currency": "usd",
                "constraints": {"max_options_weight": 0.10},
                "positions": [
                    {
                        "symbol": "brk.b",
                        "asset_type": "stock",
                        "current_weight": "30%",
                        "quantity": 5,
                    },
                    {
                        "symbol": "AAPL250620C00200000",
                        "asset_type": "option",
                        "current_weight": 0.05,
                        "quantity": 2,
                        "underlying": "aapl",
                        "expiry": "2025-06-20",
                        "strike": 200,
                        "right": "call",
                        "contract_symbol": "aapl250620c00200000",
                    },
                    {"symbol": "CASH", "asset_type": "cash", "current_weight": 0.65},
                ],
            }
        ),
        encoding="utf-8",
    )

    request = parse_portfolio_json(path)

    assert request.base_currency == "USD"
    assert request.constraints.max_options_weight == 0.10
    assert request.positions[0].instrument.symbol == "BRK.B"
    assert request.positions[0].current_weight == pytest.approx(0.30)
    assert request.positions[1].instrument.contract_symbol == "AAPL250620C00200000"


@pytest.mark.unit
def test_json_payload_parses_structured_positions():
    request = parse_portfolio_json_payload(
        {
            "trade_date": "2026-05-27",
            "positions": [
                {
                    "instrument": {"symbol": "MSFT", "asset_type": "stock"},
                    "current_weight": 0.40,
                    "quantity": 12,
                },
                {
                    "instrument": {"symbol": "CASH", "asset_type": "cash"},
                    "current_weight": 0.60,
                },
            ],
        }
    )

    assert request.positions[0].instrument.symbol == "MSFT"
    assert request.positions[1].instrument.asset_type == AssetType.CASH


@pytest.mark.unit
def test_json_position_list_uses_supplied_metadata(tmp_path):
    path = tmp_path / "portfolio.json"
    path.write_text(
        json.dumps(
            [
                {
                    "symbol": "0700.hk",
                    "asset_type": "stock",
                    "current_weight": 0.25,
                    "quantity": 100,
                },
                {"symbol": "CASH", "asset_type": "cash", "current_weight": 0.75},
            ]
        ),
        encoding="utf-8",
    )

    request = parse_portfolio_json(path, trade_date="2026-05-27", base_currency="hkd")

    assert request.base_currency == "HKD"
    assert request.positions[0].instrument.symbol == "0700.HK"


@pytest.mark.unit
def test_load_portfolio_file_dispatches_by_extension(tmp_path):
    path = tmp_path / "portfolio.json"
    path.write_text(
        json.dumps(
            {
                "trade_date": "2026-05-27",
                "positions": [{"symbol": "CASH", "asset_type": "cash", "current_weight": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    assert load_portfolio_file(path).positions[0].instrument.symbol == "CASH"


@pytest.mark.unit
def test_load_portfolio_file_rejects_unknown_extension(tmp_path):
    path = tmp_path / "portfolio.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(PortfolioParseError, match="unsupported portfolio file type"):
        load_portfolio_file(path)


@pytest.mark.unit
def test_normalize_portfolio_symbol_preserves_exchange_suffix():
    assert normalize_portfolio_symbol(" cnc.to ") == "CNC.TO"
