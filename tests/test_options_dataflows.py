import json
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.interface import get_category_for_method, route_to_vendor
from tradingagents.dataflows.yfinance_options import (
    black_scholes_greeks,
    days_to_expiry,
    lookup_option_contract,
    retrieve_option_greeks,
    retrieve_options_chain,
)
from tradingagents.agents.utils.options_tools import get_options_chain


class FakeTicker:
    options = ["2026-06-19"]
    fast_info = {"last_price": 105.0}

    def __init__(self, symbol):
        self.symbol = symbol

    def option_chain(self, expiry):
        assert expiry == "2026-06-19"
        calls = pd.DataFrame(
            [
                {
                    "contractSymbol": "AAPL260619C00100000",
                    "strike": 100.0,
                    "lastPrice": 8.5,
                    "bid": 8.4,
                    "ask": 8.6,
                    "impliedVolatility": 0.25,
                    "volume": 100,
                    "openInterest": 200,
                },
                {
                    "contractSymbol": "AAPL260619C00110000",
                    "strike": 110.0,
                    "lastPrice": 4.2,
                    "bid": 4.1,
                    "ask": 4.3,
                    "impliedVolatility": 0.28,
                    "volume": 80,
                    "openInterest": 120,
                },
            ]
        )
        puts = pd.DataFrame(
            [
                {
                    "contractSymbol": "AAPL260619P00100000",
                    "strike": 100.0,
                    "lastPrice": 3.2,
                    "bid": 3.1,
                    "ask": 3.3,
                    "impliedVolatility": 0.27,
                    "volume": 50,
                    "openInterest": 90,
                    "delta": -0.40,
                }
            ]
        )
        return SimpleNamespace(calls=calls, puts=puts)

    def history(self, period="5d"):
        return pd.DataFrame({"Close": [104.0, 105.0]})


@pytest.fixture
def fake_yfinance(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.dataflows.yfinance_options.yf.Ticker",
        FakeTicker,
    )


@pytest.mark.unit
def test_retrieve_options_chain_enriches_rows(fake_yfinance):
    chain = retrieve_options_chain(
        "aapl",
        expiry="2026-06-19",
        curr_date="2026-05-28",
    )

    assert chain["underlying"] == "AAPL"
    assert chain["underlying_price"] == pytest.approx(105.0)
    assert chain["days_to_expiry"] == 22
    assert chain["calls"][0]["right"] == "C"
    assert chain["calls"][0]["moneyness"] == pytest.approx(1.05)
    assert chain["puts"][0]["right"] == "P"


@pytest.mark.unit
def test_lookup_option_contract_matches_contract_symbol(fake_yfinance):
    contract = lookup_option_contract(
        "AAPL",
        "2026-06-19",
        100,
        "call",
        curr_date="2026-05-28",
        contract_symbol="AAPL260619C00100000",
    )

    assert contract["symbol"] == "AAPL260619C00100000"
    assert contract["strike"] == pytest.approx(100.0)
    assert contract["impliedVolatility"] == pytest.approx(0.25)
    assert contract["days_to_expiry"] == 22


@pytest.mark.unit
def test_black_scholes_fallback_returns_all_greeks():
    greeks = black_scholes_greeks(
        underlying_price=105,
        strike=100,
        days_to_expiry=30,
        implied_volatility=0.25,
        right="C",
    )

    assert set(greeks) == {"delta", "gamma", "theta", "vega", "rho"}
    assert greeks["delta"] > 0
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0


@pytest.mark.unit
def test_retrieve_option_greeks_preserves_provider_values_and_fills_missing(
    fake_yfinance,
):
    result = retrieve_option_greeks(
        "AAPL",
        "2026-06-19",
        100,
        "put",
        curr_date="2026-05-28",
        contract_symbol="AAPL260619P00100000",
    )

    assert result["greeks"]["delta"] == pytest.approx(-0.40)
    assert result["greek_sources"]["delta"] == "provider"
    assert result["greeks"]["gamma"] is not None
    assert result["greek_sources"]["gamma"] == "fallback_black_scholes"
    assert result["days_to_expiry"] == 22


@pytest.mark.unit
def test_route_to_vendor_registers_options_data(fake_yfinance):
    payload = route_to_vendor(
        "get_option_greeks",
        "AAPL",
        "2026-06-19",
        100,
        "C",
        "2026-05-28",
    )
    parsed = json.loads(payload)

    assert get_category_for_method("get_option_greeks") == "options_data"
    assert parsed["underlying"] == "AAPL"
    assert parsed["greek_sources"]["delta"] == "fallback_black_scholes"


@pytest.mark.unit
def test_days_to_expiry_never_negative():
    assert days_to_expiry("2026-05-01", "2026-05-28") == 0


@pytest.mark.unit
def test_options_tool_returns_unavailable_payload_when_vendor_fails(monkeypatch):
    def fail_route(*args, **kwargs):
        raise RuntimeError("yfinance options chain unavailable")

    monkeypatch.setattr(
        "tradingagents.agents.utils.options_tools.route_to_vendor",
        fail_route,
    )

    payload = get_options_chain.invoke(
        {
            "underlying": "AAPL",
            "expiry": "2026-06-19",
            "curr_date": "2026-05-28",
        }
    )
    parsed = json.loads(payload)

    assert parsed["status"] == "unavailable"
    assert parsed["underlying"] == "AAPL"
    assert "Continue the portfolio analysis" in parsed["message"]
