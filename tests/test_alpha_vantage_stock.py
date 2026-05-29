import json

import pytest

from tradingagents.dataflows.alpha_vantage_stock import get_stock


@pytest.mark.unit
def test_alpha_vantage_stock_falls_back_to_raw_daily_when_adjusted_unavailable(
    monkeypatch,
):
    calls = []

    def fake_make_api_request(function_name, params):
        calls.append((function_name, params))
        if function_name == "TIME_SERIES_DAILY_ADJUSTED":
            return json.dumps(
                {
                    "Information": (
                        "Thank you for using Alpha Vantage! This is a premium endpoint."
                    )
                }
            )
        return (
            "timestamp,open,high,low,close,volume\n"
            "2026-05-28,310.68,312.80,309.57,312.51,48220390\n"
            "2026-05-27,308.33,313.26,308.30,310.85,50430919\n"
        )

    monkeypatch.setattr(
        "tradingagents.dataflows.alpha_vantage_stock._make_api_request",
        fake_make_api_request,
    )

    csv = get_stock("AAPL", "2026-05-27", "2026-05-28")

    assert [call[0] for call in calls] == [
        "TIME_SERIES_DAILY_ADJUSTED",
        "TIME_SERIES_DAILY",
    ]
    assert "2026-05-28" in csv
    assert "Information" not in csv


@pytest.mark.unit
def test_alpha_vantage_stock_keeps_adjusted_data_when_available(monkeypatch):
    calls = []

    def fake_make_api_request(function_name, params):
        calls.append(function_name)
        return (
            "timestamp,open,high,low,close,adjusted_close,volume,dividend_amount,split_coefficient\n"
            "2026-05-28,310.68,312.80,309.57,312.51,312.51,48220390,0.0000,1.0\n"
        )

    monkeypatch.setattr(
        "tradingagents.dataflows.alpha_vantage_stock._make_api_request",
        fake_make_api_request,
    )

    csv = get_stock("AAPL", "2026-05-28", "2026-05-28")

    assert calls == ["TIME_SERIES_DAILY_ADJUSTED"]
    assert "adjusted_close" in csv
