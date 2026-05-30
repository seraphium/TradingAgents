from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import reddit


@pytest.mark.unit
def test_reddit_403_returns_unavailable_without_warning(monkeypatch, caplog):
    def blocked(*args, **kwargs):
        raise HTTPError(
            url="https://www.reddit.com/r/stocks/search.json",
            code=403,
            msg="Blocked",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(reddit, "urlopen", blocked)

    result = reddit.fetch_reddit_posts("AAPL", subreddits=("stocks",), timeout=0.1)

    assert result == "<reddit unavailable: HTTP 403 blocked Reddit public search>"
    assert "Reddit fetch failed" not in caplog.text
