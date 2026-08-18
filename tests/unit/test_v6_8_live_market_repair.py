from __future__ import annotations

import io
import json

from portfolio_bot.research_v6_8 import live_market_repair
from portfolio_bot.research_v6_8.live_market_repair import _is_rate_limit


def test_rate_limit_detection_is_case_insensitive() -> None:
    assert _is_rate_limit(RuntimeError("Too many requests, please try again later"))
    assert not _is_rate_limit(RuntimeError("instrument has no data"))


def test_throttle_limit_reader_uses_local_workspace_endpoint(monkeypatch) -> None:
    payload = {
        "daily_request_max": 10000,
        "daily_request_remaining": 0,
        "request_max": 5,
        "request_remaining": 5,
        "daily_volume_max": 5000000,
        "daily_volume_remaining": 4614286,
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        live_market_repair.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(json.dumps(payload).encode()),
    )
    assert live_market_repair.read_desktop_throttle_limits()[
        "daily_request_remaining"
    ] == 0
