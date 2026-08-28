"""
test_news_filter.py

Unit tests for the Forex Factory economic news filter and blackout window calculator.
"""

import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.news_filter import (
    fetch_calendar, 
    is_news_blackout, 
    get_upcoming_high_impact_events,
    _parse_currencies
)


def test_currency_parsing():
    assert _parse_currencies("EURUSD") == ["EUR", "USD"]
    assert _parse_currencies("GBPUSD") == ["GBP", "USD"]
    assert _parse_currencies("AUDUSD") == ["AUD", "USD"]
    print("[PASS] Currency parsing test passed!")


def test_calendar_fetching():
    events = fetch_calendar()
    assert isinstance(events, list), "Calendar must return a list"
    assert len(events) > 0, "Should have fetched calendar events"
    high_impact = [e for e in events if e.get("impact") == "High"]
    assert len(high_impact) > 0, "Should contain High-Impact events"
    print(f"[PASS] Calendar fetching passed ({len(events)} total events, {len(high_impact)} High-Impact)!")


def test_blackout_logic():
    now = datetime.now(timezone.utc)
    # Test synthetic event in 15 minutes
    fake_calendar = [
        {
            "title": "US CPI m/m",
            "country": "USD",
            "impact": "High",
            "date": (now + timedelta(minutes=15)).isoformat(),
        }
    ]

    # Temporarily test with fake event
    from src.news_filter import is_news_blackout
    import src.news_filter as nf
    orig_fetch = nf.fetch_calendar
    nf.fetch_calendar = lambda force_refresh=False: fake_calendar

    try:
        # EURUSD should be blocked (contains USD)
        is_blocked, reason = is_news_blackout("EURUSD", target_time=now, buffer_before_mins=30, buffer_after_mins=30)
        assert is_blocked is True, "EURUSD should be blocked before US CPI"
        assert "US CPI m/m" in reason, "Reason should mention US CPI"

        # Time outside window (e.g. 45 mins before)
        is_blocked_outside, _ = is_news_blackout("EURUSD", target_time=now - timedelta(minutes=45), buffer_before_mins=30, buffer_after_mins=30)
        assert is_blocked_outside is False, "45 mins before should not be blocked"
        print("[PASS] Blackout window calculation test passed!")
    finally:
        nf.fetch_calendar = orig_fetch


if __name__ == "__main__":
    test_currency_parsing()
    test_calendar_fetching()
    test_blackout_logic()
    print("\nALL FOREX FACTORY NEWS FILTER TESTS PASSED!")
