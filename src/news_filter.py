"""
news_filter.py

Forex Factory Economic Calendar Integration & High-Impact News Blackout Filter.
Protects EquantEdge from slippage, spread widening, and unpredictable whipsaws
by automatically blocking trade entries during High-Impact news releases.
"""

import os
import json
import csv
import io
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

CACHE_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ff_calendar.json")
CACHE_TTL_HOURS = 6              # Cache calendar on disk for 6 hours to prevent rate limits
DEFAULT_BUFFER_BEFORE_MINS = 30  # Blackout window before high-impact news
DEFAULT_BUFFER_AFTER_MINS  = 30  # Blackout window after high-impact news

_NY_TZ = ZoneInfo("America/New_York")

# Supported Forex Factory endpoint formats
ENDPOINTS = [
    ("xml", "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"),
    ("csv", "https://nfs.faireconomy.media/ff_calendar_thisweek.csv"),
    ("json", "https://nfs.faireconomy.media/ff_calendar_thisweek.json"),
]


def _parse_currencies(symbol: str) -> list[str]:
    """Extract currency codes (e.g. 'EURUSD' -> ['EUR', 'USD'])."""
    sym = symbol.upper().replace("/", "").replace("_", "").replace(".", "")
    if len(sym) >= 6:
        return [sym[:3], sym[3:6]]
    return [sym]


def _parse_xml_events(content_bytes: bytes) -> list[dict]:
    """Parse Forex Factory XML feed."""
    events = []
    root = ET.fromstring(content_bytes)
    for ev in root:
        title = ev.findtext("title", "").strip()
        country = ev.findtext("country", "").strip().upper()
        date_str = ev.findtext("date", "").strip()
        time_str = ev.findtext("time", "").strip()
        impact = ev.findtext("impact", "").strip()

        dt_iso = None
        if date_str and time_str and time_str.lower() not in ("all day", "tentative"):
            try:
                # Forex factory XML time is in New York EDT/EST
                dt_naive = datetime.strptime(f"{date_str} {time_str}", "%m-%d-%Y %I:%M%p")
                dt_ny = dt_naive.replace(tzinfo=_NY_TZ)
                dt_iso = dt_ny.astimezone(timezone.utc).isoformat()
            except Exception:
                pass

        events.append({
            "title": title,
            "country": country,
            "impact": impact,
            "date": dt_iso,
            "raw_date": date_str,
            "raw_time": time_str,
            "forecast": ev.findtext("forecast", "-"),
            "previous": ev.findtext("previous", "-"),
        })
    return events


def _parse_csv_events(content_bytes: bytes) -> list[dict]:
    """Parse Forex Factory CSV feed."""
    events = []
    text = content_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        title = row.get("Title", "").strip()
        country = row.get("Country", "").strip().upper()
        date_str = row.get("Date", "").strip()
        time_str = row.get("Time", "").strip()
        impact = row.get("Impact", "").strip()

        dt_iso = None
        if date_str and time_str and time_str.lower() not in ("all day", "tentative"):
            try:
                dt_naive = datetime.strptime(f"{date_str} {time_str}", "%m-%d-%Y %I:%M%p")
                dt_ny = dt_naive.replace(tzinfo=_NY_TZ)
                dt_iso = dt_ny.astimezone(timezone.utc).isoformat()
            except Exception:
                pass

        events.append({
            "title": title,
            "country": country,
            "impact": impact,
            "date": dt_iso,
            "raw_date": date_str,
            "raw_time": time_str,
            "forecast": row.get("Forecast", "-"),
            "previous": row.get("Previous", "-"),
        })
    return events


def fetch_calendar(force_refresh: bool = False) -> list[dict]:
    """
    Fetch economic calendar with disk caching and automatic multi-format failover.
    """
    os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)

    # 1. Return fresh disk cache if available
    if not force_refresh and os.path.exists(CACHE_FILE_PATH):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE_PATH), tz=timezone.utc)
            if datetime.now(timezone.utc) - mtime < timedelta(hours=CACHE_TTL_HOURS):
                with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        return data
        except Exception:
            pass

    # 2. Try fetching from endpoints
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }

    for fmt, url in ENDPOINTS:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                content = resp.read()
                if fmt == "xml":
                    events = _parse_xml_events(content)
                elif fmt == "csv":
                    events = _parse_csv_events(content)
                elif fmt == "json":
                    events = json.loads(content.decode("utf-8"))

                if events and len(events) > 0:
                    with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
                        json.dump(events, f, indent=2)
                    return events
        except Exception:
            continue

    # 3. Fallback to existing disk cache
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return []


def is_news_blackout(
    symbol: str,
    target_time: datetime | None = None,
    buffer_before_mins: int = DEFAULT_BUFFER_BEFORE_MINS,
    buffer_after_mins: int = DEFAULT_BUFFER_AFTER_MINS,
    impact_levels: tuple[str, ...] = ("High",),
) -> tuple[bool, str | None]:
    """
    Check if current time is within the blackout buffer of a High-Impact news release.
    """
    if target_time is None:
        target_time = datetime.now(timezone.utc)
    elif target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)

    calendar = fetch_calendar()
    if not calendar:
        return False, None

    currencies = _parse_currencies(symbol)

    for event in calendar:
        impact = str(event.get("impact", "")).strip()
        if impact not in impact_levels:
            continue

        country = str(event.get("country", "")).strip().upper()
        if country not in currencies:
            continue

        date_str = event.get("date")
        if not date_str:
            continue

        try:
            event_dt = datetime.fromisoformat(date_str)
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=timezone.utc)
            else:
                event_dt = event_dt.astimezone(timezone.utc)

            window_start = event_dt - timedelta(minutes=buffer_before_mins)
            window_end   = event_dt + timedelta(minutes=buffer_after_mins)

            if window_start <= target_time <= window_end:
                title = event.get("title", "Economic Event")
                mins_diff = int((event_dt - target_time).total_seconds() // 60)

                if mins_diff > 0:
                    timing_str = f"in {mins_diff}m"
                elif mins_diff == 0:
                    timing_str = "NOW"
                else:
                    timing_str = f"{abs(mins_diff)}m ago"

                reason = f"High-Impact [{country}] '{title}' ({timing_str}) — Blackout Active"
                return True, reason
        except Exception:
            continue

    return False, None


def get_upcoming_high_impact_events(
    symbol: str | None = None,
    hours_ahead: int = 24,
) -> list[dict]:
    """
    Return all High-Impact events scheduled in the next `hours_ahead` hours.
    """
    calendar = fetch_calendar()
    if not calendar:
        return []

    now_utc = datetime.now(timezone.utc)
    cutoff_utc = now_utc + timedelta(hours=hours_ahead)
    currencies = _parse_currencies(symbol) if symbol else None

    upcoming = []
    for event in calendar:
        if str(event.get("impact", "")).strip() != "High":
            continue

        country = str(event.get("country", "")).strip().upper()
        if currencies and country not in currencies:
            continue

        date_str = event.get("date")
        if not date_str:
            continue

        try:
            event_dt = datetime.fromisoformat(date_str)
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=timezone.utc)
            else:
                event_dt = event_dt.astimezone(timezone.utc)

            if now_utc <= event_dt <= cutoff_utc:
                upcoming.append({
                    "title": event.get("title"),
                    "country": country,
                    "datetime_utc": event_dt,
                    "date_str": date_str,
                    "forecast": event.get("forecast", "-"),
                    "previous": event.get("previous", "-"),
                    "minutes_away": int((event_dt - now_utc).total_seconds() // 60),
                })
        except Exception:
            continue

    upcoming.sort(key=lambda x: x["datetime_utc"])
    return upcoming
