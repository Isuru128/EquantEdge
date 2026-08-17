"""
db.py

Supabase Database Integration for EquantEdge.
Persists trade execution records, strategy indicators snapshot, and realized P&L.

Design:
  - Non-blocking / Fail-safe: Database errors log a warning without crashing MT5 trading.
  - Auto-configures via environment variables (SUPABASE_URL, SUPABASE_KEY).
"""

import os
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
TABLE_NAME = os.getenv("SUPABASE_TRADES_TABLE", "trades").strip()

_client = None
_client_initialized = False


def get_supabase_client():
    """Return an active Supabase client instance or None if unconfigured."""
    global _client, _client_initialized
    if _client_initialized:
        return _client

    _client_initialized = True
    if not SUPABASE_URL or not SUPABASE_KEY or "your-project-id" in SUPABASE_URL:
        print("[DB] Supabase credentials not configured in .env (running in offline mode).")
        _client = None
        return None

    try:
        from supabase import create_client, Client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[DB] Supabase database connected successfully.")
        return _client
    except Exception as exc:
        print(f"[DB] WARNING: Failed to initialize Supabase client: {exc}")
        _client = None
        return None


def log_trade_open(
    ticket: int,
    symbol: str,
    side: str,                  # 'BUY' or 'SELL'
    lot_size: float,
    open_price: float,
    sl: float | None = None,
    tp: float | None = None,
    session: str | None = None,
    ema_50: float | None = None,
    mother_high: float | None = None,
    mother_low: float | None = None,
    open_time: datetime | None = None,
) -> bool:
    """
    Insert a newly opened trade record with strategy context into Supabase.
    """
    client = get_supabase_client()
    if client is None:
        return False

    if open_time is None:
        open_time = datetime.now(timezone.utc)
    elif open_time.tzinfo is None:
        open_time = open_time.replace(tzinfo=timezone.utc)

    payload = {
        "ticket": ticket,
        "symbol": symbol,
        "side": side.upper(),
        "lot_size": float(lot_size),
        "open_time": open_time.isoformat(),
        "open_price": float(open_price),
        "sl": float(sl) if sl is not None else None,
        "tp": float(tp) if tp is not None else None,
        "session": session,
        "ema_50": float(ema_50) if ema_50 is not None else None,
        "mother_high": float(mother_high) if mother_high is not None else None,
        "mother_low": float(mother_low) if mother_low is not None else None,
    }

    try:
        res = client.table(TABLE_NAME).upsert(payload, on_conflict="ticket").execute()
        print(f"[DB] Logged open trade #{ticket} to Supabase.")
        return True
    except Exception as exc:
        print(f"[DB] WARNING: Could not log open trade #{ticket} to Supabase: {exc}")
        return False


def log_trade_close(
    ticket: int,
    close_price: float,
    close_time: datetime | None = None,
    profit_usd: float | None = None,
    close_reason: str = "CLOSE",
) -> bool:
    """
    Update an existing trade record with exit price, exit time, and realized profit.
    """
    client = get_supabase_client()
    if client is None:
        return False

    if close_time is None:
        close_time = datetime.now(timezone.utc)
    elif close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=timezone.utc)

    payload = {
        "close_price": float(close_price),
        "close_time": close_time.isoformat(),
        "profit_usd": float(profit_usd) if profit_usd is not None else None,
        "close_reason": close_reason,
    }

    try:
        res = client.table(TABLE_NAME).update(payload).eq("ticket", ticket).execute()
        print(f"[DB] Updated closed trade #{ticket} in Supabase (Profit: ${profit_usd:+.2f}).")
        return True
    except Exception as exc:
        print(f"[DB] WARNING: Could not update closed trade #{ticket} in Supabase: {exc}")
        return False


def fetch_trade_history(limit: int = 100) -> list[dict]:
    """
    Retrieve historical trade records ordered by open_time descending.
    """
    client = get_supabase_client()
    if client is None:
        return []

    try:
        res = client.table(TABLE_NAME).select("*").order("open_time", desc=True).limit(limit).execute()
        return res.data if res and hasattr(res, "data") else []
    except Exception as exc:
        print(f"[DB] WARNING: Error fetching trades from Supabase: {exc}")
        return []
