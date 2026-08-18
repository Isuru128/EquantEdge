"""
run_live.py

Main dry-run loop for the Inside Bar Breakout strategy on XAUUSD M1.

What this script does:
  1. Connects to a running MT5 terminal (demo account).
  2. Every M1 bar close, fetches the latest candles.
  3. Runs the strategy to check for a signal on the last closed bar.
  4. ONLY acts on signals during the three configured New York (UTC-4) session
     windows; all other times are skipped silently (one log line per minute).
  5. Prints any signal with entry levels (mother bar high/low).
  6. Places orders via execution.py (DRY_RUN = True by default → no real orders).

Usage:
    python -m src.run_live
"""

import time
from zoneinfo import ZoneInfo
import MetaTrader5 as mt5
from datetime import datetime, timezone

import os
import sys

# Allow running directly as a script (python src/run_live.py) or as a module (python -m src.run_live)
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.connect import connect, get_candles
    from src.strategy import get_latest_signal
    from src.execution import place_order, set_dry_run, DRY_RUN
    import src.execution as execution
    import src.db as db
else:
    from .connect import connect, get_candles
    from .strategy import get_latest_signal
    from .execution import place_order, set_dry_run, DRY_RUN
    from . import execution
    from . import db

# ── Symbol / Timeframe ────────────────────────────────────────────────────────
SYMBOL      = "XAUUSD"
TIMEFRAME   = mt5.TIMEFRAME_M1
NUM_CANDLES = 200          # must be > EMA period (50) + buffer
POLL_SECS   = 15           # poll every 15 s; M1 bars close every 60 s
RISK_PCT    = 1.0          # % of account balance to risk per trade
SL_PIPS     = 150          # stop-loss distance in pips  (tune for Gold)
TP_PIPS     = 300          # take-profit distance in pips (2:1 R:R)
# ─────────────────────────────────────────────────────────────────────────────

# ── Trading Sessions — New York time (UTC-4) ──────────────────────────────────
# Format: (start_hour, start_min, end_hour, end_min)  — 24-hour clock.
# A window whose end time is 00:00 means "until midnight" (23:59:59).
TRADING_SESSIONS_NY = [
    (20,  0,  0,  0),   # Asia     : 20:00 – 00:00  NY (midnight)
    ( 2,  0,  5,  0),   # London   : 02:00 – 05:00  NY
    ( 7,  0, 11,  0),   # New York : 07:00 – 11:00  NY
]
# America/New_York automatically switches between EDT (UTC-4) and EST (UTC-5)
_NY_TZ = ZoneInfo("America/New_York")
# ─────────────────────────────────────────────────────────────────────────────


def _now_ny() -> datetime:
    """Return the current wall-clock datetime in New York (auto EDT/EST), timezone-naive."""
    return datetime.now(tz=_NY_TZ).replace(tzinfo=None)


def is_in_trading_session() -> bool:
    """
    Return True if the current New York time falls strictly inside one of the
    3 configured trading sessions:
      • Asia     : 20:00 – 00:00 NY
      • London   : 02:00 – 05:00 NY
      • New York : 07:00 – 11:00 NY
    All other hours are Off-Hours and no trades will be executed.
    """
    now      = _now_ny()
    now_mins = now.hour * 60 + now.minute   # minutes elapsed since midnight

    for start_h, start_m, end_h, end_m in TRADING_SESSIONS_NY:
        start_mins = start_h * 60 + start_m
        end_mins   = 24 * 60 if (end_h == 0 and end_m == 0) else (end_h * 60 + end_m)

        if start_mins <= now_mins < end_mins:
            return True

    return False


def get_current_session_name() -> str:
    """Return the name of the current active trading session: Asia, London, New York, or Off-Hours."""
    now = _now_ny()
    now_mins = now.hour * 60 + now.minute
    session_names = ["Asia", "London", "New York"]
    for idx, (start_h, start_m, end_h, end_m) in enumerate(TRADING_SESSIONS_NY):
        start_mins = start_h * 60 + start_m
        end_mins   = 24 * 60 if (end_h == 0 and end_m == 0) else (end_h * 60 + end_m)
        if start_mins <= now_mins < end_mins:
            return session_names[idx]
    return "Off-Hours"


def _session_status_line() -> str:
    """One-liner showing current NY time and session status."""
    ny  = _now_ny()
    tag = "✅ IN SESSION" if is_in_trading_session() else "⏸  OUT OF SESSION"
    return f"NY {ny.strftime('%H:%M:%S')}  {tag}"


def print_signal(sig: dict) -> None:
    """Pretty-print a signal dict to the console."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if sig["signal"] == 0:
        print(f"[{ts}]  No signal  |  Trend: {sig.get('trend', '?')}  "
              f"|  EMA-50: {sig.get('ema_50', '?')}  "
              f"|  Inside bar: {sig.get('inside_bar', False)}")
        return

    direction = "🟢 BUY " if sig["signal"] == 1 else "🔴 SELL"
    entry_key = "mother_high" if sig["signal"] == 1 else "mother_low"
    entry_lvl = sig.get(entry_key)

    print(
        f"\n{'='*60}\n"
        f"  {direction} SIGNAL DETECTED\n"
        f"  Time        : {sig['datetime']}\n"
        f"  Close       : {sig['close']}\n"
        f"  EMA-50      : {sig['ema_50']}\n"
        f"  Trend       : {sig['trend']}\n"
        f"  Entry level : {entry_lvl}  ({entry_key})\n"
        f"  Mother high : {sig['mother_high']}\n"
        f"  Mother low  : {sig['mother_low']}\n"
        f"{'='*60}\n"
    )


def main() -> None:
    if "--dry-run" in sys.argv:
        set_dry_run(True)
    else:
        set_dry_run(False)

    print("EquantEdge — Inside Bar Breakout Strategy")
    print(f"Mode   : {'[DRY RUN — simulation only]' if execution.DRY_RUN else '⚡ REALTIME LIVE TRADING (Sending Real Orders to MT5 Demo)'}")
    print(f"Symbol : {SYMBOL}  |  Timeframe: M1  |  Poll: every {POLL_SECS}s")
    ny_now      = datetime.now(tz=_NY_TZ)
    utc_offset  = int(ny_now.utcoffset().total_seconds() // 3600)   # e.g. -4 or -5
    tz_abbr     = ny_now.strftime("%Z")                             # e.g. "EDT" or "EST"
    print(f"Timezone : America/New_York  ({tz_abbr}, UTC{utc_offset:+d})")
    print("Sessions :")
    for sh, sm, eh, em in TRADING_SESSIONS_NY:
        end_label = "00:00 (midnight)" if (eh == 0 and em == 0) else f"{eh:02d}:{em:02d}"
        print(f"  •  {sh:02d}:{sm:02d} – {end_label}")
    print("Press Ctrl+C to stop.\n")

    account = connect()
    open_ticket      = None   # currently tracked open position
    last_seen_bar    = None   # last bar datetime we processed
    last_out_log_min = None   # throttle "outside session" log to once/minute

    try:
        while True:
            # ── Check if tracked open position has closed in MT5 ───────────
            if open_ticket is not None and open_ticket > 0:
                pos = mt5.positions_get(ticket=open_ticket)
                if not pos:
                    # Position is closed; fetch deal outcome from MT5 history
                    deals = mt5.history_deals_get(position=open_ticket)
                    if deals and len(deals) >= 2:
                        exit_deal = deals[-1]
                        profit = sum(d.profit for d in deals)
                        reason = "TP_HIT" if profit > 0 else "SL_HIT"
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]  "
                              f"↳ Position #{open_ticket} closed! Realized P&L: ${profit:+.2f} ({reason})")
                        try:
                            db.log_trade_close(
                                ticket=open_ticket,
                                close_price=exit_deal.price,
                                close_time=datetime.fromtimestamp(exit_deal.time, tz=timezone.utc),
                                profit_usd=profit,
                                close_reason=reason,
                            )
                        except Exception:
                            pass
                    open_ticket = None

            # ── Session gate ───────────────────────────────────────────────
            if not is_in_trading_session():
                now_minute = _now_ny().strftime("%H:%M")
                if now_minute != last_out_log_min:
                    last_out_log_min = now_minute
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]  "
                          f"{_session_status_line()} — waiting for next window.")
                time.sleep(POLL_SECS)
                continue

            # Back inside a session — reset the out-of-session log throttle
            last_out_log_min = None

            # ── Fetch candles and evaluate strategy ────────────────────────
            df  = get_candles(SYMBOL, TIMEFRAME, NUM_CANDLES)
            sig = get_latest_signal(df)

            bar_time = sig.get("datetime")
            if bar_time != last_seen_bar:
                last_seen_bar = bar_time
                print(f"[{datetime.now().strftime('%H:%M:%S')}]  {_session_status_line()}")
                print_signal(sig)

                # ── Place order on signal ──────────────────────────────────
                if sig["signal"] != 0 and open_ticket is None:
                    sess_name = get_current_session_name()
                    ticket = place_order(
                        symbol          = SYMBOL,
                        signal          = sig["signal"],
                        entry_price     = sig["mother_high"] if sig["signal"] == 1 else sig["mother_low"],
                        sl_pips         = SL_PIPS,
                        tp_pips         = TP_PIPS,
                        account_balance = account.balance,
                        risk_pct        = RISK_PCT,
                        session         = sess_name,
                        ema_50          = sig.get("ema_50"),
                        mother_high     = sig.get("mother_high"),
                        mother_low      = sig.get("mother_low"),
                    )
                    if ticket and ticket > 0:
                        open_ticket = ticket
                        print(f"  ↳ Order placed — ticket #{ticket}")

            time.sleep(POLL_SECS)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mt5.shutdown()
        print("MT5 connection closed.")


if __name__ == "__main__":
    main()
