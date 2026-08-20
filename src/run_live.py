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
    from src.execution import place_order, set_dry_run, DRY_RUN, modify_position_sl, _pip_size
    import src.execution as execution
    import src.db as db
else:
    from .connect import connect, get_candles
    from .strategy import get_latest_signal
    from .execution import place_order, set_dry_run, DRY_RUN, modify_position_sl, _pip_size
    from . import execution
    from . import db


# ── Symbols / Timeframe ───────────────────────────────────────────────────────
DEFAULT_SYMBOLS = ["GBPUSD", "XAUUSD", "EURUSD"]
TIMEFRAME       = mt5.TIMEFRAME_M1   # 1-minute timeframe
NUM_CANDLES     = 200                # must be > EMA period (50) + buffer
POLL_SECS       = 10                 # poll every 10 s; M1 bars close every 60 s
RISK_PCT        = 1.0                # % of account balance to risk per trade
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


def _resolve_symbol(symbol: str) -> str:
    """Find matching broker symbol (handles US30 vs US30Cash / DJI / WS30)."""
    info = mt5.symbol_info(symbol)
    if info is not None:
        return symbol
    if "US30" in symbol:
        for alt in ("US30Cash", "DJI", "WS30", "DJ30", "WallStreet30"):
            if mt5.symbol_info(alt) is not None:
                return alt
    return symbol


def print_signal(symbol: str, sig: dict) -> None:
    """Pretty-print a signal dict to the console."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if sig["signal"] == 0:
        ma_val = sig.get("ma") or sig.get("ema_50", "?")
        ml_tag = f" | ML: {sig['xgb_prob']*100:.1f}%" if sig.get("xgb_prob") is not None else ""
        print(f"[{ts}][{symbol:<6}] No signal  |  MA: {ma_val:<8} |  Close: {sig.get('close', '?')}{ml_tag}")
        return

    direction = "🟢 BUY " if sig["signal"] == 1 else "🔴 SELL"

    print(
        f"\n{'='*65}\n"
        f"  {direction} SIGNAL DETECTED for {symbol} (M1) — {sig.get('label', '')}\n"
        f"  Time          : {sig['datetime']}\n"
        f"  Entry Price   : {sig['entry_price']}\n"
        f"  Stop Loss     : {sig['sl_price']} ({sig.get('sl_pips', 0):.1f} pips/pts)\n"
        f"  Take Profit   : {sig['tp_price']} ({sig.get('tp_pips', 0):.1f} pips/pts, 1:2.0 R:R)\n"
        f"  ML Confidence : {sig.get('xgb_prob', 'N/A')}\n"
        f"  Prev Bar H/L  : High={sig.get('prev_high')}, Low={sig.get('prev_low')}, Open={sig.get('prev_open')}\n"
        f"{'='*65}\n"
    )





def main() -> None:
    if "--dry-run" in sys.argv:
        set_dry_run(True)
    else:
        set_dry_run(False)

    symbols = DEFAULT_SYMBOLS
    tf = TIMEFRAME
    for arg in sys.argv:
        if arg.startswith("--symbols="):
            symbols = [s.strip() for s in arg.split("=")[1].split(",") if s.strip()]
        elif arg.startswith("--symbol="):
            symbols = [arg.split("=")[1].strip()]
        elif arg.startswith("--tf="):
            tf_val = int(arg.split("=")[1].strip())
            tf_map = {1: mt5.TIMEFRAME_M1, 3: mt5.TIMEFRAME_M3, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15}
            tf = tf_map.get(tf_val, mt5.TIMEFRAME_M1)


    print("EquantEdge — Multi-Asset Liquidity Sweep Reversal Strategy (with XGBoost ML)")
    print(f"Mode     : {'[DRY RUN — simulation only]' if execution.DRY_RUN else '⚡ REALTIME LIVE TRADING (Sending Real Orders to MT5 Demo)'}")
    print(f"Symbols  : {', '.join(symbols)}  |  Timeframe: M1 (1-minute)  |  Poll: every {POLL_SECS}s")
    ny_now      = datetime.now(tz=_NY_TZ)
    utc_offset  = int(ny_now.utcoffset().total_seconds() // 3600)
    tz_abbr     = ny_now.strftime("%Z")
    print(f"Timezone : America/New_York  ({tz_abbr}, UTC{utc_offset:+d})")
    print("Sessions :")
    for sh, sm, eh, em in TRADING_SESSIONS_NY:
        end_label = "00:00 (midnight)" if (eh == 0 and em == 0) else f"{eh:02d}:{em:02d}"
        print(f"  •  {sh:02d}:{sm:02d} – {end_label}")
    print("Press Ctrl+C to stop.\n")

    account = connect()
    open_tickets     = {}   # symbol -> open ticket
    trade_meta       = {}   # ticket -> {"entry": float, "initial_sl": float, "sig": int, "be_moved": bool, "risk_dist": float}
    last_seen_bars   = {}   # symbol -> last seen bar datetime
    last_out_log_min = None

    try:
        while True:
            # ── Check open positions (Close detection & +1.0R Breakeven Trailing) ──
            for sym, ticket in list(open_tickets.items()):
                if ticket and ticket > 0:
                    pos = mt5.positions_get(ticket=ticket)
                    if not pos:
                        # Position closed
                        deals = mt5.history_deals_get(position=ticket)
                        if deals and len(deals) >= 2:
                            exit_deal = deals[-1]
                            profit = sum(d.profit for d in deals)
                            reason = "TP_HIT" if profit > 0 else "SL_HIT"
                            print(f"[{datetime.now().strftime('%H:%M:%S')}][{sym}] "
                                  f"↳ Position #{ticket} closed! Realized P&L: ${profit:+.2f} ({reason})")
                            try:
                                db.log_trade_close(
                                    ticket=ticket,
                                    close_price=exit_deal.price,
                                    close_time=datetime.fromtimestamp(exit_deal.time, tz=timezone.utc),
                                    profit_usd=profit,
                                    close_reason=reason,
                                )
                            except Exception:
                                pass
                        open_tickets[sym] = None
                        trade_meta.pop(ticket, None)
                    else:
                        # Position is active — check for +1.0R Breakeven Trigger
                        p = pos[0]
                        meta = trade_meta.get(ticket)
                        if meta and not meta.get("be_moved"):
                            sig_side  = meta["sig"]
                            entry     = meta["entry"]
                            risk_dist = meta["risk_dist"]
                            pip       = _pip_size(sym)

                            # BUY: price reached entry + 1R risk distance
                            if sig_side == 1 and p.price_current >= (entry + risk_dist):
                                new_sl = entry + (1.0 * pip)
                                if modify_position_sl(ticket, _resolve_symbol(sym), new_sl):
                                    meta["be_moved"] = True
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}][{sym}] 🎯 +1.0R Reached! Trailed SL to Breakeven ({new_sl})")

                            # SELL: price reached entry - 1R risk distance
                            elif sig_side == -1 and p.price_current <= (entry - risk_dist):
                                new_sl = entry - (1.0 * pip)
                                if modify_position_sl(ticket, _resolve_symbol(sym), new_sl):
                                    meta["be_moved"] = True
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}][{sym}] 🎯 +1.0R Reached! Trailed SL to Breakeven ({new_sl})")

            # ── Session gate ───────────────────────────────────────────────
            if not is_in_trading_session():
                now_minute = _now_ny().strftime("%H:%M")
                if now_minute != last_out_log_min:
                    last_out_log_min = now_minute
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]  "
                          f"{_session_status_line()} — waiting for next window.")
                time.sleep(POLL_SECS)
                continue

            last_out_log_min = None

            # ── Multi-Symbol Candle Evaluation Loop ────────────────────────
            for sym in symbols:
                actual_sym = _resolve_symbol(sym)
                df = get_candles(actual_sym, tf, NUM_CANDLES)
                if df.empty:
                    continue

                sig = get_latest_signal(df, symbol=actual_sym, use_ml=True)
                bar_time = sig.get("datetime")


                if bar_time != last_seen_bars.get(sym):
                    last_seen_bars[sym] = bar_time
                    print_signal(sym, sig)

                    # ── Place order on signal ──────────────────────────────
                    if sig["signal"] != 0 and open_tickets.get(sym) is None:
                        sess_name = get_current_session_name()
                        ticket = place_order(
                            symbol          = actual_sym,
                            signal          = sig["signal"],
                            entry_price     = sig["entry_price"],
                            sl_price        = sig.get("sl_price"),
                            tp_price        = sig.get("tp_price"),
                            sl_pips         = sig.get("sl_pips"),
                            tp_pips         = sig.get("tp_pips"),
                            account_balance = account.balance if account else 10000.0,
                            risk_pct        = RISK_PCT,
                            session         = sess_name,
                            ema_50          = sig.get("ma"),
                            mother_high     = sig.get("prev_high"),
                            mother_low      = sig.get("prev_low"),
                        )
                        if ticket and ticket > 0:
                            open_tickets[sym] = ticket
                            risk_distance = sig.get("risk_distance") or (abs(sig["entry_price"] - (sig.get("sl_price") or sig["entry_price"])))
                            trade_meta[ticket] = {
                                "entry": sig["entry_price"],
                                "initial_sl": sig.get("sl_price"),
                                "sig": sig["signal"],
                                "be_moved": False,
                                "risk_dist": risk_distance,
                            }
                            print(f"  ↳ [{sym}] Order placed — ticket #{ticket} (Risk: {risk_distance:.4f})")

            time.sleep(POLL_SECS)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mt5.shutdown()
        print("MT5 connection closed.")


if __name__ == "__main__":
    main()



