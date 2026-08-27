"""
run_live.py

Live automated execution engine for the 15M Fair Value Gap (FVG) + 1M Market Structure Shift (MSS) Strategy.

Assets: EURUSD, GBPUSD, AUDUSD.
Timeframe: 15-Minute HTF Trend & FVG + 1-Minute LTF Market Structure Shift.
Session Restriction: Trades BLOCKED between 14:00 - 20:00 New York time (UTC-4 / UTC-5).
Risk: 1.0% per trade (fixed 1:2.0 RR).
"""

import time
from zoneinfo import ZoneInfo
import MetaTrader5 as mt5
from datetime import datetime, timezone

import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.connect import connect, get_candles
    from src.strategy import get_latest_signal, STRATEGY_NAME, SUPPORTED_SYMBOLS
    from src.execution import place_order, set_dry_run, DRY_RUN, modify_position_sl, _pip_size
    import src.execution as execution
    import src.db as db
else:
    from .connect import connect, get_candles
    from .strategy import get_latest_signal, STRATEGY_NAME, SUPPORTED_SYMBOLS
    from .execution import place_order, set_dry_run, DRY_RUN, modify_position_sl, _pip_size
    from . import execution
    from . import db


# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD"]
CANDLES_1M      = 300                # 1-minute lookback bars
CANDLES_15M     = 150                # 15-minute lookback bars (>= 96 bars for FVG)
POLL_SECS       = 10                 # Poll every 10 seconds
RISK_PCT        = 1.0                # 1.0% risk per trade
# ─────────────────────────────────────────────────────────────────────────────


# ── Trading Hours — New York time (UTC-4 / UTC-5) ─────────────────────────────
# Rule: Do NOT execute trades between 14:00 - 20:00 New York time.
# Allowed Window: 20:00 (evening) to 14:00 (afternoon next day).
BLOCKED_START_HOUR = 14
BLOCKED_START_MIN  = 0
BLOCKED_END_HOUR   = 20
BLOCKED_END_MIN    = 0

_NY_TZ = ZoneInfo("America/New_York")
# ─────────────────────────────────────────────────────────────────────────────


def _now_ny() -> datetime:
    """Return current localized time in New York (timezone-naive for comparison)."""
    return datetime.now(tz=_NY_TZ).replace(tzinfo=None)


def is_trading_allowed_now() -> bool:
    """
    Return True if the current New York time is outside the blocked 14:00 - 20:00 window.
    """
    now = _now_ny()
    now_mins = now.hour * 60 + now.minute
    block_start_mins = BLOCKED_START_HOUR * 60 + BLOCKED_START_MIN  # 14:00 -> 840 mins
    block_end_mins   = BLOCKED_END_HOUR * 60 + BLOCKED_END_MIN      # 20:00 -> 1200 mins

    # Blocked if between 14:00 and 20:00
    if block_start_mins <= now_mins < block_end_mins:
        return False
    return True


def get_current_session_label() -> str:
    """Return human readable session label."""
    now = _now_ny()
    if not is_trading_allowed_now():
        return "BLOCKED (14:00 - 20:00 NY)"
    if 20 <= now.hour or now.hour < 2:
        return "Asia Session"
    elif 2 <= now.hour < 7:
        return "London Session"
    elif 7 <= now.hour < 14:
        return "New York Session"
    return "Active Trading Window"


def _session_status_line() -> str:
    """Status string."""
    ny = _now_ny()
    tag = "✅ TRADING ACTIVE" if is_trading_allowed_now() else "⏸ BLOCKED (14:00-20:00 NY)"
    return f"NY {ny.strftime('%H:%M:%S')}  {tag}"


def print_signal(symbol: str, sig: dict) -> None:
    """Pretty-print a signal dict to console."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if sig.get("signal", 0) == 0:
        htf = sig.get("htf_trend", "N/A")
        fvg_tag = f" | 15M FVG: [{sig.get('fvg_bottom')}-{sig.get('fvg_top')}]" if sig.get("fvg_active") else " | No 15M FVG"
        ml_tag = f" | ML: {sig['xgb_prob']*100:.1f}%" if sig.get("xgb_prob") is not None else ""
        print(f"[{ts}][{symbol:<6}] No MSS signal | 15M Trend: {htf:<14}{fvg_tag}{ml_tag}")
        return

    direction = "🟢 BUY " if sig["signal"] == 1 else "🔴 SELL"

    print(
        f"\n{'='*70}\n"
        f"  {direction} MARKET STRUCTURE SHIFT DETECTED for {symbol} — {sig.get('label', '')}\n"
        f"  15M Trend     : {sig.get('htf_trend', 'N/A')}\n"
        f"  15M FVG Range : {sig.get('fvg_bottom')} - {sig.get('fvg_top')}\n"
        f"  Time          : {sig.get('datetime', 'N/A')}\n"
        f"  Entry Price   : {sig.get('entry_price')}\n"
        f"  Stop Loss     : {sig.get('sl_price')} ({sig.get('sl_pips', 0):.1f} pips)\n"
        f"  Take Profit   : {sig.get('tp_price')} ({sig.get('tp_pips', 0):.1f} pips, 1:2.0 RR)\n"
        f"  ML Confidence : {sig.get('xgb_prob', 'N/A')}\n"
        f"{'='*70}\n"
    )


def main() -> None:
    if "--dry-run" in sys.argv:
        set_dry_run(True)
    else:
        set_dry_run(False)

    symbols = DEFAULT_SYMBOLS
    for arg in sys.argv:
        if arg.startswith("--symbols="):
            symbols = [s.strip() for s in arg.split("=")[1].split(",") if s.strip()]
        elif arg.startswith("--symbol="):
            symbols = [arg.split("=")[1].strip()]

    print(f"EquantEdge — Multi-Asset Live Trading [{STRATEGY_NAME}]")
    print(f"Mode     : {'[DRY RUN — simulation only]' if execution.DRY_RUN else '⚡ REALTIME LIVE TRADING (Sending Real Orders to MT5)'}")
    print(f"Symbols  : {', '.join(symbols)}  |  Risk: {RISK_PCT}% per trade (1:2.0 RR)")
    ny_now = datetime.now(tz=_NY_TZ)
    utc_offset = int(ny_now.utcoffset().total_seconds() // 3600)
    tz_abbr = ny_now.strftime("%Z")
    print(f"Timezone : America/New_York ({tz_abbr}, UTC{utc_offset:+d})")
    print("Execution: Active 20:00 - 14:00 NY | BLOCKED 14:00 - 20:00 NY")
    print("Press Ctrl+C to stop.\n")

    account = connect()
    open_tickets = {}
    trade_meta = {}
    last_seen_bars = {}
    last_out_log_min = None

    try:
        while True:
            # ── Check open positions (Close detection & Breakeven Trailing) ────
            for sym, ticket in list(open_tickets.items()):
                if ticket and ticket > 0:
                    pos = mt5.positions_get(ticket=ticket)
                    if not pos:
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
                        p = pos[0]
                        meta = trade_meta.get(ticket)
                        if meta and not meta.get("be_moved"):
                            sig_side = meta["sig"]
                            entry = meta["entry"]
                            risk_dist = meta["risk_dist"]
                            pip = _pip_size(sym)

                            # Trailing SL to breakeven after +1.0R
                            if sig_side == 1 and p.price_current >= (entry + risk_dist):
                                new_sl = entry + (1.0 * pip)
                                if modify_position_sl(ticket, sym, new_sl):
                                    meta["be_moved"] = True
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}][{sym}] 🎯 +1.0R Reached! Trailed SL to Breakeven ({new_sl})")
                            elif sig_side == -1 and p.price_current <= (entry - risk_dist):
                                new_sl = entry - (1.0 * pip)
                                if modify_position_sl(ticket, sym, new_sl):
                                    meta["be_moved"] = True
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}][{sym}] 🎯 +1.0R Reached! Trailed SL to Breakeven ({new_sl})")

            # ── Trading Hours Gate ────────────────────────────────────────────
            if not is_trading_allowed_now():
                now_minute = _now_ny().strftime("%H:%M")
                if now_minute != last_out_log_min:
                    last_out_log_min = now_minute
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]  "
                          f"{_session_status_line()} — waiting for 20:00 NY open.")
                time.sleep(POLL_SECS)
                continue

            last_out_log_min = None

            # ── Multi-Symbol Candle Evaluation Loop ───────────────────────────
            for sym in symbols:
                # Fetch 15M HTF candles + 1M LTF candles
                df_15m = get_candles(sym, mt5.TIMEFRAME_M15, CANDLES_15M)
                df_1m  = get_candles(sym, mt5.TIMEFRAME_M1, CANDLES_1M)

                if df_1m.empty or len(df_1m) < 15:
                    continue

                sig = get_latest_signal(df=df_1m, df_15m=df_15m, symbol=sym, use_ml=True)
                bar_time = sig.get("datetime")

                if bar_time != last_seen_bars.get(sym):
                    last_seen_bars[sym] = bar_time
                    print_signal(sym, sig)

                    # ── Place order on confirmed MSS signal ───────────────────
                    if sig.get("signal", 0) != 0 and open_tickets.get(sym) is None:
                        sess_name = get_current_session_label()
                        ticket = place_order(
                            symbol=sym,
                            signal=sig["signal"],
                            entry_price=sig["entry_price"],
                            sl_price=sig.get("sl_price"),
                            tp_price=sig.get("tp_price"),
                            sl_pips=sig.get("sl_pips"),
                            tp_pips=sig.get("tp_pips"),
                            account_balance=account.balance if account else 10000.0,
                            risk_pct=RISK_PCT,
                            session=sess_name,
                            ema_50=sig.get("fvg_top"),
                        )
                        if ticket and ticket > 0:
                            open_tickets[sym] = ticket
                            risk_distance = sig.get("risk_distance") or abs(sig["entry_price"] - sig["sl_price"])
                            trade_meta[ticket] = {
                                "entry": sig["entry_price"],
                                "initial_sl": sig.get("sl_price"),
                                "sig": sig["signal"],
                                "be_moved": False,
                                "risk_dist": risk_distance,
                            }
                            print(f"  ↳ [{sym}] MSS Order placed — ticket #{ticket} (Risk: {risk_distance:.5f})")

            time.sleep(POLL_SECS)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mt5.shutdown()
        print("MT5 connection closed.")


if __name__ == "__main__":
    main()
