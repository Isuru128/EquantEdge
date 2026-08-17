"""
execution.py

Order placement and management via the MetaTrader 5 API.

Design decisions:
  - DRY_RUN = True  → all functions log what they WOULD do, no real orders.
  - DRY_RUN = False → real market/pending orders are sent to MT5.
  - Only MARKET orders are used for simplicity. Pending stop-orders can
    be added later by changing order_type to mt5.ORDER_TYPE_BUY_STOP /
    ORDER_TYPE_SELL_STOP.

Usage example:
    from src.execution import place_order, close_position, DRY_RUN
"""

import math
import os
import sys
import MetaTrader5 as mt5
from datetime import datetime

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.risk import calculate_lot_size
    import src.db as db
else:
    from .risk import calculate_lot_size
    from . import db

# ── Live execution mode (False = Send real orders to MT5 demo) ───────────────
DRY_RUN: bool = False


def set_dry_run(enable: bool) -> None:
    """Dynamically enable or disable dry-run mode."""
    global DRY_RUN
    DRY_RUN = enable
# ───────────────────────────────────────────────────────────────────────────

PIP_DIGITS = {
    # pairs where 1 pip = 0.0001
    "default": 4,
    # pairs where 1 pip = 0.01 (JPY pairs)
    "JPY": 2,
    # Gold / Silver spot: quoted to 2 decimal places (1 pip = 0.01)
    "XAU": 2,
    "XAG": 2,
}

_log_prefix = lambda: f"[{datetime.now().strftime('%H:%M:%S')}][{'DRY' if DRY_RUN else 'LIVE'}]"


def _pip_size(symbol: str) -> float:
    """Return pip size for a symbol.

    * Metals (XAU, XAG)  → 0.01  (2 decimal places)
    * JPY pairs          → 0.01  (2 decimal places)
    * All other pairs    → 0.0001 (4 decimal places)
    """
    sym = symbol.upper()
    if "XAU" in sym or "XAG" in sym:
        digits = PIP_DIGITS["XAU"]
    elif "JPY" in sym:
        digits = PIP_DIGITS["JPY"]
    else:
        digits = PIP_DIGITS["default"]
    return 10 ** -digits


def _symbol_info(symbol: str):
    """Fetch and enable symbol if needed, then return SymbolInfo."""
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol '{symbol}' not found in MT5.")
    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
    return info


def place_order(
    symbol: str,
    signal: int,            # 1 = buy, -1 = sell
    entry_price: float,     # mother bar high (buy) or low (sell)
    sl_pips: float,
    tp_pips: float,
    account_balance: float,
    risk_pct: float,
    comment: str = "EquantEdge IB",
    session: str | None = None,
    ema_50: float | None = None,
    mother_high: float | None = None,
    mother_low: float | None = None,
) -> int | None:
    """
    Place a market order in the direction of `signal`.

    Parameters
    ----------
    signal          : 1 (buy) or -1 (sell)
    entry_price     : used only for SL/TP calculation; actual fill is at market
    sl_pips         : stop-loss distance in pips
    tp_pips         : take-profit distance in pips
    account_balance : current account balance for lot sizing
    risk_pct        : fraction of balance to risk (e.g. 1.0 = 1%)
    comment         : MT5 order comment
    session         : active trading session name (e.g. 'New York')
    ema_50          : EMA indicator snapshot at signal
    mother_high     : mother bar high level
    mother_low      : mother bar low level

    Returns
    -------
    ticket number on success, None on failure.
    """
    pip = _pip_size(symbol)
    info = _symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"{_log_prefix()} ERROR: could not get tick for {symbol}")
        return None

    # Use current market price, not the pending breakout level
    if signal == 1:          # BUY
        price    = tick.ask
        sl       = round(price - sl_pips * pip, info.digits)
        tp       = round(price + tp_pips * pip, info.digits)
        order_type = mt5.ORDER_TYPE_BUY
    else:                    # SELL
        price    = tick.bid
        sl       = round(price + sl_pips * pip, info.digits)
        tp       = round(price - tp_pips * pip, info.digits)
        order_type = mt5.ORDER_TYPE_SELL

    # Position sizing
    pip_value = info.trade_tick_value * (pip / info.point)
    lot_size  = calculate_lot_size(
        account_balance = account_balance,
        risk_pct        = risk_pct,
        sl_pips         = sl_pips,
        pip_value       = pip_value,
        min_lot         = info.volume_min,
        max_lot         = info.volume_max,
        lot_step        = info.volume_step,
    )

    # ── Margin Check & Auto-Clamping ──────────────────────────────────────────
    acc_info = mt5.account_info()
    free_margin = acc_info.margin_free if acc_info else account_balance
    margin_req = mt5.order_calc_margin(order_type, symbol, lot_size, price)

    if margin_req is not None and margin_req > (free_margin * 0.95):
        margin_min = mt5.order_calc_margin(order_type, symbol, info.volume_min, price)
        if margin_min and free_margin < margin_min:
            print(
                f"{_log_prefix()} ERROR: Insufficient free margin (${free_margin:.2f}) "
                f"to open minimum lot {info.volume_min} (${margin_min:.2f} required)."
            )
            return None
        margin_per_step = mt5.order_calc_margin(order_type, symbol, info.volume_step, price) or 1.0
        max_possible_lots = math.floor((free_margin * 0.90) / margin_per_step) * info.volume_step
        clamped_lots = round(max(info.volume_min, min(max_possible_lots, info.volume_max)), 2)
        print(
            f"{_log_prefix()} Margin safety: Requested {lot_size} lots (${margin_req:.2f} margin) "
            f"exceeds free margin (${free_margin:.2f}). Adjusted to {clamped_lots} lots."
        )
        lot_size = clamped_lots

    direction = "BUY " if signal == 1 else "SELL"
    print(
        f"{_log_prefix()} {direction} {symbol}  "
        f"price={price}  sl={sl}  tp={tp}  lots={lot_size}"
    )

    if DRY_RUN:
        print(f"{_log_prefix()} DRY RUN — order NOT sent.")
        return -1   # sentinel for dry-run "success"

    # Determine broker filling mode
    filling_mode = mt5.ORDER_FILLING_IOC
    if info.filling_mode & mt5.ORDER_FILLING_IOC:
        filling_mode = mt5.ORDER_FILLING_IOC
    elif info.filling_mode & mt5.ORDER_FILLING_RETURN:
        filling_mode = mt5.ORDER_FILLING_RETURN
    elif info.filling_mode & mt5.ORDER_FILLING_FOK:
        filling_mode = mt5.ORDER_FILLING_FOK

    request = {
        "action":     mt5.TRADE_ACTION_DEAL,
        "symbol":     symbol,
        "volume":     lot_size,
        "type":       order_type,
        "price":      price,
        "sl":         sl,
        "tp":         tp,
        "deviation":  20,           # max price deviation in points
        "magic":      20260816,     # unique EA identifier
        "comment":    comment,
        "type_time":  mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    SUCCESS_RETCODES = (
        mt5.TRADE_RETCODE_DONE,           # 10009: Request completed
        mt5.TRADE_RETCODE_PLACED,         # 10008: Order placed
        mt5.TRADE_RETCODE_DONE_PARTIAL,   # 10010: Request completed partially
        0,                                # 0: Done / Success
    )

    result = mt5.order_send(request)
    is_success = result is not None and (
        result.retcode in SUCCESS_RETCODES or (getattr(result, "order", 0) > 0)
    )

    if not is_success:
        code = result.retcode if result else "None"
        print(f"{_log_prefix()} ERROR placing order: retcode={code}  "
              f"comment={getattr(result, 'comment', 'N/A')}")
        return None

    print(f"{_log_prefix()} Order filled — ticket #{result.order}")

    # Log to Supabase Database
    try:
        db.log_trade_open(
            ticket=result.order,
            symbol=symbol,
            side=direction.strip(),
            lot_size=lot_size,
            open_price=price,
            sl=sl,
            tp=tp,
            session=session,
            ema_50=ema_50,
            mother_high=mother_high,
            mother_low=mother_low,
        )
    except Exception as exc:
        print(f"{_log_prefix()} Warning: DB log trade open failed: {exc}")

    return result.order


def close_position(ticket: int, symbol: str, comment: str = "EquantEdge close") -> bool:
    """
    Close an open position by ticket number.

    Returns True on success (or DRY_RUN), False on failure.
    """
    print(f"{_log_prefix()} Closing position ticket #{ticket}")

    if DRY_RUN:
        print(f"{_log_prefix()} DRY RUN — close NOT sent.")
        return True

    position = mt5.positions_get(ticket=ticket)
    if not position:
        print(f"{_log_prefix()} No open position found for ticket #{ticket}")
        return False

    pos = position[0]
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"{_log_prefix()} ERROR: could not get tick for {symbol}")
        return False

    # Close by placing an opposite market order
    close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

    # Determine broker filling mode
    info = _symbol_info(symbol)
    filling_mode = mt5.ORDER_FILLING_IOC
    if info.filling_mode & mt5.ORDER_FILLING_IOC:
        filling_mode = mt5.ORDER_FILLING_IOC
    elif info.filling_mode & mt5.ORDER_FILLING_RETURN:
        filling_mode = mt5.ORDER_FILLING_RETURN
    elif info.filling_mode & mt5.ORDER_FILLING_FOK:
        filling_mode = mt5.ORDER_FILLING_FOK

    request = {
        "action":     mt5.TRADE_ACTION_DEAL,
        "symbol":     symbol,
        "volume":     pos.volume,
        "type":       close_type,
        "position":   ticket,
        "price":      close_price,
        "deviation":  20,
        "magic":      20260816,
        "comment":    comment,
        "type_time":  mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    result = mt5.order_send(request)
    is_success = result is not None and (
        result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_DONE_PARTIAL, 0)
        or (getattr(result, "order", 0) > 0) or (getattr(result, "deal", 0) > 0)
    )
    if not is_success:
        code = result.retcode if result else "None"
        print(f"{_log_prefix()} ERROR closing position: retcode={code} comment={getattr(result, 'comment', 'N/A')}")
        return False

    print(f"{_log_prefix()} Position #{ticket} closed successfully.")

    # Log close to Supabase Database
    try:
        db.log_trade_close(
            ticket=ticket,
            close_price=close_price,
            profit_usd=getattr(pos, 'profit', None),
            close_reason=comment,
        )
    except Exception as exc:
        print(f"{_log_prefix()} Warning: DB log trade close failed: {exc}")

    return True


def get_open_positions(symbol: str | None = None) -> list:
    """Return a list of open positions, optionally filtered by symbol."""
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    return list(positions) if positions else []
