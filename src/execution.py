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
    # Forex majors (EURUSD, GBPUSD, etc.): 1 pip = 0.0001
    "default": 4,
    # JPY pairs: 1 pip = 0.01
    "JPY": 2,
    # Gold / Silver: 1 pip = 0.01
    "XAU": 2,
    "XAG": 2,
    # US30 / Dow Jones index: 1 point = 1.0 (or 0 decimal places)
    "US30": 0,
    "DJI": 0,
    "WS30": 0,
}

_log_prefix = lambda: f"[{datetime.now().strftime('%H:%M:%S')}][{'DRY' if DRY_RUN else 'LIVE'}]"


def _pip_size(symbol: str) -> float:
    """Return pip/point size for a symbol.

    * Indices (US30, DJI, WS30) → 1.0  (1 point)
    * Metals (XAU, XAG)         → 0.01 (2 decimal places)
    * JPY pairs                 → 0.01 (2 decimal places)
    * Forex (EURUSD, GBPUSD)    → 0.0001 (4 decimal places)
    """
    sym = symbol.upper()
    if any(k in sym for k in ("US30", "DJI", "WS30")):
        return 1.0
    elif "XAU" in sym or "XAG" in sym:
        digits = PIP_DIGITS["XAU"]
    elif "JPY" in sym:
        digits = PIP_DIGITS["JPY"]
    else:
        digits = PIP_DIGITS["default"]
    return 10 ** -digits


def get_symbol_filling_modes(info) -> list[int]:
    """
    Return an ordered list of MT5 order filling modes supported by the broker for this symbol.
    info.filling_mode bitmask:
      - bit 0 (1): SYMBOL_FILLING_FOK -> mt5.ORDER_FILLING_FOK (0)
      - bit 1 (2): SYMBOL_FILLING_IOC -> mt5.ORDER_FILLING_IOC (1)
    """
    modes = []
    filling_flags = getattr(info, "filling_mode", 0)
    if filling_flags & 1:  # SYMBOL_FILLING_FOK
        modes.append(mt5.ORDER_FILLING_FOK)
    if filling_flags & 2:  # SYMBOL_FILLING_IOC
        modes.append(mt5.ORDER_FILLING_IOC)
    
    # Always append fallbacks
    for alt in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
        if alt not in modes:
            modes.append(alt)
    return modes


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
    entry_price: float,     # entry level
    sl_pips: float | None = None,
    tp_pips: float | None = None,
    account_balance: float = 5000.0,
    risk_pct: float = 1.0,
    comment: str = "EquantEdge LS",
    session: str | None = None,
    ema_50: float | None = None,
    mother_high: float | None = None,
    mother_low: float | None = None,
    sl_price: float | None = None,
    tp_price: float | None = None,
    magic: int = 20260816,
) -> int | None:
    """
    Place a market order in the direction of `signal`.

    Parameters
    ----------
    signal          : 1 (buy) or -1 (sell)
    entry_price     : reference trigger price
    sl_pips         : stop-loss distance in pips (optional if sl_price is given)
    tp_pips         : take-profit distance in pips (optional if tp_price is given)
    account_balance : current account balance for lot sizing
    risk_pct        : fraction of balance to risk (e.g. 1.0 = 1%)
    comment         : MT5 order comment
    session         : active trading session name (e.g. 'New York')
    ema_50          : EMA indicator snapshot at signal
    mother_high     : prior candle high level
    mother_low      : prior candle low level
    sl_price        : exact stop loss price (overrides sl_pips if provided)
    tp_price        : exact take profit price (overrides tp_pips if provided)

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

    # Use current market price
    if signal == 1:          # BUY
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
        if sl_price is not None:
            sl = round(sl_price, info.digits)
        else:
            sl = round(price - (sl_pips or 50.0) * pip, info.digits)
            
        if tp_price is not None:
            tp = round(tp_price, info.digits)
        else:
            tp = round(price + (tp_pips or 100.0) * pip, info.digits)

        # Enforce valid stops for BUY (SL must be < price, TP must be > price)
        if sl >= price:
            print(f"{_log_prefix()} ERROR placing order: Stop Loss ({sl}) >= Ask price ({price}). Order aborted.")
            return None
        if tp <= price:
            print(f"{_log_prefix()} ERROR placing order: Take Profit ({tp}) <= Ask price ({price}). Price moved past TP. Order aborted.")
            return None

    else:                    # SELL
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
        if sl_price is not None:
            sl = round(sl_price, info.digits)
        else:
            sl = round(price + (sl_pips or 50.0) * pip, info.digits)

        if tp_price is not None:
            tp = round(tp_price, info.digits)
        else:
            tp = round(price - (tp_pips or 100.0) * pip, info.digits)

        # Enforce valid stops for SELL (SL must be > price, TP must be < price)
        if sl <= price:
            print(f"{_log_prefix()} ERROR placing order: Stop Loss ({sl}) <= Bid price ({price}). Order aborted.")
            return None
        if tp >= price:
            print(f"{_log_prefix()} ERROR placing order: Take Profit ({tp}) >= Bid price ({price}). Price moved past TP. Order aborted.")
            return None

    effective_sl_pips = abs(price - sl) / pip if pip > 0 else (sl_pips or 50.0)
    if effective_sl_pips < 4.0:
        print(f"{_log_prefix()} ERROR placing order: Measured Stop Loss ({effective_sl_pips:.1f} pips) < 4.0 pips minimum threshold. Order aborted.")
        return None

    # Position sizing
    pip_value = info.trade_tick_value * (pip / info.point)
    lot_size  = calculate_lot_size(
        account_balance = account_balance,
        risk_pct        = risk_pct,
        sl_pips         = effective_sl_pips,
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

    SUCCESS_RETCODES = (
        mt5.TRADE_RETCODE_DONE,           # 10009: Request completed
        mt5.TRADE_RETCODE_PLACED,         # 10008: Order placed
        mt5.TRADE_RETCODE_DONE_PARTIAL,   # 10010: Request completed partially
        0,                                # 0: Done / Success
    )

    # Try supported filling modes with automatic fallback
    candidate_modes = get_symbol_filling_modes(info)
    result = None
    
    for mode in candidate_modes:
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lot_size,
            "type":         order_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    20,           # max price deviation in points
            "magic":        magic,        # unique strategy/EA identifier
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mode,
        }

        result = mt5.order_send(request)
        if result is not None and (result.retcode in SUCCESS_RETCODES or getattr(result, "order", 0) > 0):
            break  # Successfully filled
        elif result is not None and result.retcode == 10030:
            # Unsupported filling mode, try next candidate mode
            continue
        else:
            break

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
    candidate_modes = get_symbol_filling_modes(info)
    result = None

    SUCCESS_RETCODES = (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_DONE_PARTIAL, 0)

    for mode in candidate_modes:
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     ticket,
            "price":        close_price,
            "deviation":    20,
            "magic":        20260816,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mode,
        }

        result = mt5.order_send(request)
        if result is not None and (result.retcode in SUCCESS_RETCODES or getattr(result, "order", 0) > 0 or getattr(result, "deal", 0) > 0):
            break
        elif result is not None and result.retcode == 10030:
            continue
        else:
            break

    is_success = result is not None and (
        result.retcode in SUCCESS_RETCODES
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


def modify_position_sl(ticket: int, symbol: str, new_sl: float) -> bool:
    """
    Modify the stop loss of an existing open position in MT5 (e.g. moving to Breakeven).
    """
    if DRY_RUN:
        print(f"{_log_prefix()} [DRY RUN] Would modify SL for position #{ticket} to {new_sl}")
        return True

    position = mt5.positions_get(ticket=ticket)
    if not position:
        return False

    pos = position[0]
    info = _symbol_info(symbol)
    sl_rounded = round(new_sl, info.digits)

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol":   symbol,
        "sl":       sl_rounded,
        "tp":       pos.tp,
    }

    result = mt5.order_send(request)
    is_success = result is not None and result.retcode in (
        mt5.TRADE_RETCODE_DONE,
        mt5.TRADE_RETCODE_PLACED,
        0,
    )
    if is_success:
        print(f"{_log_prefix()} Position #{ticket} SL updated to {sl_rounded} (Breakeven Trailed)")
        return True
    else:
        code = result.retcode if result else "None"
        print(f"{_log_prefix()} Failed to modify SL for #{ticket}: retcode={code} comment={getattr(result, 'comment', 'N/A')}")
        return False


def get_open_positions(symbol: str | None = None) -> list:
    """Return a list of open positions, optionally filtered by symbol."""
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    return list(positions) if positions else []


def partial_close_position(ticket: int, symbol: str, close_ratio: float = 0.50) -> float | None:
    """
    Partially close an open MT5 position (e.g. 50% scale-out at +1.0R target).
    Returns the closed volume on success, or None on failure.
    """
    if DRY_RUN:
        print(f"{_log_prefix()} [DRY RUN] Would partially close {close_ratio*100:.0f}% of position #{ticket}")
        return 0.5

    position = mt5.positions_get(ticket=ticket)
    if not position:
        return None

    pos = position[0]
    info = _symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

    # Calculate partial volume (rounded to step)
    step = info.volume_step or 0.01
    raw_vol = pos.volume * close_ratio
    close_vol = max(info.volume_min, round(raw_vol / step) * step)
    close_vol = min(close_vol, pos.volume)

    if (pos.volume - close_vol) < info.volume_min:
        # If remaining is less than min lot, close full position
        close_vol = pos.volume

    candidate_modes = get_symbol_filling_modes(info)
    for mode in candidate_modes:
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "position":     ticket,
            "symbol":       symbol,
            "volume":       close_vol,
            "type":         order_type,
            "price":        price,
            "deviation":    20,
            "magic":        pos.magic,
            "comment":      "Partial Close +1.0R",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mode,
        }
        result = mt5.order_send(request)
        if result is not None and result.retcode in (
            mt5.TRADE_RETCODE_DONE,
            mt5.TRADE_RETCODE_DONE_PARTIAL,
            mt5.TRADE_RETCODE_PLACED,
            0,
        ):
            print(f"{_log_prefix()} Partial close of {close_vol} lots on #{ticket} successful (+1.0R Secured).")
            return close_vol
        elif result is not None and result.retcode == 10030:
            continue

    print(f"{_log_prefix()} Failed partial close on #{ticket}.")
    return None

