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

import MetaTrader5 as mt5
from datetime import datetime
from .risk import calculate_lot_size

# ── Set to False when you are ready to trade real money on a demo account ──
DRY_RUN: bool = True
# ───────────────────────────────────────────────────────────────────────────

PIP_DIGITS = {
    # pairs where 1 pip = 0.0001
    "default": 4,
    # pairs where 1 pip = 0.01 (JPY pairs)
    "JPY": 2,
}

_log_prefix = lambda: f"[{datetime.now().strftime('%H:%M:%S')}][{'DRY' if DRY_RUN else 'LIVE'}]"


def _pip_size(symbol: str) -> float:
    """Return pip size for a symbol (0.0001 for most pairs, 0.01 for JPY)."""
    digits = PIP_DIGITS["JPY"] if "JPY" in symbol.upper() else PIP_DIGITS["default"]
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

    direction = "BUY " if signal == 1 else "SELL"
    print(
        f"{_log_prefix()} {direction} {symbol}  "
        f"price={price}  sl={sl}  tp={tp}  lots={lot_size}"
    )

    if DRY_RUN:
        print(f"{_log_prefix()} DRY RUN — order NOT sent.")
        return -1   # sentinel for dry-run "success"

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
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        print(f"{_log_prefix()} ERROR placing order: retcode={code}  "
              f"comment={getattr(result, 'comment', 'N/A')}")
        return None

    print(f"{_log_prefix()} Order filled — ticket #{result.order}")
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
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        print(f"{_log_prefix()} ERROR closing position: retcode={code}")
        return False

    print(f"{_log_prefix()} Position #{ticket} closed successfully.")
    return True


def get_open_positions(symbol: str | None = None) -> list:
    """Return a list of open positions, optionally filtered by symbol."""
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    return list(positions) if positions else []
