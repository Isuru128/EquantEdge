"""
risk.py

Position sizing and risk controls. Keep this separate from strategy logic
so risk rules are consistent and easy to audit regardless of what
generates the signal.
"""

import math


def calculate_lot_size(
    account_balance: float,
    risk_pct: float,
    sl_pips: float,
    pip_value: float,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
    lot_step: float = 0.01,
) -> float:
    """
    Fixed-fractional position sizing.

    Formula:
        risk_amount = balance * (risk_pct / 100)
        lots        = risk_amount / (sl_pips * pip_value)

    The result is clamped to [min_lot, max_lot] and rounded down to the
    nearest lot_step (as required by most brokers).

    Parameters
    ----------
    account_balance : current account balance in account currency
    risk_pct        : percent of balance to risk, e.g. 1.0 = 1%
    sl_pips         : stop-loss size in pips
    pip_value       : monetary value of 1 pip per 1 standard lot
                      (from MT5: symbol_info.trade_tick_value)
    min_lot         : broker minimum lot size (default 0.01)
    max_lot         : broker maximum lot size (default 100.0)
    lot_step        : lot size increment (default 0.01)

    Returns
    -------
    Lot size as a float, rounded to lot_step.
    """
    if sl_pips <= 0:
        raise ValueError(f"sl_pips must be > 0, got {sl_pips}")
    if pip_value <= 0:
        raise ValueError(f"pip_value must be > 0, got {pip_value}")
    if not (0 < risk_pct <= 100):
        raise ValueError(f"risk_pct must be between 0 and 100, got {risk_pct}")

    risk_amount = account_balance * (risk_pct / 100.0)
    raw_lots    = risk_amount / (sl_pips * pip_value)

    # Round DOWN to the nearest lot_step (never round up — avoids over-risking)
    lots = math.floor(raw_lots / lot_step) * lot_step

    # Clamp to broker limits
    lots = max(min_lot, min(lots, max_lot))

    return round(lots, 2)


def check_daily_loss_limit(
    starting_balance: float,
    current_balance: float,
    max_daily_loss_pct: float,
) -> bool:
    """
    Returns True if trading should STOP for the day.

    Compares today's starting balance to the current balance.
    If the drawdown exceeds max_daily_loss_pct, returns True.
    """
    if starting_balance <= 0:
        raise ValueError("starting_balance must be positive")
    loss_pct = (starting_balance - current_balance) / starting_balance * 100
    return loss_pct >= max_daily_loss_pct


def max_position_risk(
    account_balance: float,
    open_positions_count: int,
    max_open_positions: int,
    risk_pct: float,
) -> float:
    """
    Guard against over-leveraging when multiple positions are open.

    Returns the effective risk % to use for the next trade, or 0.0 if
    the maximum number of open positions has been reached.

    Example: with max_open_positions=2 and risk_pct=1.0, the second
    trade still uses 1.0% — but a third trade is blocked (returns 0.0).
    """
    if open_positions_count >= max_open_positions:
        return 0.0
    return risk_pct


def reward_to_risk_tp(
    entry: float,
    sl: float,
    rr_ratio: float = 2.0,
) -> float:
    """
    Calculate TP price given an entry, SL, and desired R:R ratio.

    Parameters
    ----------
    entry    : entry price
    sl       : stop-loss price
    rr_ratio : desired reward-to-risk ratio (default 2.0 = 2:1)

    Returns
    -------
    take-profit price
    """
    risk_distance = abs(entry - sl)
    if entry > sl:          # long trade
        return round(entry + risk_distance * rr_ratio, 5)
    else:                   # short trade
        return round(entry - risk_distance * rr_ratio, 5)
