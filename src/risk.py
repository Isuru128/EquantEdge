"""
risk.py

Position sizing and risk controls. Keep this separate from strategy logic
so risk rules are consistent and easy to audit regardless of what
generates the signal.
"""


def calculate_position_size(
    account_balance: float,
    risk_per_trade_pct: float,
    stop_loss_pips: float,
    pip_value: float,
) -> float:
    """
    Standard fixed-fractional position sizing:
    lot_size = (balance * risk_pct) / (stop_loss_pips * pip_value)

    TODO: adapt to your broker's lot/contract size conventions.
    """
    risk_amount = account_balance * (risk_per_trade_pct / 100)
    if stop_loss_pips <= 0 or pip_value <= 0:
        raise ValueError("stop_loss_pips and pip_value must be positive.")
    return risk_amount / (stop_loss_pips * pip_value)


def check_daily_loss_limit(
    starting_balance: float,
    current_balance: float,
    max_daily_loss_pct: float,
) -> bool:
    """Returns True if trading should STOP for the day."""
    loss_pct = (starting_balance - current_balance) / starting_balance * 100
    return loss_pct >= max_daily_loss_pct
