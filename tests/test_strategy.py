"""
test_strategy.py

Unit tests for the Liquidity Sweep Reversal strategy.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy import generate_signals, get_latest_signal
from src.patterns import is_liquidity_sweep_sell, is_liquidity_sweep_buy


def test_sell_liquidity_sweep():
    """
    Test Sell Signal:
      Candle 1: Bullish (Open=4340, Close=4345, High=4346, Low=4339)
      Candle 2: Bearish (Open=4346, High=4348 [swept], Low=4338, Close=4338 [closed below 4340])
      MA is below entry price (e.g. ~4300)
    """
    # Create baseline series to populate MA (55 bars)
    dates = pd.date_range("2026-08-19 10:00", periods=60, freq="1min")
    data = {
        "datetime": dates,
        "open":  [4300.0] * 58 + [4340.0, 4346.0],
        "high":  [4305.0] * 58 + [4346.0, 4348.0],
        "low":   [4295.0] * 58 + [4339.0, 4338.0],
        "close": [4300.0] * 58 + [4345.0, 4338.0],
        "volume": [100] * 60,
    }
    df = pd.DataFrame(data)

    df_signals = generate_signals(df, ma_period=10, pip_size=0.01, pip_buffer=1.0, atr_buffer_mult=0.0, rr_ratio=2.0)
    
    # Check last row (Candle 2)
    last_row = df_signals.iloc[-1]
    assert last_row["sweep_sell"] == True, "Bearish sweep should be detected"
    assert last_row["signal"] == -1, "Sell signal should be triggered when close > MA"
    
    expected_sl = 4348.0 + 0.01  # 4348.01
    expected_risk = expected_sl - 4338.0  # 10.01
    expected_tp = 4338.0 - (2.0 * expected_risk)  # 4317.98

    assert round(last_row["sl_price"], 2) == round(expected_sl, 2), f"Expected SL {expected_sl}, got {last_row['sl_price']}"
    assert round(last_row["tp_price"], 2) == round(expected_tp, 2), f"Expected TP {expected_tp}, got {last_row['tp_price']}"
    print("[PASS] Sell Liquidity Sweep test passed!")



def test_buy_liquidity_sweep():
    """
    Test Buy Signal:
      Candle 1: Bearish (Open=4350, Close=4340, High=4352, Low=4338)
      Candle 2: Bullish (Open=4338, High=4355, Low=4335 [swept], Close=4352 [closed above 4350])
      MA is above entry price (e.g. ~4400)
    """
    dates = pd.date_range("2026-08-19 10:00", periods=60, freq="1min")
    data = {
        "datetime": dates,
        "open":  [4400.0] * 58 + [4350.0, 4338.0],
        "high":  [4405.0] * 58 + [4352.0, 4355.0],
        "low":   [4395.0] * 58 + [4338.0, 4335.0],
        "close": [4400.0] * 58 + [4340.0, 4352.0],
        "volume": [100] * 60,
    }
    df = pd.DataFrame(data)

    df_signals = generate_signals(df, ma_period=10, pip_size=0.01, pip_buffer=1.0, atr_buffer_mult=0.0, rr_ratio=2.0)
    
    last_row = df_signals.iloc[-1]
    assert last_row["sweep_buy"] == True, "Bullish sweep should be detected"
    assert last_row["signal"] == 1, "Buy signal should be triggered when close < MA"
    
    expected_sl = 4335.0 - 0.01  # 4334.99
    expected_risk = 4352.0 - expected_sl  # 17.01
    expected_tp = 4352.0 + (2.0 * expected_risk)  # 4386.02

    assert round(last_row["sl_price"], 2) == round(expected_sl, 2), f"Expected SL {expected_sl}, got {last_row['sl_price']}"
    assert round(last_row["tp_price"], 2) == round(expected_tp, 2), f"Expected TP {expected_tp}, got {last_row['tp_price']}"
    print("[PASS] Buy Liquidity Sweep test passed!")



def test_filters_and_invalids():
    """
    Test when conditions are not satisfied (e.g. liquidity not swept or close not breaking prior open).
    """
    dates = pd.date_range("2026-08-19 10:00", periods=5, freq="1min")
    
    # Case 1: High NOT swept (High2 < High1)
    df_no_sweep = pd.DataFrame({
        "datetime": dates,
        "open":  [4300, 4300, 4300, 4340.0, 4344.0],
        "high":  [4305, 4305, 4305, 4346.0, 4345.0],  # 4345 < 4346 (No sweep)
        "low":   [4295, 4295, 4295, 4339.0, 4338.0],
        "close": [4300, 4300, 4300, 4345.0, 4338.0],
        "volume": [100] * 5,
    })
    sig_no_sweep = is_liquidity_sweep_sell(df_no_sweep)
    assert sig_no_sweep.iloc[-1] == False, "Should NOT trigger without liquidity sweep"

    # Case 2: Close did NOT break prior Open (Close2 > Open1)
    df_no_close_break = pd.DataFrame({
        "datetime": dates,
        "open":  [4300, 4300, 4300, 4340.0, 4346.0],
        "high":  [4305, 4305, 4305, 4346.0, 4348.0],
        "low":   [4295, 4295, 4295, 4339.0, 4338.0],
        "close": [4300, 4300, 4300, 4345.0, 4342.0],  # 4342 > 4340 (Did not close below open)
        "volume": [100] * 5,
    })
    sig_no_close = is_liquidity_sweep_sell(df_no_close_break)
    assert sig_no_close.iloc[-1] == False, "Should NOT trigger without close below prior open"
    
    print("[PASS] Negative test filters passed!")


if __name__ == "__main__":
    test_sell_liquidity_sweep()
    test_buy_liquidity_sweep()
    test_filters_and_invalids()
    print("\nALL STRATEGY TESTS PASSED!")

