"""
test_strategy.py

Comprehensive unit tests for the 15M FVG + 1M Market Structure Shift (MSS) Strategy.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.patterns import find_15m_fvgs, get_nearest_active_fvg, detect_market_structure_shift
from src.strategy import generate_signals, get_latest_signal, _detect_pip_size


def test_15m_fvg_detection():
    """
    Test 15-Minute Bearish & Bullish Fair Value Gap detection.
    
    Bearish FVG:
      Candle 1: Low = 1.3550
      Candle 2: Impulse down (Open=1.3548, Close=1.3510, High=1.3550, Low=1.3508)
      Candle 3: High = 1.3520
      Imbalance Gap: [1.3520, 1.3550] (Candle 1 Low > Candle 3 High)
    """
    dates = pd.date_range("2026-08-25 08:00", periods=5, freq="15min")
    df_bearish_fvg = pd.DataFrame({
        "datetime": dates,
        "open":  [1.3560, 1.3555, 1.3548, 1.3515, 1.3505],
        "high":  [1.3570, 1.3565, 1.3550, 1.3520, 1.3510],
        "low":   [1.3550, 1.3550, 1.3508, 1.3500, 1.3490],  # Candle 1 (idx 1) Low = 1.3550
        "close": [1.3555, 1.3550, 1.3510, 1.3505, 1.3495],
        "volume": [500] * 5,
    })

    fvgs = find_15m_fvgs(df_bearish_fvg, max_lookback_bars=96)
    assert len(fvgs) >= 1, "Should detect at least 1 Bearish FVG"
    bear_fvg = fvgs[0]
    assert bear_fvg["type"] == "BEARISH"
    assert bear_fvg["top"] == 1.3550  # Candle 1 Low
    assert bear_fvg["bottom"] == 1.3520  # Candle 3 High
    assert bear_fvg["size"] == 0.0030
    print("[PASS] 15M Bearish FVG detection test passed!")


def test_1m_market_structure_shift_sell():
    """
    Test 1-Minute Sell Scenario (as shown in User Image 02):
      1. 15M Bearish FVG active in range [1.3520, 1.3550].
      2. Price rallies into the FVG, creates a swing peak at 1.3545 (high of last bullish leg).
      3. The swing low of the bullish leg that led to 1.3545 is at 1.3520.
      4. Prior 1M candle was at 1.3525 (above swing low).
      5. Current closed 1M candle closes at 1.3515 (below 1.3520) -> Market Structure Shift!
      6. Expected Entry = 1.3520, SL = 1.3545, TP = 1.3520 - 2*(1.3545 - 1.3520) = 1.3470 (1:2.0 RR).
    """
    fvg = {"type": "BEARISH", "direction": -1, "top": 1.3550, "bottom": 1.3520}

    dates = pd.date_range("2026-08-25 10:00", periods=15, freq="1min")
    opens =  [1.3510]*5 + [1.3522, 1.3528, 1.3535, 1.3542, 1.3538, 1.3532, 1.3526, 1.3518, 1.3515, 1.3515]
    highs =  [1.3515]*5 + [1.3528, 1.3538, 1.3545, 1.3544, 1.3540, 1.3535, 1.3528, 1.3520, 1.3518, 1.3518]
    lows =   [1.3505]*5 + [1.3520, 1.3525, 1.3532, 1.3538, 1.3530, 1.3524, 1.3521, 1.3512, 1.3510, 1.3510]
    closes = [1.3510]*5 + [1.3525, 1.3535, 1.3542, 1.3538, 1.3532, 1.3525, 1.3522, 1.3515, 1.3514, 1.3515]

    df_1m = pd.DataFrame({
        "datetime": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [100] * 15,
    })

    mss = detect_market_structure_shift(df_1m, fvg=fvg, trend_side=-1, pip_size=0.0001)
    assert mss is not None, "Should detect Market Structure Shift to sell-side"
    assert mss["signal"] == -1
    assert round(mss["sl_price"], 4) == 1.3545, f"Expected SL 1.3545, got {mss['sl_price']}"
    assert mss["entry_price"] > mss["tp_price"], "TP should be below entry for sell"
    risk = mss["sl_price"] - mss["entry_price"]
    expected_tp = mss["entry_price"] - (2.0 * risk)
    assert round(mss["tp_price"], 5) == round(expected_tp, 5), f"Expected 1:2 TP {expected_tp}, got {mss['tp_price']}"
    print(f"[PASS] 1M Sell Market Structure Shift test passed! (Entry: {mss['entry_price']}, SL: {mss['sl_price']}, TP: {mss['tp_price']})")


def test_1m_market_structure_shift_buy():
    """
    Test 1-Minute Buy Scenario (Mirror):
      1. 15M Bullish FVG active in range [1.1000, 1.1030].
      2. Price dips into FVG, creates a swing valley at 1.1005 (low of last bearish leg).
      3. The swing high of the bearish leg that led to 1.1005 is at 1.1025.
      4. Prior candle was at 1.1022 (below swing high).
      5. Current closed candle closes at 1.1028 (above 1.1025) -> Market Structure Shift!
      6. Expected Entry = 1.1025, SL = 1.1005, TP = 1.1025 + 2*(1.1025 - 1.1005) = 1.1065 (1:2.0 RR).
    """
    fvg = {"type": "BULLISH", "direction": 1, "top": 1.1030, "bottom": 1.1000}

    dates = pd.date_range("2026-08-25 10:00", periods=15, freq="1min")
    opens =  [1.1040]*5 + [1.1028, 1.1022, 1.1012, 1.1006, 1.1012, 1.1018, 1.1022, 1.1026, 1.1030, 1.1035]
    highs =  [1.1045]*5 + [1.1030, 1.1025, 1.1015, 1.1010, 1.1016, 1.1022, 1.1024, 1.1030, 1.1038, 1.1038]
    lows =   [1.1035]*5 + [1.1022, 1.1012, 1.1005, 1.1005, 1.1008, 1.1015, 1.1019, 1.1024, 1.1028, 1.1028]
    closes = [1.1040]*5 + [1.1022, 1.1015, 1.1008, 1.1012, 1.1016, 1.1020, 1.1022, 1.1028, 1.1035, 1.1035]

    df_1m = pd.DataFrame({
        "datetime": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [100] * 15,
    })

    mss = detect_market_structure_shift(df_1m, fvg=fvg, trend_side=1, pip_size=0.0001)
    assert mss is not None, "Should detect Market Structure Shift to buy-side"
    assert mss["signal"] == 1
    assert round(mss["sl_price"], 4) == 1.1005, f"Expected SL 1.1005, got {mss['sl_price']}"
    risk = mss["entry_price"] - mss["sl_price"]
    expected_tp = mss["entry_price"] + (2.0 * risk)
    assert round(mss["tp_price"], 5) == round(expected_tp, 5), f"Expected 1:2 TP {expected_tp}, got {mss['tp_price']}"
    print(f"[PASS] 1M Buy Market Structure Shift test passed! (Entry: {mss['entry_price']}, SL: {mss['sl_price']}, TP: {mss['tp_price']})")


def test_pip_size_detection():
    """Verify 0.0001 pip detection for EURUSD, GBPUSD, AUDUSD."""
    assert _detect_pip_size(symbol="EURUSD") == 0.0001
    assert _detect_pip_size(symbol="GBPUSD") == 0.0001
    assert _detect_pip_size(symbol="AUDUSD") == 0.0001
    print("[PASS] Pip size detection test passed!")


if __name__ == "__main__":
    test_15m_fvg_detection()
    test_1m_market_structure_shift_sell()
    test_1m_market_structure_shift_buy()
    test_pip_size_detection()
    print("\nALL 15M FVG + 1M MSS STRATEGY TESTS PASSED!")
