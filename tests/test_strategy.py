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
    Test 1-Minute Sell Scenario:
      1. 15M Bearish FVG active in range [1.3520, 1.3550].
      2. Price rallies into FVG to 1.3535, pulls back to swing low at 1.3520.
      3. Price rallies to swing peak at 1.3545.
      4. A 1M candle closes below 1.3520 (MSS Breakout).
      5. Subsequent 1M candle pulls back and tests 1.3520 (Retest Entry).
      6. Expected Entry = 1.3520, SL = 1.3545, TP = 1.3470 (1:2.0 RR).
    """
    fvg = {"type": "BEARISH", "direction": -1, "top": 1.3550, "bottom": 1.3520}

    dates = pd.date_range("2026-08-25 10:00", periods=16, freq="1min")
    # 0..4: baseline
    # 5: push to 1.3535 (high=1.3535, low=1.3525)
    # 6: pullback to 1.3520 (high=1.3528, low=1.3520) -> Pivot Swing Low!
    # 7: push to 1.3538 (high=1.3538, low=1.3525)
    # 8: push to peak 1.3545 (high=1.3545, low=1.3535) -> Swing Peak!
    # 9: rejection (high=1.3540, low=1.3530)
    # 10: drop (high=1.3532, low=1.3522)
    # 11: breakdown close below 1.3520 (high=1.3522, low=1.3512, close=1.3514) -> MSS Breakout!
    # 12: continuation low (high=1.3516, low=1.3508, close=1.3512)
    # 13: pullback test (high=1.3522, low=1.3514, close=1.3518) -> Retest Touch!
    # 14: closed bar confirming retest rejection (high=1.3520, low=1.3515, close=1.3517)
    # 15: active forming bar

    opens  = [1.3510]*5 + [1.3525, 1.3530, 1.3526, 1.3538, 1.3542, 1.3535, 1.3520, 1.3514, 1.3512, 1.3518, 1.3517]
    highs  = [1.3515]*5 + [1.3535, 1.3530, 1.3538, 1.3545, 1.3540, 1.3535, 1.3522, 1.3516, 1.3522, 1.3520, 1.3518]
    lows   = [1.3505]*5 + [1.3522, 1.3520, 1.3525, 1.3535, 1.3530, 1.3522, 1.3512, 1.3508, 1.3512, 1.3515, 1.3515]
    closes = [1.3510]*5 + [1.3530, 1.3525, 1.3535, 1.3542, 1.3538, 1.3528, 1.3514, 1.3512, 1.3518, 1.3517, 1.3517]

    df_1m = pd.DataFrame({
        "datetime": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [100] * 16,
    })

    mss = detect_market_structure_shift(df_1m, fvg=fvg, trend_side=-1, pip_size=0.0001, require_retest=True)
    assert mss is not None, "Should detect Market Structure Shift Retest to sell-side"
    assert mss["signal"] == -1
    assert round(mss["sl_price"], 4) == 1.3545, f"Expected SL 1.3545, got {mss['sl_price']}"
    assert round(mss["entry_price"], 4) == 1.3520, f"Expected Entry 1.3520, got {mss['entry_price']}"
    assert mss["entry_price"] > mss["tp_price"], "TP should be below entry for sell"
    risk = mss["sl_price"] - mss["entry_price"]
    expected_tp = mss["entry_price"] - (2.0 * risk)
    assert round(mss["tp_price"], 5) == round(expected_tp, 5), f"Expected 1:2 TP {expected_tp}, got {mss['tp_price']}"
    print(f"[PASS] 1M Sell Market Structure Shift Retest test passed! (Entry: {mss['entry_price']}, SL: {mss['sl_price']}, TP: {mss['tp_price']})")


def test_1m_market_structure_shift_buy():
    """
    Test 1-Minute Buy Scenario:
      1. 15M Bullish FVG active in range [1.1000, 1.1030].
      2. Price dips into FVG to 1.1015, bounces to swing high at 1.1025.
      3. Price drops to valley low at 1.1005.
      4. A 1M candle closes above 1.1025 (MSS Breakout).
      5. Subsequent 1M candle pulls back down and tests 1.1025 (Retest Entry).
      6. Expected Entry = 1.1025, SL = 1.1005, TP = 1.1065 (1:2.0 RR).
    """
    fvg = {"type": "BULLISH", "direction": 1, "top": 1.1030, "bottom": 1.1000}

    dates = pd.date_range("2026-08-25 10:00", periods=16, freq="1min")
    # 0..4: baseline
    # 5: drop into FVG (low=1.1015, high=1.1022)
    # 6: bounce to 1.1025 (high=1.1025, low=1.1018) -> Pivot Swing High!
    # 7: push down (high=1.1020, low=1.1010)
    # 8: push to valley 1.1005 (high=1.1012, low=1.1005) -> Swing Valley!
    # 9: bounce (high=1.1015, low=1.1008)
    # 10: rally (high=1.1022, low=1.1015)
    # 11: breakout close above 1.1025 (high=1.1028, low=1.1018, close=1.1026) -> MSS Breakout!
    # 12: continuation high (high=1.1032, low=1.1024, close=1.1028)
    # 13: pullback test (high=1.1028, low=1.1024, close=1.1026) -> Retest Touch!
    # 14: closed bar confirming retest support (high=1.1028, low=1.1025, close=1.1027)
    # 15: active forming bar

    opens  = [1.1040]*5 + [1.1022, 1.1018, 1.1022, 1.1012, 1.1008, 1.1015, 1.1020, 1.1026, 1.1028, 1.1026, 1.1027]
    highs  = [1.1045]*5 + [1.1022, 1.1025, 1.1020, 1.1012, 1.1015, 1.1022, 1.1028, 1.1032, 1.1028, 1.1028, 1.1028]
    lows   = [1.1035]*5 + [1.1015, 1.1018, 1.1010, 1.1005, 1.1008, 1.1015, 1.1018, 1.1024, 1.1024, 1.1025, 1.1025]
    closes = [1.1040]*5 + [1.1018, 1.1024, 1.1012, 1.1006, 1.1014, 1.1020, 1.1026, 1.1028, 1.1026, 1.1027, 1.1027]

    df_1m = pd.DataFrame({
        "datetime": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [100] * 16,
    })

    mss = detect_market_structure_shift(df_1m, fvg=fvg, trend_side=1, pip_size=0.0001, require_retest=True)
    assert mss is not None, "Should detect Market Structure Shift Retest to buy-side"
    assert mss["signal"] == 1
    assert round(mss["sl_price"], 4) == 1.1005, f"Expected SL 1.1005, got {mss['sl_price']}"
    assert round(mss["entry_price"], 4) == 1.1025, f"Expected Entry 1.1025, got {mss['entry_price']}"
    risk = mss["entry_price"] - mss["sl_price"]
    expected_tp = mss["entry_price"] + (2.0 * risk)
    assert round(mss["tp_price"], 5) == round(expected_tp, 5), f"Expected 1:2 TP {expected_tp}, got {mss['tp_price']}"
    print(f"[PASS] 1M Buy Market Structure Shift Retest test passed! (Entry: {mss['entry_price']}, SL: {mss['sl_price']}, TP: {mss['tp_price']})")


def test_pip_size_detection():
    """Verify 0.0001 pip detection for EURUSD, GBPUSD, AUDUSD."""
    assert _detect_pip_size(symbol="EURUSD") == 0.0001
    assert _detect_pip_size(symbol="GBPUSD") == 0.0001
    assert _detect_pip_size(symbol="AUDUSD") == 0.0001
    print("[PASS] Pip size detection test passed!")


def test_min_sl_filter():
    """Verify that MSS signals with Stop Loss < 4.0 pips (e.g. 0.7 pips) are filtered out."""
    fvg = {"type": "BULLISH", "direction": 1, "top": 0.7185, "bottom": 0.7180}
    dates = pd.date_range("2026-08-25 10:00", periods=10, freq="1min")
    opens =  [0.7182]*5 + [0.71822, 0.71818, 0.71816, 0.71820, 0.71825]
    highs =  [0.7183]*5 + [0.71822, 0.71820, 0.71818, 0.71822, 0.71827]
    lows =   [0.7181]*5 + [0.71818, 0.71815, 0.71815, 0.71816, 0.71820]
    closes = [0.7182]*5 + [0.71820, 0.71816, 0.71818, 0.71825, 0.71825]

    df_micro = pd.DataFrame({
        "datetime": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [100] * 10,
    })

    mss = detect_market_structure_shift(df_micro, fvg=fvg, trend_side=1, pip_size=0.0001, min_sl_pips=4.0)
    assert mss is None, "MSS with 0.7 pips SL should be filtered out by min_sl_pips=4.0!"
    print("[PASS] Minimum 4.0 pips Stop Loss filter test passed!")


if __name__ == "__main__":
    test_15m_fvg_detection()
    test_1m_market_structure_shift_sell()
    test_1m_market_structure_shift_buy()
    test_pip_size_detection()
    test_min_sl_filter()
    print("\nALL 15M FVG + 1M MSS STRATEGY TESTS PASSED!")
