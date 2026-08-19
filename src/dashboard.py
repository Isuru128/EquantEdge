"""
dashboard.py
EquantEdge - Multi-Asset Quant Trading Dashboard

A comprehensive, dark-mode quant trading UI displaying:
  * Multi-Symbol Selector & Real-Time Monitoring (GBPUSD, XAUUSD, EURUSD on M1)
  * Real-Time XGBoost ML Confidence Filter & Meta-Labeling
  * Balance, Equity, Floating P&L, and Closed Realized P&L
  * Multi-Timezone Dual Clock (Local + America/New_York with automatic EDT/EST)
  * Active Trading Session Detector (20:00-00:00, 02:00-05:00, 07:00-11:00 NY)
  * Live Executing Positions across all symbols with Breakeven Trailing Tracker
  * Live Orderbook / Market Depth Ladder with Bid/Ask Spread
  * Interactive Candlestick Chart with EMA-50, EMA-200, and Liquidity Sweep Markers
  * Session-by-Session P&L Analytics (Asia / London / New York)
  * Multi-Asset Risk & Dynamic Lot Size Calculator

Run from the project root:
    python -m src.dashboard          # Demo mode (Offline / Simulated)
    python -m src.dashboard --live   # Connects to running MT5 terminal
"""

import sys
import math
import os
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

# Matplotlib / mplfinance for embedded candlestick chart and analytics
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import mplfinance as mpf

# Set Windows AppUserModelID before any GUI initialization to replace Tkinter feather icon on the taskbar
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("EquantEdge.MultiAssetTerminal.App.1.0")
    except Exception:
        pass

# ---- Paths & Settings --------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
LIVE_MODE = "--live" in sys.argv

# Multi-Symbol Configuration
SUPPORTED_SYMBOLS = ["GBPUSD", "XAUUSD", "EURUSD"]
DEFAULT_SYMBOL = "XAUUSD"


# ---- Timezone & Trading Sessions (New York) ----------------------------------
NY_TZ = ZoneInfo("America/New_York")

TRADING_SESSIONS_NY = [
    {"name": "Asia",     "start": (20, 0), "end": (24, 0), "code": "ASIA",     "color": "#8A2BE2", "time_range": "20:00 - 00:00 NY"},
    {"name": "London",   "start": (2, 0),  "end": (5, 0),  "code": "LONDON",   "color": "#00C8A0", "time_range": "02:00 - 05:00 NY"},
    {"name": "New York", "start": (7, 0),  "end": (11, 0), "code": "NEW_YORK", "color": "#FF8C00", "time_range": "07:00 - 11:00 NY"},
]

# ---- Colour Palette (Cyberpunk / Modern Quant Dark Theme) --------------------
BG         = "#0B0E14"
PANEL      = "#121722"
PANEL2     = "#1A2130"
BORDER     = "#263248"
ACCENT     = "#00F0FF"   # Neon Cyan
ACCENT2    = "#A855F7"   # Purple
ACCENT3    = "#F43F5E"   # Neon Rose
BUY_CLR    = "#10B981"   # Emerald Green
SELL_CLR   = "#EF4444"   # Red
TEXT       = "#F1F5F9"
TEXT_DIM   = "#94A3B8"
GOLD       = "#F59E0B"
WHITE      = "#FFFFFF"

# ---- Fonts -------------------------------------------------------------------
F_TITLE    = ("Segoe UI", 15, "bold")
F_SECTION  = ("Segoe UI", 11, "bold")
F_LABEL    = ("Segoe UI", 9)
F_LABEL_B  = ("Segoe UI", 9, "bold")
F_MONO     = ("Consolas", 9)
F_MONO_B   = ("Consolas", 9, "bold")
F_CARD_NUM = ("Segoe UI", 15, "bold")
F_CARD_LBL = ("Segoe UI", 8)
F_BTN      = ("Segoe UI", 9, "bold")

# ---- mplfinance dark style ---------------------------------------------------
CHART_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up=BUY_CLR, down=SELL_CLR,
        edge={"up": BUY_CLR, "down": SELL_CLR},
        wick={"up": BUY_CLR, "down": SELL_CLR},
        volume={"up": BUY_CLR, "down": SELL_CLR},
    ),
    facecolor=PANEL,
    figcolor=PANEL,
    gridcolor=BORDER,
    rc={
        "axes.labelcolor":  TEXT_DIM,
        "axes.edgecolor":   BORDER,
        "xtick.color":      TEXT_DIM,
        "ytick.color":      TEXT_DIM,
        "figure.facecolor": PANEL,
        "axes.facecolor":   PANEL,
        "text.color":       TEXT,
        "font.family":      "Consolas",
        "font.size":        8,
    }
)


# ==============================================================================
#  TIME & SESSION UTILITIES
# ==============================================================================

def get_ny_now() -> datetime:
    """Return current localized time in New York."""
    return datetime.now(tz=NY_TZ)


def get_active_session_info() -> dict:
    """Check if current NY time is within any of the 3 configured sessions."""
    ny_now = get_ny_now()
    now_mins = ny_now.hour * 60 + ny_now.minute

    for sess in TRADING_SESSIONS_NY:
        start_mins = sess["start"][0] * 60 + sess["start"][1]
        end_mins = sess["end"][0] * 60 + sess["end"][1]

        if start_mins <= now_mins < end_mins:
            return {
                "in_session": True,
                "name": sess["name"],
                "code": sess["code"],
                "color": sess["color"],
                "time_range": sess.get("time_range", f"{sess['start'][0]:02d}:{sess['start'][1]:02d} - {sess['end'][0]:02d}:{sess['end'][1]:02d} NY"),
            }

    return {
        "in_session": False,
        "name": "Off-Hours",
        "code": "OFF_HOURS",
        "color": "#64748B",
        "time_range": "Closed / Out of Session",
    }


def classify_trade_session(trade_time) -> str:
    """Classify trade timestamp into Asia, London, New York, or Off-Hours."""
    try:
        if isinstance(trade_time, (int, float)):
            dt_ny = datetime.fromtimestamp(trade_time, tz=timezone.utc).astimezone(NY_TZ)
        elif isinstance(trade_time, (datetime, pd.Timestamp)):
            if trade_time.tzinfo is None:
                dt_ny = trade_time.replace(tzinfo=timezone.utc).astimezone(NY_TZ)
            else:
                dt_ny = trade_time.astimezone(NY_TZ)
        else:
            return "Off-Hours"
    except Exception:
        return "Off-Hours"

    mins = dt_ny.hour * 60 + dt_ny.minute
    for sess in TRADING_SESSIONS_NY:
        start_mins = sess["start"][0] * 60 + sess["start"][1]
        end_mins = sess["end"][0] * 60 + sess["end"][1]
        if start_mins <= mins < end_mins:
            return sess["name"]

    return "Off-Hours"


# ==============================================================================
#  DATA LAYER & STRATEGY ENRICHMENT
# ==============================================================================

def load_simulated_candles(symbol: str, count: int = 200) -> pd.DataFrame:
    """Generate realistic 1-minute simulated candles for offline testing."""
    times = pd.date_range(end=datetime.now(), periods=count, freq="1min")
    base_prices = {"XAUUSD": 4340.0, "EURUSD": 1.1585, "GBPUSD": 1.3540}
    scale_steps = {"XAUUSD": 0.50, "EURUSD": 0.0001, "GBPUSD": 0.0001}

    base = base_prices.get(symbol, 100.0)
    step = scale_steps.get(symbol, 0.01)

    records = []
    cur = base
    for i, t in enumerate(times):
        chg = (math.sin(i * 0.15) + (0.5 - (i % 5) * 0.25)) * step * 2.0
        open_p = cur
        close_p = open_p + chg
        high_p = max(open_p, close_p) + abs(chg) * 0.4
        low_p = min(open_p, close_p) - abs(chg) * 0.4
        cur = close_p
        records.append({
            "datetime": t,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": 100 + (i % 20) * 10,
        })
    return pd.DataFrame(records)


def enrich_symbol_data(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    """Compute Liquidity Sweep strategy indicators, signals, and ML predictions."""
    if __package__ is None or __package__ == "":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.strategy import generate_signals, get_latest_signal, MA_PERIOD, HTF_EMA_PERIOD
    else:
        from .strategy import generate_signals, get_latest_signal, MA_PERIOD, HTF_EMA_PERIOD

    df = generate_signals(df, symbol=symbol)
    sig_info = get_latest_signal(df, use_ml=True)
    return df, sig_info


def fetch_multi_symbol_state(selected_symbol: str = DEFAULT_SYMBOL) -> dict:
    """
    Fetch live candles and signals for all 3 symbols (GBPUSD, XAUUSD, EURUSD),
    plus active account balance, open positions, orderbook, and historical deals from MT5.
    """
    if not LIVE_MODE:
        multi_data = {}
        for sym in SUPPORTED_SYMBOLS:
            raw_df = load_simulated_candles(sym, 200)
            df, sig = enrich_symbol_data(raw_df, sym)
            multi_data[sym] = {"df": df, "sig": sig}

        return {
            "multi_data": multi_data,
            "active_symbol": selected_symbol,
            "account": {
                "balance": 10000.0,
                "equity": 10000.0,
                "currency": "USD",
                "profit": 0.0,
                "login": "SIMULATION",
                "server": "Offline Demo",
                "leverage": 100,
            },
            "open_positions": [],
            "executed_deals": [],
            "closed_pnl_today": 0.0,
            "orderbook": [],
        }

    import MetaTrader5 as mt5
    if __package__ is None or __package__ == "":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.connect import connect, get_candles
    else:
        from .connect import connect, get_candles

    acc = connect(silent=True)
    if not acc:
        raise RuntimeError("Failed to connect to MT5.")

    multi_data = {}
    for sym in SUPPORTED_SYMBOLS:
        try:
            raw_df = get_candles(sym, mt5.TIMEFRAME_M1, 250)
            if not raw_df.empty:
                df, sig = enrich_symbol_data(raw_df, sym)
                multi_data[sym] = {"df": df, "sig": sig}
        except Exception:
            pass

    account_info = {
        "login": acc.login,
        "server": acc.server,
        "currency": acc.currency,
        "balance": acc.balance,
        "equity": acc.equity,
        "margin": acc.margin,
        "free_margin": acc.margin_free,
        "profit": acc.profit,
        "leverage": acc.leverage,
    }

    server_offset_seconds = 0
    tick = mt5.symbol_info_tick(selected_symbol)
    if tick:
        server_offset_seconds = round((tick.time - datetime.now(timezone.utc).timestamp()) / 3600.0) * 3600

    positions = mt5.positions_get()
    open_positions = []
    if positions:
        for p in positions:
            pos_utc_ts = p.time - server_offset_seconds
            pos_dt_ny = datetime.fromtimestamp(pos_utc_ts, tz=timezone.utc).astimezone(NY_TZ)
            open_positions.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "time": pos_dt_ny.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "price_current": p.price_current,
                "swap": p.swap,
                "profit": p.profit,
                "comment": p.comment,
            })

    now_utc = datetime.now(timezone.utc)
    from_date = now_utc - timedelta(days=7)
    deals = mt5.history_deals_get(from_date, now_utc + timedelta(days=1))
    executed_deals = []
    closed_pnl_today = 0.0
    today_ny_date = datetime.now(tz=NY_TZ).date()

    open_deal_times = {}
    if deals:
        for d in deals:
            if d.entry == 0:
                open_deal_times[d.position_id] = d.time

        for d in deals:
            open_raw_time = open_deal_times.get(d.position_id, d.time)
            utc_open_ts = open_raw_time - server_offset_seconds
            dt_open_ny = datetime.fromtimestamp(utc_open_ts, tz=timezone.utc).astimezone(NY_TZ)
            session_tag = classify_trade_session(utc_open_ts)

            deal_type_str = "BUY" if d.type == mt5.DEAL_TYPE_BUY else ("SELL" if d.type == mt5.DEAL_TYPE_SELL else "OTHER")
            is_closing_deal = (d.entry == 1) or (d.profit != 0)
            if is_closing_deal and dt_open_ny.date() == today_ny_date:
                closed_pnl_today += d.profit

            executed_deals.append({
                "ticket": d.ticket,
                "order": d.order,
                "time": dt_open_ny.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": d.symbol,
                "type": deal_type_str,
                "entry": "IN" if d.entry == 0 else ("OUT" if d.entry == 1 else "IN/OUT"),
                "volume": d.volume,
                "price": d.price,
                "profit": d.profit,
                "commission": d.commission,
                "session": session_tag,
                "comment": d.comment,
            })

    orderbook = []
    mt5.market_book_add(selected_symbol)
    book = mt5.market_book_get(selected_symbol)
    if book:
        for item in book:
            orderbook.append({
                "type": "SELL" if item.type == mt5.BOOK_TYPE_SELL else "BUY",
                "price": item.price,
                "volume": item.volume,
            })
    else:
        tick = mt5.symbol_info_tick(selected_symbol)
        if tick:
            step = 0.10 if "XAU" in selected_symbol else 0.0001
            for i in range(5, 0, -1):
                orderbook.append({"type": "SELL", "price": round(tick.ask + i * step, 5), "volume": 10 + i * 4})
            orderbook.append({"type": "SELL", "price": round(tick.ask, 5), "volume": 35})
            orderbook.append({"type": "BUY", "price": round(tick.bid, 5), "volume": 40})
            for i in range(1, 6):
                orderbook.append({"type": "BUY", "price": round(tick.bid - i * step, 5), "volume": 12 + i * 3})

    mt5.shutdown()

    return {
        "multi_data": multi_data,
        "active_symbol": selected_symbol,
        "account": account_info,
        "open_positions": open_positions,
        "executed_deals": executed_deals,
        "closed_pnl_today": closed_pnl_today,
        "orderbook": orderbook,
    }


# ==============================================================================
#  CUSTOM WIDGETS
# ==============================================================================

class Card(tk.Frame):
    """Sleek dark container with modern subtle border."""
    def __init__(self, parent, bg_color=PANEL, **kw):
        super().__init__(parent, bg=bg_color, relief="flat",
                         highlightbackground=BORDER, highlightthickness=1,
                         padx=12, pady=10, **kw)


class StatCard(tk.Frame):
    """Refined KPI metric card with badge/subtext support."""
    def __init__(self, parent, label, value, subtext="", color=ACCENT, **kw):
        super().__init__(parent, bg=PANEL2, relief="flat",
                         highlightbackground=BORDER, highlightthickness=1,
                         padx=12, pady=8, **kw)
        self._val_var = tk.StringVar(value=value)
        self._sub_var = tk.StringVar(value=subtext)

        header_frame = tk.Frame(self, bg=PANEL2)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text=label.upper(), font=F_CARD_LBL, bg=PANEL2, fg=TEXT_DIM).pack(side="left")

        self._lbl = tk.Label(self, textvariable=self._val_var,
                             font=F_CARD_NUM, bg=PANEL2, fg=color)
        self._lbl.pack(anchor="w", pady=(2, 0))

        if subtext:
            self._sub_lbl = tk.Label(self, textvariable=self._sub_var,
                                     font=("Segoe UI", 8), bg=PANEL2, fg=TEXT_DIM)
            self._sub_lbl.pack(anchor="w")
        else:
            self._sub_lbl = None

    def update_value(self, v, sub=None, color=None):
        self._val_var.set(v)
        if sub is not None:
            self._sub_var.set(sub)
        if color:
            self._lbl.config(fg=color)


class MultiSymbolCard(tk.Frame):
    """Mini overview card for a specific asset (GBPUSD, XAUUSD, EURUSD)."""
    def __init__(self, parent, symbol: str, on_click_callback=None, **kw):
        super().__init__(parent, bg=PANEL2, relief="flat",
                         highlightbackground=BORDER, highlightthickness=1,
                         padx=10, pady=6, cursor="hand2", **kw)
        self.symbol = symbol
        self._callback = on_click_callback

        top = tk.Frame(self, bg=PANEL2)
        top.pack(fill="x")
        self._sym_lbl = tk.Label(top, text=symbol, font=("Segoe UI", 10, "bold"), bg=PANEL2, fg=ACCENT)
        self._sym_lbl.pack(side="left")

        self._sig_badge = tk.Label(top, text="SCANNING", font=("Segoe UI", 7, "bold"), bg=BORDER, fg=TEXT, padx=4, pady=1)
        self._sig_badge.pack(side="right")

        self._price_var = tk.StringVar(value="--")
        tk.Label(self, textvariable=self._price_var, font=("Consolas", 12, "bold"), bg=PANEL2, fg=TEXT).pack(anchor="w", pady=(1, 0))

        self._ml_var = tk.StringVar(value="ML: --")
        tk.Label(self, textvariable=self._ml_var, font=("Segoe UI", 8), bg=PANEL2, fg=TEXT_DIM).pack(anchor="w")

        for w in (self, top, self._sym_lbl, self._sig_badge):
            w.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        if self._callback:
            self._callback(self.symbol)

    def set_active_border(self, is_active: bool):
        clr = ACCENT if is_active else BORDER
        self.config(highlightbackground=clr, highlightthickness=2 if is_active else 1)

    def update_data(self, price: float, sig: dict):
        digits = 2 if "XAU" in self.symbol else 5
        self._price_var.set(f"{price:.{digits}f}")

        sig_val = sig.get("signal", 0)
        prob = sig.get("xgb_prob")
        prob_str = f"ML: {prob*100:.1f}%" if prob is not None else "ML: Ready"
        self._ml_var.set(prob_str)

        if sig_val == 1:
            self._sig_badge.config(text="🟢 BUY SETUP", bg=BUY_CLR, fg=WHITE)
        elif sig_val == -1:
            self._sig_badge.config(text="🔴 SELL SETUP", bg=SELL_CLR, fg=WHITE)
        else:
            self._sig_badge.config(text="WAITING", bg=BORDER, fg=TEXT_DIM)


class HRule(tk.Frame):
    """Single-pixel horizontal line divider."""
    def __init__(self, parent, color=BORDER, **kw):
        super().__init__(parent, bg=color, height=1, **kw)


# ==============================================================================
#  PRICE CANDLESTICK CHART
# ==============================================================================

class CandlestickChart(tk.Frame):
    """Candlestick chart widget with EMA-50, EMA-200, and Liquidity Sweep markers."""
    BARS_TO_SHOW = 100

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._df = None
        self._symbol = DEFAULT_SYMBOL
        self._fig = None
        self._canvas = None
        self._toolbar = None

        self._chart_box = tk.Frame(self, bg=PANEL)
        self._chart_box.pack(side="top", fill="both", expand=True)

        self._toolbar_box = tk.Frame(self, bg=PANEL2)
        self._toolbar_box.pack(side="bottom", fill="x")

        self._build_empty()

    def _build_empty(self):
        fig, ax = plt.subplots(facecolor=PANEL)
        ax.set_facecolor(PANEL)
        ax.text(0.5, 0.5, "Initializing Price Chart...",
                ha="center", va="center", color=TEXT_DIM,
                fontsize=11, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        self._attach(fig)

    def _attach(self, fig):
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._toolbar:
            self._toolbar.destroy()
            self._toolbar = None
        if self._fig:
            plt.close(self._fig)

        self._fig = fig

        canvas = FigureCanvasTkAgg(fig, master=self._chart_box)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self._toolbar_box)
        toolbar.configure(background=PANEL2)
        for child in toolbar.winfo_children():
            try:
                child.configure({"background": PANEL2, "foreground": TEXT_DIM})
            except Exception:
                pass
        toolbar.update()

        self._canvas = canvas
        self._toolbar = toolbar

    def update_data(self, df: pd.DataFrame, symbol: str = DEFAULT_SYMBOL):
        self._df = df.copy()
        self._symbol = symbol
        self._render()

    def _render(self):
        if self._df is None or len(self._df) < 5:
            return

        plot_df = self._df.tail(self.BARS_TO_SHOW).copy()
        plot_df = plot_df.set_index("datetime")
        plot_df.index = pd.DatetimeIndex(plot_df.index)
        plot_df = plot_df[["open", "high", "low", "close", "volume"]].copy()

        extra_plots = []

        if "ma_50" in self._df.columns:
            ema50_vals = self._df["ma_50"].tail(self.BARS_TO_SHOW).values
            extra_plots.append(mpf.make_addplot(
                ema50_vals, color=ACCENT2, width=1.5, linestyle="-", panel=0, label="EMA 50"
            ))

        if "ema_200" in self._df.columns:
            ema200_vals = self._df["ema_200"].tail(self.BARS_TO_SHOW).values
            extra_plots.append(mpf.make_addplot(
                ema200_vals, color=GOLD, width=1.2, linestyle="--", panel=0, label="EMA 200"
            ))

        sig_series = self._df["signal"].tail(self.BARS_TO_SHOW)
        hi_series = self._df["high"].tail(self.BARS_TO_SHOW)
        lo_series = self._df["low"].tail(self.BARS_TO_SHOW)

        buy_vals = [lo * 0.9997 if s == 1 else float("nan") for lo, s in zip(lo_series.values, sig_series.values)]
        sell_vals = [hi * 1.0003 if s == -1 else float("nan") for hi, s in zip(hi_series.values, sig_series.values)]

        if any(not math.isnan(v) for v in buy_vals):
            extra_plots.append(mpf.make_addplot(
                buy_vals, type="scatter", markersize=130,
                marker="^", color=BUY_CLR, panel=0, label="BUY Sweep"
            ))
        if any(not math.isnan(v) for v in sell_vals):
            extra_plots.append(mpf.make_addplot(
                sell_vals, type="scatter", markersize=130,
                marker="v", color=SELL_CLR, panel=0, label="SELL Sweep"
            ))

        fig, axes = mpf.plot(
            plot_df,
            type="candle",
            style=CHART_STYLE,
            volume=False,
            addplot=extra_plots,
            returnfig=True,
            figsize=(12, 5.5),
            tight_layout=True,
            warn_too_much_data=99999,
            datetime_format="%H:%M",
            xrotation=0,
        )

        main_ax = axes[0]
        main_ax.set_facecolor(PANEL)
        main_ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in main_ax.spines.values():
            spine.set_edgecolor(BORDER)

        xmin, xmax = main_ax.get_xlim()
        main_ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.20)

        handles = [
            mpatches.Patch(color=ACCENT2, label="EMA 50"),
            mpatches.Patch(color=GOLD, label="EMA 200 (HTF)"),
            mpatches.Patch(color=BUY_CLR, label="BUY Liquidity Sweep"),
            mpatches.Patch(color=SELL_CLR, label="SELL Liquidity Sweep"),
        ]
        main_ax.legend(handles=handles, loc="upper left",
                       facecolor=PANEL2, edgecolor=BORDER,
                       labelcolor=TEXT, fontsize=7.5, framealpha=0.9)

        last_close = self._df["close"].iloc[-1]
        digits = 2 if "XAU" in self._symbol else 5
        main_ax.set_title(
            f"{self._symbol} M1  |  Last Close: {last_close:.{digits}f}  |  "
            f"{'MT5 LIVE CONNECTION' if LIVE_MODE else 'SIMULATED DATA'}",
            color=TEXT, fontsize=9.5, pad=8, loc="left", fontweight="bold"
        )
        self._attach(fig)


# ==============================================================================
#  MAIN DASHBOARD APPLICATION
# ==============================================================================

class EquantEdgeDashboard(tk.Tk):

    REFRESH_MS = 10_000   # 10s auto-refresh for M1

    def __init__(self):
        super().__init__()
        self.title("EquantEdge  |  Multi-Asset Quant Trading Terminal  (GBPUSD, XAUUSD, EURUSD M1)")
        self.geometry("1500x920")
        self.minsize(1200, 750)
        self.configure(bg=BG)

        # Set Application & Taskbar Icon
        icon_path_ico = os.path.join(BASE_DIR, "assets", "icon.ico")
        icon_path_png = os.path.join(BASE_DIR, "assets", "icon.png")
        if os.path.exists(icon_path_ico):
            try:
                self.iconbitmap(default=icon_path_ico)
            except Exception:
                try:
                    self.wm_iconbitmap(icon_path_ico)
                except Exception:
                    pass
        if os.path.exists(icon_path_png):
            try:
                self._app_icon_img = tk.PhotoImage(file=icon_path_png)
                self.iconphoto(True, self._app_icon_img)
            except Exception:
                pass

        # Open in Full Size (Maximized) by default
        try:
            self.state("zoomed")
        except Exception:
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            self.geometry(f"{screen_w}x{screen_h}+0+0")

        self._active_symbol = DEFAULT_SYMBOL


        self._state = {}
        self._after_id = None
        self._risk_vars = {}

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._setup_style()
        self._build_ui()

        self._load_data_async()


    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure("TV.Treeview",
                    background=PANEL, foreground=TEXT,
                    fieldbackground=PANEL, rowheight=24,
                    font=F_MONO, borderwidth=0)
        s.configure("TV.Treeview.Heading",
                    background=PANEL2, foreground=ACCENT,
                    font=("Segoe UI", 9, "bold"), relief="flat")
        s.map("TV.Treeview",
              background=[("selected", ACCENT2)],
              foreground=[("selected", WHITE)])
        s.layout("TV.Treeview", [("TV.Treeview.treearea", {"sticky": "nswe"})])

        s.configure("Dark.Vertical.TScrollbar",
                    background=PANEL2, troughcolor=PANEL,
                    arrowcolor=TEXT_DIM, borderwidth=0)
        s.configure("Dark.TNotebook", background=BG, borderwidth=0)
        s.configure("Dark.TNotebook.Tab",
                    background=PANEL, foreground=TEXT_DIM,
                    font=("Segoe UI", 9, "bold"), padding=[16, 7])
        s.map("Dark.TNotebook.Tab",
              background=[("selected", PANEL2)],
              foreground=[("selected", ACCENT)])

    def _build_ui(self):
        # 1. Top Header Bar
        self._build_header()

        # 2. Multi-Symbol Asset Monitor Strip + KPI Row
        self._build_top_monitoring_strip()

        # 3. Multi-Tab Workspace
        self._build_tabs()

        # 4. Footer Status Bar
        self._build_footer()

        # 5. Start Clock
        self._clock_tick()

    def _build_header(self):
        top = tk.Frame(self, bg=PANEL, height=58,
                       highlightbackground=BORDER, highlightthickness=1)
        top.pack(fill="x")
        top.pack_propagate(False)

        brand_frame = tk.Frame(top, bg=PANEL)
        brand_frame.pack(side="left", padx=16)

        tk.Label(brand_frame, text="⚡ EquantEdge", font=F_TITLE,
                 bg=PANEL, fg=ACCENT).pack(side="left")
        tk.Label(brand_frame, text="MULTI-ASSET QUANT", font=("Segoe UI", 7, "bold"),
                 bg=ACCENT2, fg=WHITE, padx=5, pady=1).pack(side="left", padx=8)

        mode_color = BUY_CLR if LIVE_MODE else GOLD
        mode_text = "● LIVE MT5 CONNECTED" if LIVE_MODE else "○ DEMO SIMULATION"
        tk.Label(top, text=mode_text, font=F_LABEL_B,
                 bg=PANEL, fg=mode_color).pack(side="left", padx=10)

        self._session_badge_var = tk.StringVar(value="Checking session...")
        self._session_badge = tk.Label(top, textvariable=self._session_badge_var,
                                       font=F_LABEL_B, bg=PANEL2, fg=ACCENT,
                                       padx=10, pady=3, relief="flat",
                                       highlightbackground=BORDER, highlightthickness=1)
        self._session_badge.pack(side="left", padx=10)

        self._status_var = tk.StringVar(value="Connecting...")
        tk.Label(top, textvariable=self._status_var, font=F_LABEL,
                 bg=PANEL, fg=TEXT_DIM).pack(side="left", padx=8)

        right_frame = tk.Frame(top, bg=PANEL)
        right_frame.pack(side="right", padx=16)

        tk.Button(right_frame, text="↻ Refresh", font=F_BTN,
                  bg=ACCENT2, fg=WHITE, relief="flat",
                  activebackground="#8833D7", activeforeground=WHITE,
                  padx=12, pady=3, cursor="hand2",
                  command=self._load_data_async).pack(side="right", padx=8)

        clock_box = tk.Frame(right_frame, bg=PANEL2, padx=10, pady=2,
                             highlightbackground=BORDER, highlightthickness=1)
        clock_box.pack(side="right", padx=6)

        self._clock_local_var = tk.StringVar(value="")
        self._clock_ny_var = tk.StringVar(value="")

        tk.Label(clock_box, textvariable=self._clock_ny_var, font=F_MONO_B,
                 bg=PANEL2, fg=ACCENT).pack(anchor="e")
        tk.Label(clock_box, textvariable=self._clock_local_var, font=F_MONO,
                 bg=PANEL2, fg=TEXT_DIM).pack(anchor="e")

    def _build_top_monitoring_strip(self):
        container = tk.Frame(self, bg=BG)
        container.pack(fill="x", padx=12, pady=(8, 2))

        sym_box = tk.Frame(container, bg=BG)
        sym_box.pack(side="left", fill="x", expand=True)

        self._symbol_cards = {}
        for sym in SUPPORTED_SYMBOLS:
            sc = MultiSymbolCard(sym_box, symbol=sym, on_click_callback=self._select_symbol)
            sc.pack(side="left", fill="x", expand=True, padx=3)
            self._symbol_cards[sym] = sc

        self._symbol_cards[self._active_symbol].set_active_border(True)

        kpi_box = tk.Frame(container, bg=BG)
        kpi_box.pack(side="right", fill="x", expand=True)

        self._cards = {}
        kpis = [
            ("balance",     "Account Balance",  "--", "$ USD", ACCENT),
            ("equity",      "Equity",           "--", "$ USD", WHITE),
            ("floating_pnl","Floating P&L",     "--", "Open positions", TEXT_DIM),
            ("closed_pnl",  "Today Closed P&L", "--", "Net realized", TEXT_DIM),
        ]
        for key, lbl, val, sub, clr in kpis:
            c = StatCard(kpi_box, label=lbl, value=val, subtext=sub, color=clr)
            c.pack(side="left", fill="x", expand=True, padx=3)
            self._cards[key] = c

    def _select_symbol(self, symbol: str):
        if symbol == self._active_symbol:
            return
        self._active_symbol = symbol
        for sym, card in self._symbol_cards.items():
            card.set_active_border(sym == symbol)
        self._load_data_async()

    def _build_tabs(self):
        nb = ttk.Notebook(self, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=12, pady=6)

        t_chart    = tk.Frame(nb, bg=BG)
        t_orders   = tk.Frame(nb, bg=BG)
        t_sessions = tk.Frame(nb, bg=BG)
        t_market   = tk.Frame(nb, bg=BG)
        t_risk     = tk.Frame(nb, bg=BG)

        nb.add(t_chart,    text="  📈 Chart & Level-2 Depth  ")
        nb.add(t_orders,   text="  ⚡ Open Positions & Executions  ")
        nb.add(t_sessions, text="  🕒 3 Sessions P&L Analytics  ")
        nb.add(t_market,   text="  📊 Candlestick Data  ")
        nb.add(t_risk,     text="  🧮 Multi-Asset Risk Calculator  ")

        self._build_chart_orderbook_tab(t_chart)
        self._build_orders_tab(t_orders)
        self._build_sessions_tab(t_sessions)
        self._build_market_tab(t_market)
        self._build_risk_tab(t_risk)

    def _build_chart_orderbook_tab(self, p):
        p.columnconfigure(0, weight=3)
        p.columnconfigure(1, weight=1)
        p.rowconfigure(0, weight=1)

        chart_card = Card(p)
        chart_card.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)
        chart_card.columnconfigure(0, weight=1)
        chart_card.rowconfigure(1, weight=1)

        toolbar = tk.Frame(chart_card, bg=PANEL2, padx=8, pady=4,
                           highlightbackground=BORDER, highlightthickness=1)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self._chart_title_lbl = tk.Label(toolbar, text=f"Asset: {self._active_symbol} | Timeframe: M1", font=F_LABEL_B,
                                         bg=PANEL2, fg=ACCENT)
        self._chart_title_lbl.pack(side="left", padx=4)

        tk.Label(toolbar, text="| Bars:", font=F_LABEL, bg=PANEL2, fg=TEXT_DIM).pack(side="left", padx=4)
        self._bars_var = tk.IntVar(value=100)
        for n in [60, 100, 150, 200]:
            tk.Radiobutton(
                toolbar, text=str(n), variable=self._bars_var, value=n,
                font=F_LABEL, bg=PANEL2, fg=TEXT_DIM,
                selectcolor=PANEL, activebackground=PANEL2,
                activeforeground=ACCENT, indicatoron=False,
                relief="flat", padx=8, pady=2, cursor="hand2",
                command=self._on_bars_changed
            ).pack(side="left", padx=2)

        tk.Label(toolbar, text="| Sessions:", font=F_LABEL, bg=PANEL2, fg=TEXT_DIM).pack(side="left", padx=(10, 4))
        for sess in TRADING_SESSIONS_NY:
            tk.Label(toolbar, text=f"■ {sess['name']}", font=("Segoe UI", 8),
                     bg=PANEL2, fg=sess["color"]).pack(side="left", padx=3)

        self._chart = CandlestickChart(chart_card)
        self._chart.grid(row=1, column=0, sticky="nsew")

        book_card = Card(p)
        book_card.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=4)
        book_card.columnconfigure(0, weight=1)
        book_card.rowconfigure(2, weight=1)

        self._book_title_lbl = tk.Label(book_card, text=f"📖 {self._active_symbol} Orderbook",
                                        font=F_SECTION, bg=PANEL, fg=ACCENT)
        self._book_title_lbl.grid(row=0, column=0, sticky="w")
        tk.Label(book_card, text="Live Bid/Ask Spread Depth Ladder",
                 font=F_LABEL, bg=PANEL, fg=TEXT_DIM).grid(row=1, column=0, sticky="w", pady=(0, 6))

        book_cols = ("side", "price", "volume")
        self._book_tv = ttk.Treeview(book_card, columns=book_cols, show="headings",
                                     style="TV.Treeview", selectmode="browse")
        self._book_tv.heading("side", text="Side")
        self._book_tv.heading("price", text="Price")
        self._book_tv.heading("volume", text="Depth / Vol")
        self._book_tv.column("side", width=60, anchor="center")
        self._book_tv.column("price", width=100, anchor="center")
        self._book_tv.column("volume", width=90, anchor="center")

        self._book_tv.tag_configure("ask", foreground=SELL_CLR)
        self._book_tv.tag_configure("bid", foreground=BUY_CLR)

        book_sb = ttk.Scrollbar(book_card, orient="vertical", command=self._book_tv.yview,
                                style="Dark.Vertical.TScrollbar")
        self._book_tv.configure(yscrollcommand=book_sb.set)
        self._book_tv.grid(row=2, column=0, sticky="nsew")
        book_sb.grid(row=2, column=1, sticky="ns")

    def _on_bars_changed(self):
        active_data = self._state.get("multi_data", {}).get(self._active_symbol)
        if active_data and "df" in active_data:
            self._chart.BARS_TO_SHOW = self._bars_var.get()
            self._chart.update_data(active_data["df"], symbol=self._active_symbol)

    def _build_orders_tab(self, p):
        p.columnconfigure(0, weight=1)
        p.rowconfigure(0, weight=1)
        p.rowconfigure(1, weight=1)

        top_card = Card(p)
        top_card.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))
        top_card.columnconfigure(0, weight=1)
        top_card.rowconfigure(1, weight=1)

        header1 = tk.Frame(top_card, bg=PANEL)
        header1.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        tk.Label(header1, text="⚡ Multi-Asset Active Positions (Open Orders)",
                 font=F_SECTION, bg=PANEL, fg=ACCENT).pack(side="left")
        self._pos_count_lbl = tk.Label(header1, text="0 Open Positions",
                                       font=F_LABEL_B, bg=PANEL2, fg=TEXT, padx=6, pady=1)
        self._pos_count_lbl.pack(side="right")

        pos_cols = ("ticket", "symbol", "type", "volume", "open_price", "curr_price", "sl", "tp", "pnl", "open_time")
        pos_hdrs = ("Ticket", "Symbol", "Type", "Lots", "Open Price", "Current Price", "SL", "TP", "P&L ($)", "Opened (NY Time)")
        pos_wids = (90, 80, 70, 60, 95, 95, 85, 85, 100, 150)

        self._pos_tv = ttk.Treeview(top_card, columns=pos_cols, show="headings",
                                    style="TV.Treeview", selectmode="browse")
        for c, h, w in zip(pos_cols, pos_hdrs, pos_wids):
            self._pos_tv.heading(c, text=h)
            self._pos_tv.column(c, width=w, anchor="center")

        self._pos_tv.tag_configure("profit", foreground=BUY_CLR)
        self._pos_tv.tag_configure("loss", foreground=SELL_CLR)

        pos_sb = ttk.Scrollbar(top_card, orient="vertical", command=self._pos_tv.yview,
                               style="Dark.Vertical.TScrollbar")
        self._pos_tv.configure(yscrollcommand=pos_sb.set)
        self._pos_tv.grid(row=1, column=0, sticky="nsew")
        pos_sb.grid(row=1, column=1, sticky="ns")

        bot_card = Card(p)
        bot_card.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))
        bot_card.columnconfigure(0, weight=1)
        bot_card.rowconfigure(1, weight=1)

        header2 = tk.Frame(bot_card, bg=PANEL)
        header2.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        tk.Label(header2, text="📋 Executed Orders History (Closed Deals Today)",
                 font=F_SECTION, bg=PANEL, fg=ACCENT).pack(side="left")
        self._deals_pnl_lbl = tk.Label(header2, text="Realized P&L: $0.00",
                                       font=F_LABEL_B, bg=PANEL2, fg=TEXT, padx=6, pady=1)
        self._deals_pnl_lbl.pack(side="right")

        deal_cols = ("ticket", "time", "symbol", "type", "entry", "volume", "price", "profit", "commission", "session")
        deal_hdrs = ("Deal #", "Time (NY Time)", "Symbol", "Type", "Entry/Exit", "Lots", "Fill Price", "Realized P&L ($)", "Commission", "Trading Session")
        deal_wids = (80, 150, 80, 65, 75, 60, 95, 120, 85, 140)

        self._deal_tv = ttk.Treeview(bot_card, columns=deal_cols, show="headings",
                                     style="TV.Treeview", selectmode="browse")
        for c, h, w in zip(deal_cols, deal_hdrs, deal_wids):
            self._deal_tv.heading(c, text=h)
            self._deal_tv.column(c, width=w, anchor="center")

        self._deal_tv.tag_configure("profit", foreground=BUY_CLR)
        self._deal_tv.tag_configure("loss", foreground=SELL_CLR)

        deal_sb = ttk.Scrollbar(bot_card, orient="vertical", command=self._deal_tv.yview,
                                style="Dark.Vertical.TScrollbar")
        self._deal_tv.configure(yscrollcommand=deal_sb.set)
        self._deal_tv.grid(row=1, column=0, sticky="nsew")
        deal_sb.grid(row=1, column=1, sticky="ns")

    def _build_sessions_tab(self, p):
        for col_idx in range(len(TRADING_SESSIONS_NY)):
            p.columnconfigure(col_idx, weight=1)
        p.rowconfigure(0, weight=0)
        p.rowconfigure(1, weight=1)

        self._session_widgets = {}
        for i, sess in enumerate(TRADING_SESSIONS_NY):
            card = Card(p)
            card.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)

            h = tk.Frame(card, bg=PANEL)
            h.pack(fill="x")
            tk.Label(h, text=f"● {sess['name']}", font=F_SECTION,
                     bg=PANEL, fg=sess["color"]).pack(side="left")
            time_str = sess.get("time_range", f"{sess['start'][0]:02d}:{sess['start'][1]:02d} - {sess['end'][0]:02d}:{sess['end'][1]:02d} NY")
            tk.Label(h, text=time_str, font=F_LABEL_B, bg=PANEL2, fg=TEXT, padx=6, pady=1).pack(side="right")

            HRule(card, color=BORDER).pack(fill="x", pady=8)

            vars_dict = {}
            for k, lbl in [
                ("pnl",      "Net Realized P&L:"),
                ("trades",   "Total Trades:"),
                ("win_rate", "Win Rate:"),
                ("wins",     "Winning / Losing Trades:"),
                ("status",   "Session Status:"),
            ]:
                row = tk.Frame(card, bg=PANEL)
                row.pack(fill="x", pady=3)
                tk.Label(row, text=lbl, font=F_LABEL, bg=PANEL, fg=TEXT_DIM).pack(side="left")
                v = tk.StringVar(value="--")
                vars_dict[k] = v
                tk.Label(row, textvariable=v, font=F_MONO_B, bg=PANEL, fg=TEXT).pack(side="right")

            self._session_widgets[sess["name"]] = vars_dict

        chart_card = Card(p)
        chart_card.grid(row=1, column=0, columnspan=len(TRADING_SESSIONS_NY), sticky="nsew", padx=4, pady=4)
        chart_card.columnconfigure(0, weight=1)
        chart_card.rowconfigure(1, weight=1)

        tk.Label(chart_card, text="📊 Session P&L Distribution Comparison",
                 font=F_SECTION, bg=PANEL, fg=ACCENT).grid(row=0, column=0, sticky="w", pady=(0, 4))

        self._sess_chart_frame = tk.Frame(chart_card, bg=PANEL)
        self._sess_chart_frame.grid(row=1, column=0, sticky="nsew")
        self._sess_fig, self._sess_ax = plt.subplots(figsize=(10, 3), facecolor=PANEL)
        self._sess_canvas = FigureCanvasTkAgg(self._sess_fig, master=self._sess_chart_frame)
        self._sess_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _render_session_chart(self, pnl_by_session: dict):
        self._sess_ax.clear()
        self._sess_ax.set_facecolor(PANEL)

        sessions = [s["name"] for s in TRADING_SESSIONS_NY]
        pnls = [pnl_by_session.get(s, 0.0) for s in sessions]
        colors = [BUY_CLR if p >= 0 else SELL_CLR for p in pnls]

        y_pos = range(len(sessions))
        bars = self._sess_ax.barh(y_pos, pnls, color=colors, height=0.45, edgecolor=BORDER)

        self._sess_ax.set_yticks(y_pos)
        self._sess_ax.set_yticklabels(sessions, color=TEXT, fontsize=9, fontweight="bold")
        self._sess_ax.axvline(0, color=TEXT_DIM, linestyle="--", linewidth=0.8)

        for bar, pnl in zip(bars, pnls):
            width = bar.get_width()
            align = "left" if width >= 0 else "right"
            offset = 2 if width >= 0 else -2
            self._sess_ax.text(width + offset, bar.get_y() + bar.get_height()/2,
                               f"${pnl:+,.2f}", va="center", ha=align,
                               color=BUY_CLR if pnl >= 0 else SELL_CLR,
                               fontweight="bold", fontsize=9)

        for spine in self._sess_ax.spines.values():
            spine.set_edgecolor(BORDER)
        self._sess_ax.tick_params(colors=TEXT_DIM, labelsize=8)
        self._sess_ax.set_xlabel("Net Realized Profit / Loss ($ USD)", color=TEXT_DIM, fontsize=8)
        self._sess_fig.subplots_adjust(left=0.22, right=0.92, top=0.92, bottom=0.22)
        self._sess_canvas.draw()

    def _build_market_tab(self, p):
        p.columnconfigure(0, weight=1)
        p.rowconfigure(0, weight=1)

        f = Card(p)
        f.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)

        self._market_title_lbl = tk.Label(f, text=f"Recent 1-Minute Candlesticks ({self._active_symbol})",
                                          font=F_SECTION, bg=PANEL, fg=ACCENT)
        self._market_title_lbl.grid(row=0, column=0, sticky="w", pady=(0, 6))

        cols = ("datetime", "open", "high", "low", "close", "volume", "sweep_sell", "sweep_buy", "signal")
        hdrs = ("Datetime", "Open", "High", "Low", "Close", "Volume", "Sweep Sell", "Sweep Buy", "Signal")
        wids = (140, 85, 85, 85, 85, 75, 95, 95, 120)

        self._market_tv = ttk.Treeview(f, columns=cols, show="headings",
                                       style="TV.Treeview", selectmode="browse")
        for c, h, w in zip(cols, hdrs, wids):
            self._market_tv.heading(c, text=h)
            self._market_tv.column(c, width=w, anchor="center")

        self._market_tv.tag_configure("buy", foreground=BUY_CLR)
        self._market_tv.tag_configure("sell", foreground=SELL_CLR)

        vsb = ttk.Scrollbar(f, orient="vertical", command=self._market_tv.yview,
                            style="Dark.Vertical.TScrollbar")
        self._market_tv.configure(yscrollcommand=vsb.set)
        self._market_tv.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")

    def _build_risk_tab(self, p):
        outer = tk.Frame(p, bg=BG)
        outer.pack(fill="both", expand=True)

        card = Card(outer)
        card.place(relx=0.5, rely=0.5, anchor="center", width=560)

        tk.Label(card, text="🧮 Multi-Asset Position Size & Risk Manager",
                 font=F_SECTION, bg=PANEL, fg=ACCENT).pack(anchor="w")
        tk.Label(card, text="Fixed-fractional lot sizing  |  1:1.5 Risk-Reward Ratio (M1)",
                 font=F_LABEL, bg=PANEL, fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        HRule(card).pack(fill="x", pady=8)

        fields = [
            ("Symbol",             "symbol",   "XAUUSD"),
            ("Account Balance ($)", "balance",  "100.0"),
            ("Risk % of Balance",  "risk_pct", "1.0"),
            ("Stop-Loss (pips)",   "sl_pips",  "15.0"),
            ("Pip Value ($/lot)",  "pip_val",  "1.0"),
            ("Min Lot",            "min_lot",  "0.01"),
            ("Max Lot",            "max_lot",  "50.0"),
            ("Lot Step",           "lot_step", "0.01"),
        ]

        for lbl_text, key, default in fields:
            row = tk.Frame(card, bg=PANEL)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=lbl_text, width=22, anchor="w",
                     font=F_LABEL, bg=PANEL, fg=TEXT_DIM).pack(side="left")
            var = tk.StringVar(value=default)
            self._risk_vars[key] = var
            tk.Entry(row, textvariable=var, font=F_MONO, bg=PANEL2, fg=TEXT,
                     insertbackground=TEXT, relief="flat", bd=4,
                     width=14).pack(side="left", padx=4)

        HRule(card).pack(fill="x", pady=10)
        tk.Button(card, text="Calculate Position Size", font=F_BTN,
                  bg=ACCENT, fg=BG, relief="flat",
                  activebackground="#00C4D6", activeforeground=BG,
                  padx=18, pady=6, cursor="hand2",
                  command=self._calc_lot).pack()

        self._lot_result = tk.StringVar(value="")
        tk.Label(card, textvariable=self._lot_result,
                 font=F_CARD_NUM, bg=PANEL, fg=ACCENT).pack(pady=6)

        self._rr_result = tk.StringVar(value="")
        tk.Label(card, textvariable=self._rr_result,
                 font=F_MONO, bg=PANEL, fg=TEXT_DIM).pack()

    def _calc_lot(self):
        try:
            bal  = float(self._risk_vars["balance"].get())
            rp   = float(self._risk_vars["risk_pct"].get())
            sl   = float(self._risk_vars["sl_pips"].get())
            pv   = float(self._risk_vars["pip_val"].get())
            mn   = float(self._risk_vars["min_lot"].get())
            mx   = float(self._risk_vars["max_lot"].get())
            step = float(self._risk_vars["lot_step"].get())

            if sl <= 0 or pv <= 0 or not (0 < rp <= 100):
                raise ValueError("Inputs out of range.")

            lots = math.floor((bal * rp / 100) / (sl * pv) / step) * step
            lots = round(max(mn, min(lots, mx)), 2)
            risk_usd = sl * pv * lots
            reward_usd = risk_usd * 1.5

            self._lot_result.set(f"Calculated Volume: {lots:.2f} Lots")
            self._rr_result.set(f"Risk: ${risk_usd:,.2f}   |   Take-Profit (1:1.5): ${reward_usd:,.2f}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _build_footer(self):
        sbar = tk.Frame(self, bg=PANEL, height=26,
                        highlightbackground=BORDER, highlightthickness=1)
        sbar.pack(fill="x", side="bottom")
        tk.Label(sbar,
                 text="  Active Pairs: GBPUSD, XAUUSD, EURUSD  |  Timeframe: M1  |  Strategy: Liquidity Sweep Reversal + XGBoost ML"
                      "  |  Timezone: America/New_York (Auto EDT/EST)  |  EquantEdge Pro Terminal",
                 font=("Segoe UI", 8), bg=PANEL, fg=TEXT_DIM).pack(side="left")

    def _load_data_async(self):
        self._status_var.set(f"Fetching live data for {self._active_symbol} & multi-pairs...")
        def worker():
            try:
                state = fetch_multi_symbol_state(self._active_symbol)
                self.after(0, self._populate_all, state)
            except Exception as exc:
                err_msg = str(exc)
                def on_error(msg=err_msg):
                    self._status_var.set(f"Error: {msg}")
                    if self._after_id:
                        try:
                            self.after_cancel(self._after_id)
                        except Exception:
                            pass
                    self._after_id = self.after(self.REFRESH_MS, self._load_data_async)
                self.after(0, on_error)
        threading.Thread(target=worker, daemon=True).start()

    def _populate_all(self, state: dict):
        self._state = state
        multi_data = state.get("multi_data", {})
        account = state.get("account", {})
        open_positions = state.get("open_positions", [])
        executed_deals = state.get("executed_deals", [])
        closed_pnl = state.get("closed_pnl_today", 0.0)
        orderbook = state.get("orderbook", [])

        # 1. Update Multi-Symbol Mini Cards
        for sym in SUPPORTED_SYMBOLS:
            if sym in multi_data:
                df = multi_data[sym]["df"]
                sig = multi_data[sym]["sig"]
                last_price = df["close"].iloc[-1]
                if sym in self._symbol_cards:
                    self._symbol_cards[sym].update_data(last_price, sig)

        # 2. Update KPI Cards
        self._populate_stat_cards(account, closed_pnl)

        # 3. Update Chart & Orderbook for Active Symbol
        if self._active_symbol in multi_data:
            active_df = multi_data[self._active_symbol]["df"]
            self._chart_title_lbl.config(text=f"Asset: {self._active_symbol}  |  Timeframe: M1 (1-min)")
            self._book_title_lbl.config(text=f"📖 {self._active_symbol} Orderbook")
            self._market_title_lbl.config(text=f"Recent 1-Minute Candlesticks ({self._active_symbol})")

            self._chart.BARS_TO_SHOW = self._bars_var.get()
            self._chart.update_data(active_df, symbol=self._active_symbol)
            self._populate_market_table(active_df)

        self._populate_orderbook(orderbook)
        self._populate_orders_tab(open_positions, executed_deals, closed_pnl)
        self._populate_sessions_tab(executed_deals)

        if "balance" in account:
            self._risk_vars["balance"].set(str(account["balance"]))
            self._risk_vars["symbol"].set(self._active_symbol)

        self._status_var.set(f"Synced {datetime.now().strftime('%H:%M:%S')}  |  {len(open_positions)} active position(s)")

        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
        self._after_id = self.after(self.REFRESH_MS, self._load_data_async)

    def _populate_stat_cards(self, account: dict, closed_pnl: float):
        bal = account.get("balance", 0.0)
        eq = account.get("equity", 0.0)
        curr = account.get("currency", "USD")
        self._cards["balance"].update_value(f"${bal:,.2f}", sub=f"{curr} Account")
        self._cards["equity"].update_value(f"${eq:,.2f}", sub="Live Net Equity")

        floating = account.get("profit", 0.0)
        fl_clr = BUY_CLR if floating > 0 else (SELL_CLR if floating < 0 else TEXT_DIM)
        self._cards["floating_pnl"].update_value(f"{floating:+,.2f} $", sub="Active Positions", color=fl_clr)

        cl_clr = BUY_CLR if closed_pnl > 0 else (SELL_CLR if closed_pnl < 0 else TEXT_DIM)
        self._cards["closed_pnl"].update_value(f"{closed_pnl:+,.2f} $", sub="Realized Today", color=cl_clr)

    def _populate_orderbook(self, orderbook: list):
        tv = self._book_tv
        tv.delete(*tv.get_children())
        if not orderbook:
            tv.insert("", "end", values=("--", "No DOM Data", "--"))
            return

        digits = 2 if "XAU" in self._active_symbol else 5
        for item in orderbook:
            side = item.get("type", "BUY")
            tag = "ask" if side == "SELL" else "bid"
            tv.insert("", "end", values=(
                side,
                f"{item.get('price', 0):.{digits}f}",
                str(int(item.get('volume', 0))),
            ), tags=(tag,))

    def _populate_orders_tab(self, open_positions: list, executed_deals: list, closed_pnl: float):
        pos_tv = self._pos_tv
        pos_tv.delete(*pos_tv.get_children())
        self._pos_count_lbl.config(text=f"{len(open_positions)} Open Position(s)")

        if not open_positions:
            pos_tv.insert("", "end", values=("--", "--", "--", "--", "No active open positions", "--", "--", "--", "--", "--"))
        else:
            for pos in open_positions:
                pnl = pos["profit"]
                tag = "profit" if pnl >= 0 else "loss"
                digits = 2 if "XAU" in pos.get("symbol", "") else 5
                pos_tv.insert("", "end", values=(
                    str(pos["ticket"]),
                    pos["symbol"],
                    pos["type"],
                    f"{pos['volume']:.2f}",
                    f"{pos['price_open']:.{digits}f}",
                    f"{pos['price_current']:.{digits}f}",
                    f"{pos['sl']:.{digits}f}",
                    f"{pos['tp']:.{digits}f}",
                    f"{pnl:+,.2f}",
                    str(pos["time"])[:19],
                ), tags=(tag,))

        deal_tv = self._deal_tv
        deal_tv.delete(*deal_tv.get_children())
        self._deals_pnl_lbl.config(
            text=f"Total Realized: ${closed_pnl:+,.2f}",
            fg=BUY_CLR if closed_pnl >= 0 else SELL_CLR
        )

        if not executed_deals:
            deal_tv.insert("", "end", values=("--", "--", "--", "--", "--", "--", "No executed orders today yet", "--", "--", "--"))
        else:
            for deal in executed_deals[::-1]:
                pnl = deal["profit"]
                tag = "profit" if pnl > 0 else ("loss" if pnl < 0 else "")
                digits = 2 if "XAU" in deal.get("symbol", "") else 5
                deal_tv.insert("", "end", values=(
                    str(deal["ticket"]),
                    str(deal["time"])[:19],
                    deal["symbol"],
                    deal["type"],
                    deal["entry"],
                    f"{deal['volume']:.2f}",
                    f"{deal['price']:.{digits}f}",
                    f"{pnl:+,.2f}" if pnl != 0 else "$0.00",
                    f"${deal['commission']:.2f}",
                    deal["session"],
                ), tags=(tag,))

    def _populate_sessions_tab(self, executed_deals: list):
        stats = {
            sess["name"]: {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
            for sess in TRADING_SESSIONS_NY
        }

        for deal in executed_deals:
            sess_name = deal.get("session")
            is_closed = (deal.get("entry") in ("OUT", "IN/OUT")) or (deal.get("profit", 0.0) != 0.0)
            if is_closed and sess_name in stats:
                pnl = deal.get("profit", 0.0)
                stats[sess_name]["pnl"] += pnl
                stats[sess_name]["trades"] += 1
                if pnl > 0:
                    stats[sess_name]["wins"] += 1
                elif pnl < 0:
                    stats[sess_name]["losses"] += 1

        pnl_map = {}
        active_sess = get_active_session_info()

        for sess_name, data in stats.items():
            if sess_name not in self._session_widgets:
                continue
            w = self._session_widgets[sess_name]
            pnl = data["pnl"]
            trades = data["trades"]
            wins = data["wins"]
            losses = data["losses"]
            win_rate = (wins / trades * 100) if trades > 0 else 0.0

            w["pnl"].set(f"${pnl:+,.2f}")
            w["trades"].set(str(trades))
            w["win_rate"].set(f"{win_rate:.1f}%")
            w["wins"].set(f"{wins}W / {losses}L")

            is_cur = active_sess["in_session"] and active_sess["name"] == sess_name
            w["status"].set("ACTIVE NOW" if is_cur else "IDLE")

            pnl_map[sess_name] = pnl

        self._render_session_chart(pnl_map)

    def _populate_market_table(self, df: pd.DataFrame):
        tv = self._market_tv
        tv.delete(*tv.get_children())
        digits = 2 if "XAU" in self._active_symbol else 5
        for _, row in df.tail(150).iloc[::-1].iterrows():
            sig = int(row.get("signal", 0))
            vals = (
                str(row["datetime"])[:19],
                f"{row['open']:.{digits}f}",
                f"{row['high']:.{digits}f}",
                f"{row['low']:.{digits}f}",
                f"{row['close']:.{digits}f}",
                str(int(row.get("volume", 0))),
                "YES" if bool(row.get("sweep_sell", False)) else "-",
                "YES" if bool(row.get("sweep_buy", False)) else "-",
                row.get("signal_label", ""),
            )
            tag = "buy" if sig == 1 else ("sell" if sig == -1 else "")
            tv.insert("", "end", values=vals, tags=(tag,))

    def _clock_tick(self):
        try:
            if not self.winfo_exists():
                return
            local_now = datetime.now().astimezone()
            offset = local_now.utcoffset()
            if offset is not None:
                total_seconds = int(offset.total_seconds())
                sign = "+" if total_seconds >= 0 else "-"
                total_seconds = abs(total_seconds)
                hrs = total_seconds // 3600
                mins = (total_seconds % 3600) // 60
                tz_str = f"UTC{sign}{hrs:02d}:{mins:02d}" if mins else f"UTC{sign}{hrs}"
            else:
                tz_str = "UTC"
            self._clock_local_var.set(local_now.strftime(f"LOCAL: %H:%M:%S  ({tz_str})"))

            ny_now = get_ny_now()
            offset_hrs = int(ny_now.utcoffset().total_seconds() // 3600)
            tz_name = ny_now.strftime("%Z")
            self._clock_ny_var.set(ny_now.strftime(f"NY: %H:%M:%S  {tz_name} (UTC{offset_hrs:+d})"))

            sess_info = get_active_session_info()
            if sess_info["in_session"]:
                self._session_badge_var.set(f"🟢 {sess_info['name']} ({sess_info['time_range']})")
                self._session_badge.config(fg=BUY_CLR, highlightbackground=BUY_CLR)
            else:
                self._session_badge_var.set("⏸ OUTSIDE SESSIONS (EXECUTION PAUSED)")
                self._session_badge.config(fg=TEXT_DIM, highlightbackground=BORDER)

            self._clock_after_id = self.after(1000, self._clock_tick)
        except Exception:
            pass

    def destroy(self):
        try:
            if hasattr(self, "_after_id") and self._after_id:
                self.after_cancel(self._after_id)
            if hasattr(self, "_clock_after_id") and self._clock_after_id:
                self.after_cancel(self._clock_after_id)
        except Exception:
            pass
        plt.close("all")
        super().destroy()


if __name__ == "__main__":
    EquantEdgeDashboard().mainloop()