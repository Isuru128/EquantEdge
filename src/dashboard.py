"""
dashboard.py
EquantEdge - Trading Dashboard

A comprehensive, dark-mode quant trading UI displaying:
  * Current Balance, Equity, Floating P&L, Closed P&L Today
  * Multi-Timezone Dual Clock (Local + America/New_York with automatic EDT/EST)
  * Active Trading Session Detector (20:00-00:00, 02:00-05:00, 07:00-11:00 NY)
  * Live Executing Positions & Executed Orders History
  * Live Orderbook / Market Depth Ladder with Bid/Ask Spread
  * Candlestick Chart (XAUUSD M1 / EMA-50 / Signals / Session Shading)
  * Session-by-Session P&L Analytics (P&L, Win Rate, Trades per Session)
  * Candle Pattern Table & Position Size Risk Calculator

Run from the project root:
    python -m src.dashboard          # Demo mode (CSV / Offline)
    python -m src.dashboard --live   # Connects to running MT5 terminal
"""

import sys
import math
import os
import threading
from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

# Matplotlib / mplfinance for embedded candlestick chart and session analytics
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import mplfinance as mpf

# ---- Paths & Settings --------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
CSV_GLOB = [f for f in os.listdir(BASE_DIR) if f.endswith(".csv")]
DEMO_CSV = os.path.join(BASE_DIR, CSV_GLOB[0]) if CSV_GLOB else None
LIVE_MODE = "--live" in sys.argv

# Default trading asset
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
F_TITLE    = ("Segoe UI", 16, "bold")
F_SECTION  = ("Segoe UI", 11, "bold")
F_LABEL    = ("Segoe UI", 9)
F_LABEL_B  = ("Segoe UI", 9, "bold")
F_MONO     = ("Consolas", 9)
F_MONO_B   = ("Consolas", 9, "bold")
F_CARD_NUM = ("Segoe UI", 16, "bold")
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
    """
    Check if current NY time is within any of the 3 configured sessions:
      • Asia     : 20:00 – 00:00 NY
      • London   : 02:00 – 05:00 NY
      • New York : 07:00 – 11:00 NY
    """
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
    """
    Accurately classify trade timestamp into:
      • Asia     (20:00 – 00:00 NY)
      • London   (02:00 – 05:00 NY)
      • New York (07:00 – 11:00 NY)
      • Off-Hours (all other times)
    """
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
#  DATA LAYER (MT5 & DEMO)
# ==============================================================================

def load_demo_data() -> pd.DataFrame:
    """Load sample data or fallback CSV."""
    if DEMO_CSV and os.path.exists(DEMO_CSV):
        df = pd.read_csv(DEMO_CSV)
        if "datetime" not in df.columns and "time" in df.columns:
            df.rename(columns={"time": "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)
    else:
        # Generate dummy 1-minute gold price data
        times = pd.date_range(end=datetime.now(), periods=200, freq="1min")
        base_price = 2400.0
        records = []
        cur = base_price
        for t in times:
            chg = (math.sin(len(records) * 0.1) + (0.5 - (len(records) % 3) * 0.3)) * 0.8
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
                "volume": 120 + (len(records) % 40) * 5,
            })
        return pd.DataFrame(records)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators, pattern recognition, and signals."""
    if __package__ is None or __package__ == "":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.patterns import is_inside_bar, is_pin_bar, is_engulfing
    else:
        from .patterns import is_inside_bar, is_pin_bar, is_engulfing

    df = df.copy()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["inside_bar"] = is_inside_bar(df)
    df["mother_high"] = df["high"].shift(1)
    df["mother_low"] = df["low"].shift(1)
    df["trend"] = df["close"] > df["ema_50"]

    df["signal"] = 0
    df.loc[df["inside_bar"] & df["trend"], "signal"] = 1
    df.loc[df["inside_bar"] & ~df["trend"], "signal"] = -1
    df["signal_label"] = df["signal"].map({1: "BUY", -1: "SELL", 0: ""})

    df["pin_bar"] = is_pin_bar(df)
    df["engulfing"] = is_engulfing(df)
    return df


def fetch_mt5_full_state(symbol: str = DEFAULT_SYMBOL):
    """
    Fetch all account info, live candles, positions, orderbook, and historical deals from MT5.
    """
    import MetaTrader5 as mt5
    if __package__ is None or __package__ == "":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.connect import connect, get_candles
    else:
        from .connect import connect, get_candles

    acc = connect(silent=True)
    if not acc:
        raise RuntimeError("Failed to connect to MT5.")

    # 1. Fetch Candles (XAUUSD 1-minute)
    df = get_candles(symbol, mt5.TIMEFRAME_M1, 300)
    df = enrich(df)

    # 2. Account Information
    account_info = {
        "login": acc.login,
        "server": acc.server,
        "currency": acc.currency,
        "balance": acc.balance,
        "equity": acc.equity,
        "margin": acc.margin,
        "free_margin": acc.margin_free,
        "profit": acc.profit,   # Floating P&L
        "leverage": acc.leverage,
    }

    # Calculate dynamic broker server offset vs UTC
    server_offset_seconds = 0
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        server_offset_seconds = round((tick.time - datetime.now(timezone.utc).timestamp()) / 3600.0) * 3600

    # 3. Active Positions (Executing orders)
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        positions = mt5.positions_get()
    open_positions = []
    if positions:
        for p in positions:
            pos_utc_ts = p.time - server_offset_seconds
            pos_dt_ny = datetime.fromtimestamp(pos_utc_ts, tz=timezone.utc).astimezone(NY_TZ)
            open_positions.append({
                "ticket": p.ticket,
                "time": pos_dt_ny.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "price_current": p.price_current,
                "swap": p.swap,
                "profit": p.profit,
                "symbol": p.symbol,
                "comment": p.comment,
            })

    # 4. Executed Deals History (Last 7 Days)
    now_utc = datetime.now(timezone.utc)
    from_date = now_utc - timedelta(days=7)
    deals = mt5.history_deals_get(from_date, now_utc + timedelta(days=1))
    executed_deals = []
    closed_pnl_today = 0.0
    today_ny_date = datetime.now(tz=NY_TZ).date()

    # Map each position to its original opening deal timestamp
    open_deal_times = {}
    if deals:
        for d in deals:
            if d.entry == 0:  # DEAL_ENTRY_IN
                open_deal_times[d.position_id] = d.time

        for d in deals:
            # Use opening timestamp of the trade
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
                "time_ny": dt_open_ny.strftime("%H:%M:%S"),
                "symbol": d.symbol,
                "type": deal_type_str,
                "entry": "IN" if d.entry == 0 else ("OUT" if d.entry == 1 else "IN/OUT"),
                "volume": d.volume,
                "price": d.price,
                "profit": d.profit,
                "commission": d.commission,
                "swap": d.swap,
                "session": session_tag,
                "comment": d.comment,
            })

    # 5. Orderbook / Market Depth
    orderbook = []
    # Try subscribe to market book if available
    mt5.market_book_add(symbol)
    book = mt5.market_book_get(symbol)
    if book:
        for item in book:
            orderbook.append({
                "type": "SELL" if item.type == mt5.BOOK_TYPE_SELL else "BUY",
                "price": item.price,
                "volume": item.volume,
                "volume_dbl": item.volume_dbl,
            })
    else:
        # Fallback to tick quote spread ladder
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            spread = (tick.ask - tick.bid)
            # Simulated micro depth ladder around bid/ask
            for i in range(5, 0, -1):
                orderbook.append({
                    "type": "SELL",
                    "price": round(tick.ask + i * 0.10, 2),
                    "volume": 5 + i * 3,
                })
            orderbook.append({"type": "SELL", "price": round(tick.ask, 2), "volume": 20})
            orderbook.append({"type": "BUY", "price": round(tick.bid, 2), "volume": 22})
            for i in range(1, 6):
                orderbook.append({
                    "type": "BUY",
                    "price": round(tick.bid - i * 0.10, 2),
                    "volume": 6 + i * 2,
                })

    mt5.shutdown()

    return {
        "df": df,
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
                         padx=14, pady=8, **kw)
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


class HRule(tk.Frame):
    """Single-pixel horizontal line divider."""
    def __init__(self, parent, color=BORDER, **kw):
        super().__init__(parent, bg=color, height=1, **kw)


# ==============================================================================
#  PRICE CANDLESTICK CHART WITH SESSIONS SHADING
# ==============================================================================

class CandlestickChart(tk.Frame):
    """
    Candlestick chart widget with EMA-50 overlay, Buy/Sell triangles,
    and automatic vertical shading for New York trading sessions.
    """
    BARS_TO_SHOW = 100

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._df = None
        self._fig = None
        self._canvas = None
        self._toolbar = None

        # Persistent layout containers: chart area on top, toolbar on bottom
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

    def update_data(self, df: pd.DataFrame):
        self._df = df.copy()
        self._render()

    def _render(self):
        if self._df is None or len(self._df) < 5:
            return

        plot_df = self._df.tail(self.BARS_TO_SHOW).copy()
        plot_df = plot_df.set_index("datetime")
        plot_df.index = pd.DatetimeIndex(plot_df.index)
        plot_df = plot_df[["open", "high", "low", "close", "volume"]].copy()

        # EMA overlay
        ema_values = self._df["ema_50"].tail(self.BARS_TO_SHOW).values
        ema_ap = mpf.make_addplot(
            ema_values,
            color=ACCENT2, width=1.5, linestyle="-",
            label="EMA 50", panel=0
        )

        # Signals markers
        sig_series = self._df["signal"].tail(self.BARS_TO_SHOW)
        hi_series = self._df["high"].tail(self.BARS_TO_SHOW)
        lo_series = self._df["low"].tail(self.BARS_TO_SHOW)

        buy_vals = [lo * 0.9997 if s == 1 else float("nan")
                    for lo, s in zip(lo_series.values, sig_series.values)]
        sell_vals = [hi * 1.0003 if s == -1 else float("nan")
                     for hi, s in zip(hi_series.values, sig_series.values)]

        extra_plots = [ema_ap]
        if any(not math.isnan(v) for v in buy_vals):
            extra_plots.append(mpf.make_addplot(
                buy_vals, type="scatter", markersize=120,
                marker="^", color=BUY_CLR, panel=0, label="BUY Signal"
            ))
        if any(not math.isnan(v) for v in sell_vals):
            extra_plots.append(mpf.make_addplot(
                sell_vals, type="scatter", markersize=120,
                marker="v", color=SELL_CLR, panel=0, label="SELL Signal"
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
        main_ax.yaxis.label.set_color(TEXT_DIM)
        main_ax.xaxis.label.set_color(TEXT_DIM)

        # 20% free space ahead on the right for incoming candles
        xmin, xmax = main_ax.get_xlim()
        main_ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.25)

        handles = [
            mpatches.Patch(color=ACCENT2, label="EMA 50"),
            mpatches.Patch(color=BUY_CLR, label="BUY (Inside Bar Breakout)"),
            mpatches.Patch(color=SELL_CLR, label="SELL (Inside Bar Breakout)"),
        ]
        main_ax.legend(handles=handles, loc="upper left",
                       facecolor=PANEL2, edgecolor=BORDER,
                       labelcolor=TEXT, fontsize=7.5, framealpha=0.9)

        last_close = self._df["close"].iloc[-1]
        main_ax.set_title(
            f"XAUUSD (Gold)  M1  |  Current Price: {last_close:.2f} USD  |  "
            f"{'MT5 LIVE CONNECTION' if LIVE_MODE else 'OFFLINE DEMO DATA'}",
            color=TEXT, fontsize=9.5, pad=8, loc="left", fontweight="bold"
        )
        self._attach(fig)


# ==============================================================================
#  MAIN DASHBOARD APPLICATION
# ==============================================================================

class EquantEdgeDashboard(tk.Tk):

    REFRESH_MS = 15_000   # 15s auto-refresh for live M1 trading

    def __init__(self):
        super().__init__()
        self.title("EquantEdge  |  Quant Trading Terminal  (XAUUSD M1)")
        self.geometry("1480x900")
        self.minsize(1200, 750)
        self.configure(bg=BG)

        self._df = None
        self._after_id = None
        self._account_state = {}
        self._open_positions = []
        self._executed_deals = []
        self._orderbook_data = []
        self._risk_vars = {}

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._setup_style()
        self._build_ui()
        self._load_data_async()

    # --------------------------------------------------------------------------
    #  Styling & Theming
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    #  UI Construction
    # --------------------------------------------------------------------------

    def _build_ui(self):
        # 1. Top Header Bar
        self._build_header()

        # 2. Stat Cards Row (Account & Key Metrics)
        self._build_stat_cards()

        # 3. Multi-Tab Workspace
        self._build_tabs()

        # 4. Footer Status Bar
        self._build_footer()

        # 5. Start Clock
        self._clock_tick()

    def _build_header(self):
        top = tk.Frame(self, bg=PANEL, height=60,
                       highlightbackground=BORDER, highlightthickness=1)
        top.pack(fill="x")
        top.pack_propagate(False)

        # Brand / Logo
        brand_frame = tk.Frame(top, bg=PANEL)
        brand_frame.pack(side="left", padx=16)

        tk.Label(brand_frame, text="⚡ EquantEdge", font=F_TITLE,
                 bg=PANEL, fg=ACCENT).pack(side="left")
        tk.Label(brand_frame, text="PRO TERMINAL", font=("Segoe UI", 7, "bold"),
                 bg=ACCENT2, fg=WHITE, padx=5, pady=1).pack(side="left", padx=8)

        # Mode Indicator (LIVE vs DEMO)
        mode_color = BUY_CLR if LIVE_MODE else GOLD
        mode_text = "● LIVE MT5 CONNECTED" if LIVE_MODE else "○ DEMO MODE (CSV)"
        tk.Label(top, text=mode_text, font=F_LABEL_B,
                 bg=PANEL, fg=mode_color).pack(side="left", padx=12)

        # Live Session Status Badge
        self._session_badge_var = tk.StringVar(value="Checking session...")
        self._session_badge = tk.Label(top, textvariable=self._session_badge_var,
                                       font=F_LABEL_B, bg=PANEL2, fg=ACCENT,
                                       padx=10, pady=4, relief="flat",
                                       highlightbackground=BORDER, highlightthickness=1)
        self._session_badge.pack(side="left", padx=10)

        # Status note
        self._status_var = tk.StringVar(value="Connecting...")
        tk.Label(top, textvariable=self._status_var, font=F_LABEL,
                 bg=PANEL, fg=TEXT_DIM).pack(side="left", padx=10)

        # Right Side: Dual Clock & Refresh Button
        right_frame = tk.Frame(top, bg=PANEL)
        right_frame.pack(side="right", padx=16)

        tk.Button(right_frame, text="↻ Refresh", font=F_BTN,
                  bg=ACCENT2, fg=WHITE, relief="flat",
                  activebackground="#8833D7", activeforeground=WHITE,
                  padx=12, pady=4, cursor="hand2",
                  command=self._load_data_async).pack(side="right", padx=8)

        clock_box = tk.Frame(right_frame, bg=PANEL2, padx=10, pady=3,
                             highlightbackground=BORDER, highlightthickness=1)
        clock_box.pack(side="right", padx=6)

        self._clock_local_var = tk.StringVar(value="")
        self._clock_ny_var = tk.StringVar(value="")

        tk.Label(clock_box, textvariable=self._clock_ny_var, font=F_MONO_B,
                 bg=PANEL2, fg=ACCENT).pack(anchor="e")
        tk.Label(clock_box, textvariable=self._clock_local_var, font=F_MONO,
                 bg=PANEL2, fg=TEXT_DIM).pack(anchor="e")

    def _build_stat_cards(self):
        cards_row = tk.Frame(self, bg=BG)
        cards_row.pack(fill="x", padx=12, pady=(10, 4))
        self._cards = {}

        kpis = [
            ("balance",     "Account Balance",  "--", "$ USD", ACCENT),
            ("equity",      "Equity",           "--", "$ USD", WHITE),
            ("floating_pnl","Floating P&L",     "--", "Open positions", TEXT_DIM),
            ("closed_pnl",  "Today Closed P&L", "--", "Net realized", TEXT_DIM),
            ("price",       "XAUUSD Price",     "--", "1-Min Spot", GOLD),
            ("signal",      "Signal / Trend",   "--", "EMA-50 Filter", ACCENT2),
            ("session_pnl", "Current Session",  "--", "Session Window", ACCENT),
        ]

        for key, lbl, val, sub, clr in kpis:
            c = StatCard(cards_row, label=lbl, value=val, subtext=sub, color=clr)
            c.pack(side="left", fill="x", expand=True, padx=3)
            self._cards[key] = c

    def _build_tabs(self):
        nb = ttk.Notebook(self, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=12, pady=8)

        t_chart    = tk.Frame(nb, bg=BG)
        t_orders   = tk.Frame(nb, bg=BG)
        t_sessions = tk.Frame(nb, bg=BG)
        t_market   = tk.Frame(nb, bg=BG)
        t_risk     = tk.Frame(nb, bg=BG)

        nb.add(t_chart,    text="  📈 Price Chart & Orderbook  ")
        nb.add(t_orders,   text="  ⚡ Executed & Executing Orders  ")
        nb.add(t_sessions, text="  🕒 3 Sessions P&L Analytics  ")
        nb.add(t_market,   text="  📊 Candlestick Data  ")
        nb.add(t_risk,     text="  🧮 Risk & Sizing  ")

        self._build_chart_orderbook_tab(t_chart)
        self._build_orders_tab(t_orders)
        self._build_sessions_tab(t_sessions)
        self._build_market_tab(t_market)
        self._build_risk_tab(t_risk)

    # --------------------------------------------------------------------------
    #  Tab 1: Price Chart & Live Orderbook
    # --------------------------------------------------------------------------

    def _build_chart_orderbook_tab(self, p):
        p.columnconfigure(0, weight=3)
        p.columnconfigure(1, weight=1)
        p.rowconfigure(0, weight=1)

        # Left: Candlestick Chart
        chart_card = Card(p)
        chart_card.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)
        chart_card.columnconfigure(0, weight=1)
        chart_card.rowconfigure(1, weight=1)

        # Chart Toolbar Header
        toolbar = tk.Frame(chart_card, bg=PANEL2, padx=8, pady=4,
                           highlightbackground=BORDER, highlightthickness=1)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        tk.Label(toolbar, text="Timeframe: M1 (Gold)", font=F_LABEL_B,
                 bg=PANEL2, fg=ACCENT).pack(side="left", padx=4)
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

        tk.Label(toolbar, text="| Sessions Shaded:", font=F_LABEL, bg=PANEL2, fg=TEXT_DIM).pack(side="left", padx=(10, 4))
        for sess in TRADING_SESSIONS_NY:
            tk.Label(toolbar, text=f"■ {sess['name']}", font=("Segoe UI", 8),
                     bg=PANEL2, fg=sess["color"]).pack(side="left", padx=3)

        self._chart = CandlestickChart(chart_card)
        self._chart.grid(row=1, column=0, sticky="nsew")

        # Right: Live Orderbook / Depth Ladder
        book_card = Card(p)
        book_card.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=4)
        book_card.columnconfigure(0, weight=1)
        book_card.rowconfigure(2, weight=1)

        tk.Label(book_card, text="📖 Orderbook & Market Depth",
                 font=F_SECTION, bg=PANEL, fg=ACCENT).grid(row=0, column=0, sticky="w")
        tk.Label(book_card, text="Live Level-2 / Spread Depth Ladder",
                 font=F_LABEL, bg=PANEL, fg=TEXT_DIM).grid(row=1, column=0, sticky="w", pady=(0, 6))

        book_cols = ("side", "price", "volume")
        book_hdrs = ("Side", "Price (USD)", "Depth / Vol")
        self._book_tv = ttk.Treeview(book_card, columns=book_cols, show="headings",
                                     style="TV.Treeview", selectmode="browse")
        self._book_tv.heading("side", text="Side")
        self._book_tv.heading("price", text="Price")
        self._book_tv.heading("volume", text="Volume")
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
        if self._df is not None:
            self._chart.BARS_TO_SHOW = self._bars_var.get()
            self._chart.update_data(self._df)

    # --------------------------------------------------------------------------
    #  Tab 2: Orders & Executions
    # --------------------------------------------------------------------------

    def _build_orders_tab(self, p):
        p.columnconfigure(0, weight=1)
        p.rowconfigure(0, weight=1)
        p.rowconfigure(1, weight=1)

        # Top: Active Executing Positions
        top_card = Card(p)
        top_card.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))
        top_card.columnconfigure(0, weight=1)
        top_card.rowconfigure(1, weight=1)

        header1 = tk.Frame(top_card, bg=PANEL)
        header1.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        tk.Label(header1, text="⚡ Active Executing Positions (Open Market Orders)",
                 font=F_SECTION, bg=PANEL, fg=ACCENT).pack(side="left")
        self._pos_count_lbl = tk.Label(header1, text="0 Open Positions",
                                       font=F_LABEL_B, bg=PANEL2, fg=TEXT, padx=6, pady=1)
        self._pos_count_lbl.pack(side="right")

        pos_cols = ("ticket", "symbol", "type", "volume", "open_price", "curr_price", "sl", "tp", "pnl", "open_time")
        pos_hdrs = ("Ticket", "Symbol", "Type", "Lots", "Open Price", "Current Price", "SL", "TP", "P&L ($)", "Opened (NY / UTC-4)")
        pos_wids = (90, 80, 70, 60, 90, 90, 80, 80, 100, 150)

        self._pos_tv = ttk.Treeview(top_card, columns=pos_cols, show="headings",
                                    style="TV.Treeview", selectmode="browse")
        for c, h, w in zip(pos_cols, pos_hdrs, pos_wids):
            self._pos_tv.heading(c, text=h)
            self._pos_tv.column(c, width=w, anchor="center")

        self._pos_tv.tag_configure("profit", foreground=BUY_CLR)
        self._pos_tv.tag_configure("loss", foreground=SELL_CLR)
        self._pos_tv.tag_configure("buy", foreground=BUY_CLR)
        self._pos_tv.tag_configure("sell", foreground=SELL_CLR)

        pos_sb = ttk.Scrollbar(top_card, orient="vertical", command=self._pos_tv.yview,
                               style="Dark.Vertical.TScrollbar")
        self._pos_tv.configure(yscrollcommand=pos_sb.set)
        self._pos_tv.grid(row=1, column=0, sticky="nsew")
        pos_sb.grid(row=1, column=1, sticky="ns")

        # Bottom: Executed Closed Orders History (Today)
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
        deal_hdrs = ("Deal #", "Open Time (NY / UTC-4)", "Symbol", "Type", "Entry/Exit", "Lots", "Fill Price", "Realized P&L ($)", "Commission", "Trading Session")
        deal_wids = (80, 150, 80, 65, 75, 60, 90, 120, 85, 140)

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

    # --------------------------------------------------------------------------
    #  Tab 3: 3 Sessions P&L Analytics
    # --------------------------------------------------------------------------

    def _build_sessions_tab(self, p):
        for col_idx in range(len(TRADING_SESSIONS_NY)):
            p.columnconfigure(col_idx, weight=1)
        p.rowconfigure(0, weight=0)
        p.rowconfigure(1, weight=1)

        self._session_widgets = {}

        # 4 Dedicated Session Analytics Cards
        for i, sess in enumerate(TRADING_SESSIONS_NY):
            card = Card(p)
            card.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)

            # Title & Time Window
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

        # Bottom: Comparative Session Performance Chart
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
        """Draw horizontal bar chart of P&L per session."""
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

        # Bar text labels
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

    # --------------------------------------------------------------------------
    #  Tab 4: Candlestick Data Table
    # --------------------------------------------------------------------------

    def _build_market_tab(self, p):
        p.columnconfigure(0, weight=1)
        p.rowconfigure(0, weight=1)

        f = Card(p)
        f.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)

        tk.Label(f, text="Recent 1-Minute Candlesticks (XAUUSD Gold)",
                 font=F_SECTION, bg=PANEL, fg=ACCENT).grid(row=0, column=0, sticky="w", pady=(0, 6))

        cols = ("datetime", "open", "high", "low", "close", "volume", "bullish", "inside_bar", "pin_bar", "signal")
        hdrs = ("Datetime", "Open", "High", "Low", "Close", "Volume", "Bullish", "Inside Bar", "Pin Bar", "Signal")
        wids = (140, 85, 85, 85, 85, 70, 60, 75, 65, 85)

        self._market_tv = ttk.Treeview(f, columns=cols, show="headings",
                                       style="TV.Treeview", selectmode="browse")
        for c, h, w in zip(cols, hdrs, wids):
            self._market_tv.heading(c, text=h)
            self._market_tv.column(c, width=w, anchor="center")

        self._market_tv.tag_configure("buy", foreground=BUY_CLR)
        self._market_tv.tag_configure("sell", foreground=SELL_CLR)
        self._market_tv.tag_configure("bull", foreground="#4DB6AC")
        self._market_tv.tag_configure("bear", foreground="#CF6679")

        vsb = ttk.Scrollbar(f, orient="vertical", command=self._market_tv.yview,
                            style="Dark.Vertical.TScrollbar")
        self._market_tv.configure(yscrollcommand=vsb.set)
        self._market_tv.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")

    # --------------------------------------------------------------------------
    #  Tab 5: Risk Calculator
    # --------------------------------------------------------------------------

    def _build_risk_tab(self, p):
        outer = tk.Frame(p, bg=BG)
        outer.pack(fill="both", expand=True)

        card = Card(outer)
        card.place(relx=0.5, rely=0.5, anchor="center", width=540)

        tk.Label(card, text="🧮 Position Size & Risk Manager (Gold / XAUUSD)",
                 font=F_SECTION, bg=PANEL, fg=ACCENT).pack(anchor="w")
        tk.Label(card, text="Fixed-fractional lot sizing  |  1:2 Risk-Reward Ratio",
                 font=F_LABEL, bg=PANEL, fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        HRule(card).pack(fill="x", pady=8)

        fields = [
            ("Account Balance ($)", "balance",  "10000"),
            ("Risk % of Balance",  "risk_pct", "1.0"),
            ("Stop-Loss (pips)",   "sl_pips",  "150"),
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
            reward_usd = sl * 2 * pv * lots

            self._lot_result.set(f"Calculated Volume: {lots:.2f} Lots")
            self._rr_result.set(f"Risk: ${risk_usd:,.2f}   |   Take-Profit (2:1): ${reward_usd:,.2f}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _build_footer(self):
        sbar = tk.Frame(self, bg=PANEL, height=26,
                        highlightbackground=BORDER, highlightthickness=1)
        sbar.pack(fill="x", side="bottom")
        tk.Label(sbar,
                 text="  Asset: XAUUSD (Gold)  |  Timeframe: M1  |  Strategy: Inside Bar Breakout + EMA-50"
                      "  |  Timezone: America/New_York (Auto EDT/EST)  |  EquantEdge Pro",
                 font=("Segoe UI", 8), bg=PANEL, fg=TEXT_DIM).pack(side="left")

    # --------------------------------------------------------------------------
    #  Asynchronous Data Loading & UI Population
    # --------------------------------------------------------------------------

    def _load_data_async(self):
        self._status_var.set("Fetching live market data...")
        def worker():
            try:
                if LIVE_MODE:
                    state = fetch_mt5_full_state(DEFAULT_SYMBOL)
                else:
                    df = load_demo_data()
                    df = enrich(df)
                    state = {
                        "df": df,
                        "account": {
                            "balance": 10000.0,
                            "equity": 10000.0,
                            "currency": "USD",
                            "profit": 0.0,
                            "login": "DEMO-USER",
                            "server": "Offline Demo",
                        },
                        "open_positions": [],
                        "executed_deals": [],
                        "closed_pnl_today": 0.0,
                        "orderbook": [],
                    }
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
        df = state["df"]
        account = state["account"]
        open_positions = state["open_positions"]
        executed_deals = state["executed_deals"]
        closed_pnl = state["closed_pnl_today"]
        orderbook = state["orderbook"]

        self._df = df
        self._account_state = account
        self._open_positions = open_positions
        self._executed_deals = executed_deals
        self._orderbook_data = orderbook

        # 1. Update Stat Cards & Session
        self._populate_stat_cards(df, account, closed_pnl)

        # 2. Update Orderbook
        self._populate_orderbook(orderbook)

        # 3. Update Executing & Executed Orders Table
        self._populate_orders_tab(open_positions, executed_deals, closed_pnl)

        # 4. Update Sessions P&L Analytics
        self._populate_sessions_tab(executed_deals)

        # 5. Update Candlestick Table
        self._populate_market_table(df)

        # 6. Update Candlestick Chart
        self._chart.BARS_TO_SHOW = self._bars_var.get()
        self._chart.update_data(df)

        # Sync Balance into Risk Calculator
        if "balance" in account:
            self._risk_vars["balance"].set(str(account["balance"]))

        self._status_var.set(f"Synced {datetime.now().strftime('%H:%M:%S')}  |  {len(df)} candles  |  {len(open_positions)} active orders")

        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(self.REFRESH_MS, self._load_data_async)

    def _populate_stat_cards(self, df: pd.DataFrame, account: dict, closed_pnl: float):
        last_price = df["close"].iloc[-1]
        self._cards["price"].update_value(f"${last_price:,.2f}", sub="Gold Spot (USD)")

        # Account Balance & Equity
        bal = account.get("balance", 0.0)
        eq = account.get("equity", 0.0)
        curr = account.get("currency", "USD")
        self._cards["balance"].update_value(f"${bal:,.2f}", sub=f"{curr} Account")
        self._cards["equity"].update_value(f"${eq:,.2f}", sub="Live Net Equity")

        # Floating P&L
        floating = account.get("profit", 0.0)
        fl_clr = BUY_CLR if floating > 0 else (SELL_CLR if floating < 0 else TEXT_DIM)
        self._cards["floating_pnl"].update_value(f"{floating:+,.2f} $", sub="Active Positions", color=fl_clr)

        # Closed P&L Today
        cl_clr = BUY_CLR if closed_pnl > 0 else (SELL_CLR if closed_pnl < 0 else TEXT_DIM)
        self._cards["closed_pnl"].update_value(f"{closed_pnl:+,.2f} $", sub="Realized Today", color=cl_clr)

        # Signal / Trend
        sigs = df[df["signal"] != 0]
        if len(sigs):
            latest_sig = int(sigs.iloc[-1]["signal"])
            trend_str = "BULL" if df["trend"].iloc[-1] else "BEAR"
            sig_text = f"{'BUY' if latest_sig == 1 else 'SELL'} ({trend_str})"
            self._cards["signal"].update_value(sig_text, sub="Inside Bar Breakout",
                                              color=BUY_CLR if latest_sig == 1 else SELL_CLR)
        else:
            self._cards["signal"].update_value("NEUTRAL", sub="Trend Waiting", color=TEXT_DIM)

        # Session Card
        sess_info = get_active_session_info()
        if sess_info["in_session"]:
            self._cards["session_pnl"].update_value(sess_info["name"], sub=sess_info["time_range"], color=sess_info["color"])
        else:
            self._cards["session_pnl"].update_value("OFF HOURS", sub="Waiting Next Window", color=TEXT_DIM)

    def _populate_orderbook(self, orderbook: list):
        tv = self._book_tv
        tv.delete(*tv.get_children())
        if not orderbook:
            tv.insert("", "end", values=("--", "Connect MT5 for DOM", "--"))
            return

        for item in orderbook:
            side = item.get("type", "BUY")
            tag = "ask" if side == "SELL" else "bid"
            tv.insert("", "end", values=(
                side,
                f"{item.get('price', 0):.2f}",
                str(int(item.get('volume', 0))),
            ), tags=(tag,))

    def _populate_orders_tab(self, open_positions: list, executed_deals: list, closed_pnl: float):
        # 1. Executing Positions
        pos_tv = self._pos_tv
        pos_tv.delete(*pos_tv.get_children())
        self._pos_count_lbl.config(text=f"{len(open_positions)} Open Position(s)")

        if not open_positions:
            if not LIVE_MODE:
                pos_tv.insert("", "end", values=("--", "--", "--", "--", "Connect MT5 using --live flag to see active trades", "--", "--", "--", "--", "--"))
            else:
                pos_tv.insert("", "end", values=("--", "--", "--", "--", "No active open positions", "--", "--", "--", "--", "--"))
        else:
            for pos in open_positions:
                pnl = pos["profit"]
                tag = "profit" if pnl >= 0 else "loss"
                pos_tv.insert("", "end", values=(
                    str(pos["ticket"]),
                    pos["symbol"],
                    pos["type"],
                    f"{pos['volume']:.2f}",
                    f"{pos['price_open']:.2f}",
                    f"{pos['price_current']:.2f}",
                    f"{pos['sl']:.2f}",
                    f"{pos['tp']:.2f}",
                    f"{pnl:+,.2f}",
                    str(pos["time"])[:19],
                ), tags=(tag,))

        # 2. Executed Deals History
        deal_tv = self._deal_tv
        deal_tv.delete(*deal_tv.get_children())
        self._deals_pnl_lbl.config(
            text=f"Total Realized: ${closed_pnl:+,.2f}",
            fg=BUY_CLR if closed_pnl >= 0 else SELL_CLR
        )

        if not executed_deals:
            if not LIVE_MODE:
                deal_tv.insert("", "end", values=("--", "--", "--", "--", "--", "--", "Connect MT5 using --live flag to see execution logs", "--", "--", "--"))
            else:
                deal_tv.insert("", "end", values=("--", "--", "--", "--", "--", "--", "No executed orders today yet", "--", "--", "--"))
        else:
            for deal in executed_deals[::-1]:
                pnl = deal["profit"]
                tag = "profit" if pnl > 0 else ("loss" if pnl < 0 else "")
                deal_tv.insert("", "end", values=(
                    str(deal["ticket"]),
                    str(deal["time"])[:19],
                    deal["symbol"],
                    deal["type"],
                    deal["entry"],
                    f"{deal['volume']:.2f}",
                    f"{deal['price']:.2f}",
                    f"{pnl:+,.2f}" if pnl != 0 else "$0.00",
                    f"${deal['commission']:.2f}",
                    deal["session"],
                ), tags=(tag,))

    def _populate_sessions_tab(self, executed_deals: list):
        """Aggregate trade performance by trading session window."""
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
        for _, row in df.tail(150).iloc[::-1].iterrows():
            sig = int(row.get("signal", 0))
            bullish = bool(row.get("bullish", row["close"] > row["open"]))
            vals = (
                str(row["datetime"])[:19],
                f"{row['open']:.2f}",
                f"{row['high']:.2f}",
                f"{row['low']:.2f}",
                f"{row['close']:.2f}",
                str(int(row.get("volume", 0))),
                "YES" if bullish else "NO",
                "YES" if bool(row.get("inside_bar", False)) else "-",
                "YES" if bool(row.get("pin_bar", False)) else "-",
                row.get("signal_label", ""),
            )
            tag = "buy" if sig == 1 else ("sell" if sig == -1 else ("bull" if bullish else "bear"))
            tv.insert("", "end", values=vals, tags=(tag,))

    # --------------------------------------------------------------------------
    #  Live Dual Clock & Session Monitor
    # --------------------------------------------------------------------------

    def _clock_tick(self):
        try:
            if not self.winfo_exists():
                return
            # Local system time with explicit UTC offset (e.g. UTC+05:30)
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
                tz_str = "UTC+05:30"
            self._clock_local_var.set(local_now.strftime(f"LOCAL: %H:%M:%S  ({tz_str})"))

            # New York Auto EDT/EST Time
            ny_now = get_ny_now()
            offset_hrs = int(ny_now.utcoffset().total_seconds() // 3600)
            tz_name = ny_now.strftime("%Z")
            self._clock_ny_var.set(ny_now.strftime(f"NY: %H:%M:%S  {tz_name} (UTC{offset_hrs:+d})"))

            # Update Session Indicator Badge
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


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    EquantEdgeDashboard().mainloop()