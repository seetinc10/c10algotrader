"""
Supertrend + 200 EMA Algorithmic Trading Bot
Connects to IBKR TWS paper trading via ib_insync.
Uses 1-minute bars with live updates.
"""

# CRITICAL: Create event loop before importing ib_insync.
# Python 3.12+ removed implicit loop creation, and eventkit
# (ib_insync dependency) requires a loop at import time.
import asyncio
import asyncio.timeouts as _timeouts

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

# ---------------------------------------------------------------------------
# Python 3.14 compatibility patch
# ---------------------------------------------------------------------------
# Python 3.14 changed asyncio.Timeout.__aenter__ to require a "current task"
# (via tasks.current_task()). Under nest_asyncio the reentrant
# run_until_complete does not set a current task, so every call to
# asyncio.wait_for — even with timeout=None — crashes with:
#   RuntimeError("Timeout should be used inside a task")
#
# Patch: replace the Timeout.__aenter__ with a version that tolerates a
# missing current task (sets self._task = None and skips rescheduling when
# there is no deadline).
# ---------------------------------------------------------------------------

_orig_timeout_enter = _timeouts.Timeout.__aenter__


async def _patched_timeout_enter(self):
    from asyncio import tasks as _tasks
    if self._state is not _timeouts._State.CREATED:
        raise RuntimeError("Timeout has already been entered")
    task = _tasks.current_task()
    if task is None:
        # No current task (nest_asyncio reentrant loop).
        # Allow entry but disable cancellation-based timeout.
        self._state = _timeouts._State.ENTERED
        self._task = None
        self._cancelling = 0
        return self
    return await _orig_timeout_enter(self)


_timeouts.Timeout.__aenter__ = _patched_timeout_enter

import tkinter as tk
from tkinter import messagebox
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import nest_asyncio

nest_asyncio.apply(_loop)

from ib_insync import IB, Stock, MarketOrder
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


TRADE_HISTORY_FILE = Path(
    os.environ.get("C10_TRADE_HISTORY_FILE", Path(__file__).parent / "trade_history.json")
).expanduser()
MAX_TRADE_HISTORY_ENTRIES = 5000

# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class BaseStrategy:
    """Abstract base for all trading strategies.

    Every subclass MUST:
    - Set ``name`` (display name for GUI dropdown).
    - Set ``min_bars`` (minimum bars needed before signals are valid).
    - Implement ``compute(bars) -> dict`` returning at least:
        'direction' : np.ndarray int (+1 bullish, -1 bearish, 0 neutral)
        'atr'       : np.ndarray float (for trailing stop; NaN-padded)
      Optionally return 'trade_direction' — a filtered version of direction
      that the engine/backtester will prefer for entries.
    - Implement ``plot_indicators(ax_price, ax_sub, indicators, x, start, colors)``
    """

    name: str = "Base"
    description: str = "No description available."
    min_bars: int = 50
    needs_subpanel: bool = False
    bar_size: str = '1 min'       # IB bar size for live + backtest
    duration: str = '2 D'         # IB duration for initial data load

    # -- shared indicator helpers --------------------------------------------

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        n = len(data)
        ema = np.full(n, np.nan)
        if n < period:
            return ema
        ema[period - 1] = np.mean(data[:period])
        alpha = 2.0 / (period + 1)
        for i in range(period, n):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        n = len(data)
        sma = np.full(n, np.nan)
        if n < period:
            return sma
        cumsum = np.cumsum(data)
        sma[period - 1:] = (cumsum[period - 1:] - np.concatenate(
            ([0.0], cumsum[:n - period]))) / period
        return sma

    def _atr(self, high: np.ndarray, low: np.ndarray,
             close: np.ndarray, period: int = 14) -> np.ndarray:
        n = len(close)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
        atr = np.full(n, np.nan)
        if n < period:
            return atr
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    def _rolling_std(self, data: np.ndarray, period: int) -> np.ndarray:
        n = len(data)
        std = np.full(n, np.nan)
        if n < period:
            return std
        for i in range(period - 1, n):
            std[i] = np.std(data[i - period + 1:i + 1], ddof=0)
        return std

    def _empty(self, n: int) -> dict:
        return {
            'direction': np.zeros(n, dtype=int),
            'atr': np.full(n, np.nan),
        }

    def compute(self, bars) -> dict:
        raise NotImplementedError

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        """Override to draw strategy-specific overlays on the chart."""
        pass


# ---------------------------------------------------------------------------
# Strategy: Supertrend + 200 EMA  (original)
# ---------------------------------------------------------------------------

class SupertrendStrategy(BaseStrategy):
    """Computes Supertrend indicator and 200 EMA from OHLC bar data."""

    name = "Supertrend + 200 EMA"
    description = (
        "TREND-FOLLOWING strategy combining two filters.\n\n"
        "HOW IT WORKS\n"
        "The Supertrend indicator builds dynamic support/resistance bands "
        "around price using ATR. When price closes above the upper band the "
        "trend flips bullish (green line below price); when it closes below "
        "the lower band it flips bearish (red line above price). A 200-bar "
        "EMA acts as a trend filter: long signals are only valid when price "
        "is above the 200 EMA, short signals only when below.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Supertrend flips bullish AND price > 200 EMA\n"
        "  SHORT : Supertrend flips bearish AND price < 200 EMA\n\n"
        "EXIT RULES\n"
        "  - Supertrend flips against your position\n"
        "  - ATR trailing stop hit (configurable multiplier)\n\n"
        "BEST CONDITIONS\n"
        "  - Strong trending markets (tech, momentum stocks)\n"
        "  - Avoid during choppy, range-bound sessions\n"
        "  - Works on any timeframe; default uses 1-min bars\n\n"
        "PARAMETERS\n"
        "  ATR Period     : 7   (Supertrend sensitivity)\n"
        "  ATR Multiplier : 3.0 (band width)\n"
        "  EMA Period     : 200 (trend filter)\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  2.5x  (trend strategy — lets winners run with a wide stop)\n\n"
        "CHART INDICATORS\n"
        "  - Yellow line  : 200 EMA\n"
        "  - Green/Red line : Supertrend (below price = bullish, above = bearish)"
    )

    def __init__(self, atr_period: int = 7, atr_multiplier: float = 3.0,
                 ema_period: int = 200):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.ema_period = ema_period
        self.min_bars = ema_period + atr_period + 5

    def compute(self, bars) -> dict:
        """Return dict with ema200, supertrend, direction, trade_direction, atr."""
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)

        # --- True Range / ATR ---
        atr = self._atr(high, low, close, self.atr_period)
        p = self.atr_period
        if n < p:
            return self._st_empty(n)

        # --- Basic upper / lower bands ---
        hl2 = (high + low) / 2.0
        basic_upper = hl2 + self.atr_multiplier * atr
        basic_lower = hl2 - self.atr_multiplier * atr

        # --- Final bands (with trailing logic) ---
        final_upper = np.full(n, np.nan)
        final_lower = np.full(n, np.nan)
        final_upper[p - 1] = basic_upper[p - 1]
        final_lower[p - 1] = basic_lower[p - 1]

        for i in range(p, n):
            if (basic_upper[i] < final_upper[i - 1] or
                    close[i - 1] > final_upper[i - 1]):
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i - 1]
            if (basic_lower[i] > final_lower[i - 1] or
                    close[i - 1] < final_lower[i - 1]):
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i - 1]

        # --- Direction and Supertrend line ---
        direction = np.zeros(n, dtype=int)
        supertrend = np.full(n, np.nan)
        direction[p - 1] = 1
        supertrend[p - 1] = final_lower[p - 1]

        for i in range(p, n):
            if supertrend[i - 1] == final_upper[i - 1]:
                if close[i] > final_upper[i]:
                    direction[i] = 1
                else:
                    direction[i] = -1
            else:
                if close[i] < final_lower[i]:
                    direction[i] = -1
                else:
                    direction[i] = 1
            supertrend[i] = (final_lower[i] if direction[i] == 1
                             else final_upper[i])

        # --- 200 EMA ---
        ema = self._ema(close, self.ema_period)

        # --- trade_direction: direction filtered by EMA ---
        trade_dir = direction.copy()
        for i in range(n):
            if np.isnan(ema[i]):
                trade_dir[i] = 0
            elif direction[i] > 0 and close[i] <= ema[i]:
                trade_dir[i] = 0
            elif direction[i] < 0 and close[i] >= ema[i]:
                trade_dir[i] = 0

        return {
            'ema200': ema,
            'supertrend': supertrend,
            'direction': direction,
            'trade_direction': trade_dir,
            'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        ema = indicators['ema200'][start:]
        st = indicators['supertrend'][start:]
        dirs = indicators['direction'][start:]

        mask_ema = ~np.isnan(ema)
        if mask_ema.any():
            ax_price.plot(x[mask_ema], ema[mask_ema],
                          color=colors['yellow'], linewidth=1.0,
                          label='200 EMA', zorder=2)

        mask_st = ~np.isnan(st)
        for i in range(1, len(st)):
            if not (mask_st[i] and mask_st[i - 1]):
                continue
            color = colors['green'] if dirs[i] == 1 else colors['red']
            ax_price.plot([x[i - 1], x[i]], [st[i - 1], st[i]],
                          color=color, linewidth=2, zorder=4)

    def _st_empty(self, n: int) -> dict:
        nan = np.full(n, np.nan)
        return {
            'ema200': nan.copy(),
            'supertrend': nan.copy(),
            'direction': np.zeros(n, dtype=int),
            'trade_direction': np.zeros(n, dtype=int),
            'atr': nan.copy(),
        }


# ---------------------------------------------------------------------------
# Strategy: Dual Moving Average (50/200 Golden / Death Cross)
# ---------------------------------------------------------------------------

class DualMovingAverageStrategy(BaseStrategy):
    name = "Dual MA (50/200)"
    description = (
        "TREND-FOLLOWING strategy using two exponential moving averages.\n\n"
        "HOW IT WORKS\n"
        "Tracks a fast EMA (50) and slow EMA (200). When the fast EMA "
        "crosses above the slow EMA it signals a 'Golden Cross' (bullish). "
        "When it crosses below it signals a 'Death Cross' (bearish). This "
        "is one of the oldest and most widely used trend-following systems.\n\n"
        "ENTRY RULES\n"
        "  LONG  : 50 EMA crosses above 200 EMA\n"
        "  SHORT : 50 EMA crosses below 200 EMA\n\n"
        "EXIT RULES\n"
        "  - Opposite crossover occurs\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Works best in markets with strong, sustained trends\n"
        "  - Produces fewer but higher-conviction signals\n"
        "  - Slower to react; avoids whipsaws in choppy markets\n"
        "  - Uses 5-min bars for smoother crossovers\n\n"
        "PARAMETERS\n"
        "  Fast EMA  : 50\n"
        "  Slow EMA  : 200\n"
        "  ATR Period : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  3.0x  (slow trend strategy — wide stop avoids premature exits)\n\n"
        "CHART INDICATORS\n"
        "  - Blue line   : 50 EMA (fast)\n"
        "  - Yellow line  : 200 EMA (slow)"
    )
    bar_size = '5 mins'
    duration = '5 D'

    def __init__(self, fast_period=50, slow_period=200, atr_period=14):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.min_bars = slow_period + 5

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.slow_period:
            return self._empty(n)

        fast_ma = self._ema(close, self.fast_period)
        slow_ma = self._ema(close, self.slow_period)
        atr = self._atr(high, low, close, self.atr_period)

        direction = np.zeros(n, dtype=int)
        for i in range(n):
            if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
                continue
            direction[i] = 1 if fast_ma[i] > slow_ma[i] else -1

        return {
            'fast_ma': fast_ma, 'slow_ma': slow_ma,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        fast = indicators['fast_ma'][start:]
        slow = indicators['slow_ma'][start:]
        m1 = ~np.isnan(fast)
        m2 = ~np.isnan(slow)
        if m1.any():
            ax_price.plot(x[m1], fast[m1], color='#4fc3f7', linewidth=1.2,
                          label=f'{self.fast_period} EMA', zorder=2)
        if m2.any():
            ax_price.plot(x[m2], slow[m2], color=colors['yellow'], linewidth=1.2,
                          label=f'{self.slow_period} EMA', zorder=2)


# ---------------------------------------------------------------------------
# Strategy: Donchian Channel Breakout (Turtle Traders)
# ---------------------------------------------------------------------------

class DonchianBreakoutStrategy(BaseStrategy):
    name = "Donchian Breakout"
    description = (
        "BREAKOUT strategy based on the Turtle Traders method.\n\n"
        "HOW IT WORKS\n"
        "Draws a channel from the highest high and lowest low of the last "
        "20 bars. When price breaks above the upper channel a bullish "
        "breakout is signalled; when it breaks below the lower channel a "
        "bearish breakout is signalled. The shaded area between channels "
        "is a neutral 'no-trade' zone.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Close breaks above the 20-bar high\n"
        "  SHORT : Close breaks below the 20-bar low\n\n"
        "EXIT RULES\n"
        "  - Price breaks through the opposite channel\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Stocks making new highs or lows (momentum breakouts)\n"
        "  - Earnings gaps, news-driven moves\n"
        "  - Avoid in tight consolidation ranges (many false breakouts)\n"
        "  - Uses 5-min bars for fewer false signals\n\n"
        "PARAMETERS\n"
        "  Channel Period : 20 (lookback for high/low)\n"
        "  ATR Period     : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  2.5x  (breakout strategy — needs room for post-breakout pullbacks)\n\n"
        "CHART INDICATORS\n"
        "  - Green dashed : Upper channel (20-bar high)\n"
        "  - Red dashed   : Lower channel (20-bar low)\n"
        "  - Blue shading : Channel range"
    )
    bar_size = '5 mins'
    duration = '5 D'

    def __init__(self, channel_period=20, atr_period=14):
        self.channel_period = channel_period
        self.atr_period = atr_period
        self.min_bars = channel_period + 5

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.channel_period + 1:
            return self._empty(n)

        p = self.channel_period
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        for i in range(p, n):
            upper[i] = np.max(high[i - p:i])
            lower[i] = np.min(low[i - p:i])

        atr = self._atr(high, low, close, self.atr_period)

        direction = np.zeros(n, dtype=int)
        for i in range(p + 1, n):
            if np.isnan(upper[i]) or np.isnan(lower[i]):
                direction[i] = direction[i - 1]
                continue
            if close[i] > upper[i]:
                direction[i] = 1
            elif close[i] < lower[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        return {
            'upper_channel': upper, 'lower_channel': lower,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        upper = indicators['upper_channel'][start:]
        lower = indicators['lower_channel'][start:]
        mu = ~np.isnan(upper)
        ml = ~np.isnan(lower)
        if mu.any():
            ax_price.plot(x[mu], upper[mu], color=colors['green'],
                          linewidth=1.0, linestyle='--', label='Upper', zorder=2)
        if ml.any():
            ax_price.plot(x[ml], lower[ml], color=colors['red'],
                          linewidth=1.0, linestyle='--', label='Lower', zorder=2)
        both = mu & ml
        if both.any():
            ax_price.fill_between(x[both], upper[both], lower[both],
                                  color='#4fc3f7', alpha=0.06, zorder=1)


# ---------------------------------------------------------------------------
# Strategy: Bollinger Band Bounce (Mean Reversion)
# ---------------------------------------------------------------------------

class BollingerBounceStrategy(BaseStrategy):
    name = "Bollinger Bounce"
    description = (
        "MEAN-REVERSION strategy using Bollinger Bands.\n\n"
        "HOW IT WORKS\n"
        "Bollinger Bands plot a 20-bar SMA with upper and lower bands at "
        "+/- 2 standard deviations. When price touches the lower band the "
        "stock is considered oversold and a long signal fires. When price "
        "touches the upper band the stock is overbought and a short signal "
        "fires. The idea is that price tends to revert back to the mean.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Close touches or drops below the lower band\n"
        "  SHORT : Close touches or rises above the upper band\n\n"
        "EXIT RULES\n"
        "  - Price reaches the opposite band\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Range-bound, sideways markets\n"
        "  - Stocks oscillating around a stable mean\n"
        "  - AVOID in strong trending markets (bands ride the trend)\n"
        "  - Uses 5-min bars for cleaner signals\n\n"
        "PARAMETERS\n"
        "  BB Period : 20 (SMA lookback)\n"
        "  BB StdDev : 2.0 (band width multiplier)\n"
        "  ATR Period : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  1.5x  (mean-reversion — tight stop as a safety net; strategy has\n"
        "         its own band-based exit, or use 0 to rely on bands only)\n\n"
        "CHART INDICATORS\n"
        "  - Blue dashed : Middle band (20 SMA)\n"
        "  - Red line    : Upper band (+2 std dev)\n"
        "  - Green line  : Lower band (-2 std dev)\n"
        "  - Blue shading : Band range"
    )
    bar_size = '5 mins'
    duration = '5 D'

    def __init__(self, bb_period=20, bb_std=2.0, atr_period=14):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.min_bars = bb_period + 5

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.bb_period:
            return self._empty(n)

        mid = self._sma(close, self.bb_period)
        std = self._rolling_std(close, self.bb_period)
        upper = mid + self.bb_std * std
        lower = mid - self.bb_std * std
        atr = self._atr(high, low, close, self.atr_period)

        direction = np.zeros(n, dtype=int)
        for i in range(self.bb_period, n):
            if np.isnan(upper[i]):
                direction[i] = direction[i - 1]
                continue
            if close[i] <= lower[i]:
                direction[i] = 1   # mean-reversion buy
            elif close[i] >= upper[i]:
                direction[i] = -1  # mean-reversion sell
            else:
                direction[i] = direction[i - 1]

        return {
            'bb_upper': upper, 'bb_mid': mid, 'bb_lower': lower,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        upper = indicators['bb_upper'][start:]
        mid = indicators['bb_mid'][start:]
        lower = indicators['bb_lower'][start:]
        m = ~np.isnan(mid)
        if m.any():
            ax_price.plot(x[m], mid[m], color='#4fc3f7', linewidth=0.8,
                          linestyle='--', label='BB Mid', zorder=2)
            ax_price.plot(x[m], upper[m], color=colors['red'], linewidth=0.8,
                          label='BB Upper', zorder=2)
            ax_price.plot(x[m], lower[m], color=colors['green'], linewidth=0.8,
                          label='BB Lower', zorder=2)
            ax_price.fill_between(x[m], upper[m], lower[m],
                                  color='#4fc3f7', alpha=0.06, zorder=1)


# ---------------------------------------------------------------------------
# Strategy: VWAP Reversion
# ---------------------------------------------------------------------------

class VWAPReversionStrategy(BaseStrategy):
    name = "VWAP Reversion"
    description = (
        "MEAN-REVERSION strategy using the Volume Weighted Average Price.\n\n"
        "HOW IT WORKS\n"
        "VWAP is the average price weighted by volume, reset each trading "
        "day at 9:30 ET. Upper and lower bands are drawn at +/- 1.5 "
        "standard deviations from VWAP. When price drops below the lower "
        "band it signals an oversold bounce opportunity; when price rises "
        "above the upper band it signals an overbought reversal.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Close drops below VWAP lower band\n"
        "  SHORT : Close rises above VWAP upper band\n\n"
        "EXIT RULES\n"
        "  - Price reaches the opposite band\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Intraday trading (VWAP resets daily)\n"
        "  - High-volume stocks with institutional activity\n"
        "  - Best during mid-day when VWAP stabilizes\n"
        "  - Avoid first 10-15 minutes (VWAP is unstable)\n\n"
        "PARAMETERS\n"
        "  Std Multiplier : 1.5 (band width around VWAP)\n"
        "  ATR Period     : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  1.5x  (mean-reversion — tight safety stop; strategy exits at\n"
        "         opposite band, or use 0 to rely on bands only)\n\n"
        "CHART INDICATORS\n"
        "  - Blue line   : VWAP\n"
        "  - Red dashed  : Upper band (+1.5 std)\n"
        "  - Green dashed : Lower band (-1.5 std)\n"
        "  - Blue shading : Band range"
    )

    def __init__(self, std_mult=1.5, atr_period=14):
        self.std_mult = std_mult
        self.atr_period = atr_period
        self.min_bars = 30

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        volume = np.array([getattr(b, 'volume', 0) for b in bars], dtype=float)
        n = len(close)
        if n < 2:
            return self._empty(n)

        # Detect day boundaries and compute intraday VWAP
        _ET = ZoneInfo('America/New_York')
        vwap = np.full(n, np.nan)
        vwap_upper = np.full(n, np.nan)
        vwap_lower = np.full(n, np.nan)
        typical = (high + low + close) / 3.0

        cum_tp_vol = 0.0
        cum_vol = 0.0
        cum_tp2_vol = 0.0
        prev_date = None

        for i in range(n):
            try:
                dt = bars[i].date
                if hasattr(dt, 'astimezone'):
                    dt = dt.astimezone(_ET)
                cur_date = dt.date() if hasattr(dt, 'date') else None
            except Exception:
                cur_date = prev_date

            if cur_date != prev_date:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                cum_tp2_vol = 0.0
                prev_date = cur_date

            vol = max(volume[i], 1.0)
            cum_tp_vol += typical[i] * vol
            cum_vol += vol
            cum_tp2_vol += (typical[i] ** 2) * vol

            v = cum_tp_vol / cum_vol
            vwap[i] = v
            variance = max(cum_tp2_vol / cum_vol - v * v, 0.0)
            std = variance ** 0.5
            vwap_upper[i] = v + self.std_mult * std
            vwap_lower[i] = v - self.std_mult * std

        atr = self._atr(high, low, close, self.atr_period)

        direction = np.zeros(n, dtype=int)
        for i in range(1, n):
            if np.isnan(vwap_lower[i]):
                direction[i] = direction[i - 1]
                continue
            if close[i] <= vwap_lower[i]:
                direction[i] = 1
            elif close[i] >= vwap_upper[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        return {
            'vwap': vwap, 'vwap_upper': vwap_upper, 'vwap_lower': vwap_lower,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        vwap = indicators['vwap'][start:]
        upper = indicators['vwap_upper'][start:]
        lower = indicators['vwap_lower'][start:]
        m = ~np.isnan(vwap)
        if m.any():
            ax_price.plot(x[m], vwap[m], color='#4fc3f7', linewidth=1.2,
                          label='VWAP', zorder=2)
            ax_price.plot(x[m], upper[m], color=colors['red'], linewidth=0.8,
                          linestyle='--', label='VWAP Upper', zorder=2)
            ax_price.plot(x[m], lower[m], color=colors['green'], linewidth=0.8,
                          linestyle='--', label='VWAP Lower', zorder=2)
            ax_price.fill_between(x[m], upper[m], lower[m],
                                  color='#4fc3f7', alpha=0.06, zorder=1)


# ---------------------------------------------------------------------------
# Strategy: Opening Range Breakout
# ---------------------------------------------------------------------------

class OpeningRangeBreakoutStrategy(BaseStrategy):
    name = "Opening Range Breakout"
    description = (
        "BREAKOUT strategy based on the first 30 minutes of trading.\n\n"
        "HOW IT WORKS\n"
        "Records the highest high and lowest low during the first 30 "
        "minutes after market open (9:30-10:00 ET). After the opening "
        "range is established, the strategy waits for price to break "
        "above or below it. Only ONE trade per day is allowed -- whichever "
        "direction breaks first.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Close breaks above the opening range high\n"
        "  SHORT : Close breaks below the opening range low\n"
        "  (only the first breakout of the day is taken)\n\n"
        "EXIT RULES\n"
        "  - Profit target hit (PT xR multiplier of range width)\n"
        "  - ATR trailing stop hit\n"
        "  - End of day (manual or engine stop)\n\n"
        "BEST CONDITIONS\n"
        "  - High-volume stocks with strong opens (gap ups/downs)\n"
        "  - Earnings days, news catalysts, sector momentum\n"
        "  - Most effective in the first 2 hours after the range forms\n"
        "  - Avoid on low-volume, narrow-range days\n\n"
        "PARAMETERS\n"
        "  Range Minutes  : 30 (9:30-10:00 ET)\n"
        "  Profit Target  : 1.5x range width (set via PT field in GUI)\n"
        "  ATR Period     : 14\n\n"
        "CHART INDICATORS\n"
        "  - Green dashed : Opening range high\n"
        "  - Red dashed   : Opening range low\n"
        "  - Yellow shading : Opening range zone\n"
        "  - Cyan dotted  : Profit target level\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  2.0x  (breakout — secondary protection after profit target;\n"
        "         catches runners that blow past the PT level)\n\n"
        "RECOMMENDED PROFIT TARGET\n"
        "  1.5x  (1.5 times the opening range width; e.g., if range is\n"
        "         $2.00, target is $3.00 above breakout for longs)\n"
        "  Use 0 to disable profit target and rely on trail only\n\n"
        "TIPS\n"
        "  - A wider opening range = larger potential move but wider stop\n"
        "  - A narrow opening range = tighter stop, faster breakout\n"
        "  - Combine PT 1.5x with Trail 2.0x for best results\n"
        "  - This strategy works best as a one-shot daily trade"
    )

    def __init__(self, range_minutes=30, atr_period=14, profit_mult=1.5):
        self.range_minutes = range_minutes
        self.atr_period = atr_period
        self.profit_mult = profit_mult
        self.min_bars = 60

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < 2:
            return self._empty(n)

        _ET = ZoneInfo('America/New_York')
        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)
        atr = self._atr(high, low, close, self.atr_period)

        # Parse bar timestamps once
        bar_dates = []
        bar_mins = []
        for i in range(n):
            try:
                dt = bars[i].date
                if hasattr(dt, 'astimezone'):
                    dt = dt.astimezone(_ET)
                bar_dates.append(dt.date() if hasattr(dt, 'date') else None)
                bar_mins.append(dt.hour * 60 + dt.minute)
            except Exception:
                bar_dates.append(None)
                bar_mins.append(-1)

        open_min = 9 * 60 + 30
        range_end_min = open_min + self.range_minutes

        # --- Pass 1: build opening range per day ---
        day_data = {}
        for i in range(n):
            cur_date = bar_dates[i]
            mm = bar_mins[i]
            if cur_date is None or mm < 0:
                continue
            if cur_date not in day_data:
                day_data[cur_date] = {
                    'range_high': -float('inf'),
                    'range_low': float('inf'),
                    'range_done': False,
                    'took_trade': False,  # only one trade per day
                }
            dd = day_data[cur_date]
            if open_min <= mm < range_end_min:
                dd['range_high'] = max(dd['range_high'], high[i])
                dd['range_low'] = min(dd['range_low'], low[i])
            elif mm >= range_end_min and not dd['range_done']:
                dd['range_done'] = True

        # --- Pass 2: one entry per direction per day ---
        direction = np.zeros(n, dtype=int)
        profit_target = np.full(n, np.nan)
        for i in range(n):
            cur_date = bar_dates[i]
            if cur_date is None or cur_date not in day_data:
                continue
            dd = day_data[cur_date]
            if not dd['range_done'] or dd['range_high'] <= -float('inf'):
                continue

            rh = dd['range_high']
            rl = dd['range_low']
            orb_high[i] = rh
            orb_low[i] = rl
            range_width = rh - rl

            if close[i] > rh and not dd['took_trade']:
                dd['took_trade'] = True
                direction[i] = 1
                dd['pt'] = rh + self.profit_mult * range_width
            elif close[i] < rl and not dd['took_trade']:
                dd['took_trade'] = True
                direction[i] = -1
                dd['pt'] = rl - self.profit_mult * range_width
            else:
                direction[i] = direction[i - 1] if i > 0 else 0

            # Carry profit target forward while in a trade
            if dd['took_trade'] and 'pt' in dd:
                profit_target[i] = dd['pt']

        return {
            'orb_high': orb_high, 'orb_low': orb_low,
            'direction': direction, 'trade_direction': direction, 'atr': atr,
            'profit_target': profit_target,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        orb_h = indicators['orb_high'][start:]
        orb_l = indicators['orb_low'][start:]
        mh = ~np.isnan(orb_h)
        if mh.any():
            ax_price.plot(x[mh], orb_h[mh], color=colors['green'],
                          linewidth=1.2, linestyle='--', label='ORB High', zorder=2)
            ax_price.plot(x[mh], orb_l[mh], color=colors['red'],
                          linewidth=1.2, linestyle='--', label='ORB Low', zorder=2)
            ax_price.fill_between(x[mh], orb_h[mh], orb_l[mh],
                                  color=colors['yellow'], alpha=0.06, zorder=1)
        # Profit target line
        if 'profit_target' in indicators:
            pt = indicators['profit_target'][start:]
            mpt = ~np.isnan(pt)
            if mpt.any():
                ax_price.plot(x[mpt], pt[mpt], color='#00bcd4',
                              linewidth=1.0, linestyle=':',
                              label='Profit Target', zorder=2)


# ---------------------------------------------------------------------------
# Strategy: ATR Trailing Stop (Chandelier-style)
# ---------------------------------------------------------------------------

class ATRTrailingStopStrategy(BaseStrategy):
    name = "ATR Trailing Stop"
    description = (
        "TREND-FOLLOWING strategy using a Chandelier-style trailing stop.\n\n"
        "HOW IT WORKS\n"
        "Computes a dynamic stop line that trails behind price at a "
        "distance of 3x ATR. In a long trend the stop ratchets up as "
        "price rises but never drops. When price falls below the stop "
        "line the trend flips bearish and the stop moves above price. "
        "A 50 EMA is plotted as a visual trend filter.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Price crosses above the trailing stop line\n"
        "  SHORT : Price crosses below the trailing stop line\n\n"
        "EXIT RULES\n"
        "  - Stop line flips direction (always in a trade)\n"
        "  - Engine ATR trailing stop provides a secondary exit\n\n"
        "BEST CONDITIONS\n"
        "  - Strong trending stocks with big moves\n"
        "  - Lets winners run with a dynamic trailing stop\n"
        "  - Will whipsaw in choppy, range-bound markets\n"
        "  - Good for swing trades and momentum plays\n\n"
        "PARAMETERS\n"
        "  ATR Period    : 14\n"
        "  ATR Multiplier : 3.0 (stop distance)\n"
        "  Entry EMA     : 50 (visual trend filter)\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  0.0  (DISABLED — this strategy IS a trailing stop system;\n"
        "        the engine trail would conflict with its built-in stop)\n\n"
        "CHART INDICATORS\n"
        "  - Green/Red line : Trailing stop line\n"
        "    (green below price = bullish, red above = bearish)\n"
        "  - Yellow line    : 50 EMA trend filter"
    )

    def __init__(self, atr_period=14, atr_mult=3.0, entry_ema=50):
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.entry_ema = entry_ema
        self.min_bars = max(atr_period, entry_ema) + 5

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        atr = self._atr(high, low, close, self.atr_period)
        ema = self._ema(close, self.entry_ema)

        # Chandelier stop line
        stop_line = np.full(n, np.nan)
        direction = np.zeros(n, dtype=int)
        p = self.atr_period

        # Seed
        direction[p - 1] = 1 if close[p - 1] > close[0] else -1
        for i in range(p, n):
            if np.isnan(atr[i]):
                direction[i] = direction[i - 1]
                continue
            dist = self.atr_mult * atr[i]
            if direction[i - 1] >= 0:
                # Long mode: stop below
                new_stop = close[i] - dist
                prev_stop = stop_line[i - 1] if not np.isnan(stop_line[i - 1]) else new_stop
                stop_line[i] = max(new_stop, prev_stop) if close[i] > prev_stop else new_stop
                if close[i] < stop_line[i]:
                    direction[i] = -1
                    stop_line[i] = close[i] + dist
                else:
                    direction[i] = 1
            else:
                # Short mode: stop above
                new_stop = close[i] + dist
                prev_stop = stop_line[i - 1] if not np.isnan(stop_line[i - 1]) else new_stop
                stop_line[i] = min(new_stop, prev_stop) if close[i] < prev_stop else new_stop
                if close[i] > stop_line[i]:
                    direction[i] = 1
                    stop_line[i] = close[i] - dist
                else:
                    direction[i] = -1

        return {
            'atr_stop_line': stop_line, 'ema_filter': ema,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        stop = indicators['atr_stop_line'][start:]
        ema = indicators['ema_filter'][start:]
        dirs = indicators['direction'][start:]
        ms = ~np.isnan(stop)
        me = ~np.isnan(ema)
        if me.any():
            ax_price.plot(x[me], ema[me], color=colors['yellow'],
                          linewidth=1.0, label=f'{self.entry_ema} EMA', zorder=2)
        for i in range(1, len(stop)):
            if not (ms[i] and ms[i - 1]):
                continue
            c = colors['green'] if dirs[i] == 1 else colors['red']
            ax_price.plot([x[i - 1], x[i]], [stop[i - 1], stop[i]],
                          color=c, linewidth=1.5, zorder=4)


# ---------------------------------------------------------------------------
# Strategy: MACD Crossover
# ---------------------------------------------------------------------------

class MACDCrossoverStrategy(BaseStrategy):
    name = "MACD Crossover"
    description = (
        "MOMENTUM strategy using the Moving Average Convergence Divergence.\n\n"
        "HOW IT WORKS\n"
        "MACD is the difference between a 12-bar and 26-bar EMA. A 9-bar "
        "EMA of the MACD line is the 'signal line'. When the MACD line "
        "crosses above the signal line momentum is shifting bullish. When "
        "it crosses below momentum is shifting bearish. The histogram "
        "shows the gap between MACD and signal -- taller bars = stronger "
        "momentum.\n\n"
        "ENTRY RULES\n"
        "  LONG  : MACD line crosses above the signal line\n"
        "  SHORT : MACD line crosses below the signal line\n\n"
        "EXIT RULES\n"
        "  - Opposite MACD crossover\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Trending markets with clear momentum shifts\n"
        "  - Confirming direction after a pullback\n"
        "  - Can lag in fast-moving markets (uses EMAs)\n"
        "  - Works well combined with price action confirmation\n\n"
        "PARAMETERS\n"
        "  Fast EMA    : 12\n"
        "  Slow EMA    : 26\n"
        "  Signal EMA  : 9\n"
        "  ATR Period  : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  2.0x  (momentum — gives room for pullbacks between crossovers)\n\n"
        "CHART INDICATORS (sub-panel)\n"
        "  - Blue line   : MACD line (12 EMA - 26 EMA)\n"
        "  - Orange line : Signal line (9 EMA of MACD)\n"
        "  - Green/Red bars : Histogram (MACD - Signal)"
    )
    needs_subpanel = True

    def __init__(self, fast=12, slow=26, signal=9, atr_period=14):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.atr_period = atr_period
        self.min_bars = slow + signal + 5

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        ema_fast = self._ema(close, self.fast)
        ema_slow = self._ema(close, self.slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line[~np.isnan(macd_line)], self.signal)

        # Re-align signal_line to full array
        sig_full = np.full(n, np.nan)
        valid_idx = np.where(~np.isnan(macd_line))[0]
        if len(signal_line) == len(valid_idx):
            sig_full[valid_idx] = signal_line

        histogram = macd_line - sig_full
        atr = self._atr(high, low, close, self.atr_period)

        direction = np.zeros(n, dtype=int)
        for i in range(n):
            if np.isnan(macd_line[i]) or np.isnan(sig_full[i]):
                continue
            direction[i] = 1 if macd_line[i] > sig_full[i] else -1

        return {
            'macd_line': macd_line, 'signal_line': sig_full,
            'histogram': histogram,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        if ax_sub is None:
            return
        macd = indicators['macd_line'][start:]
        sig = indicators['signal_line'][start:]
        hist = indicators['histogram'][start:]

        m = ~np.isnan(macd)
        ms = ~np.isnan(sig)
        if m.any():
            ax_sub.plot(x[m], macd[m], color='#4fc3f7', linewidth=1.0,
                        label='MACD', zorder=3)
        if ms.any():
            ax_sub.plot(x[ms], sig[ms], color='#ff9800', linewidth=1.0,
                        label='Signal', zorder=3)
        mh = ~np.isnan(hist)
        if mh.any():
            pos = np.where(hist > 0, hist, 0)
            neg = np.where(hist < 0, hist, 0)
            ax_sub.bar(x[mh], pos[mh], color=colors['green'], alpha=0.5, width=0.8)
            ax_sub.bar(x[mh], neg[mh], color=colors['red'], alpha=0.5, width=0.8)
        ax_sub.axhline(y=0, color='#555', linewidth=0.5)
        ax_sub.legend(loc='upper left', fontsize=7,
                      facecolor='#111', edgecolor='#333', labelcolor='#ccc')


# ---------------------------------------------------------------------------
# Strategy: RSI Overbought/Oversold (with 200 EMA trend filter)
# ---------------------------------------------------------------------------

class RSIOverboughtOversoldStrategy(BaseStrategy):
    name = "RSI (30/70)"
    description = (
        "MEAN-REVERSION strategy using the Relative Strength Index.\n\n"
        "HOW IT WORKS\n"
        "RSI measures the speed and magnitude of recent price changes on "
        "a 0-100 scale. An RSI below 30 means the stock is oversold "
        "(potential bounce). An RSI above 70 means overbought (potential "
        "pullback). A 200 EMA trend filter ensures you only take longs "
        "above the EMA and shorts below it.\n\n"
        "ENTRY RULES\n"
        "  LONG  : RSI crosses up through 30 AND price > 200 EMA\n"
        "  SHORT : RSI crosses down through 70 AND price < 200 EMA\n\n"
        "EXIT RULES\n"
        "  - Opposite RSI signal fires\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Stocks that oscillate within a range\n"
        "  - After sharp drops/rallies that overextend price\n"
        "  - The EMA filter prevents fighting the main trend\n"
        "  - Avoid during strong momentum runs (RSI stays extreme)\n\n"
        "PARAMETERS\n"
        "  RSI Period   : 14\n"
        "  Overbought   : 70\n"
        "  Oversold     : 30\n"
        "  Trend EMA    : 200\n"
        "  ATR Period   : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  1.5x  (mean-reversion — tight safety stop; strategy exits on\n"
        "         opposite RSI signal, or use 0 to rely on RSI only)\n\n"
        "CHART INDICATORS\n"
        "  - Yellow line : 200 EMA (price chart)\n"
        "  - Sub-panel:\n"
        "    - Blue line   : RSI value\n"
        "    - Red dashed  : Overbought level (70)\n"
        "    - Green dashed : Oversold level (30)\n"
        "    - Shaded zones : Overbought/oversold regions"
    )
    needs_subpanel = True

    def __init__(self, rsi_period=14, overbought=70, oversold=30,
                 trend_ema=200, atr_period=14):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.trend_ema = trend_ema
        self.atr_period = atr_period
        self.min_bars = max(rsi_period, trend_ema) + 5

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.rsi_period + 1:
            return self._empty(n)

        # RSI via Wilder smoothing
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        p = self.rsi_period
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        rsi = np.full(n, np.nan)

        if n > p:
            avg_gain[p] = np.mean(gain[1:p + 1])
            avg_loss[p] = np.mean(loss[1:p + 1])
            for i in range(p + 1, n):
                avg_gain[i] = (avg_gain[i - 1] * (p - 1) + gain[i]) / p
                avg_loss[i] = (avg_loss[i - 1] * (p - 1) + loss[i]) / p
            for i in range(p, n):
                if avg_loss[i] == 0:
                    rsi[i] = 100.0
                else:
                    rs = avg_gain[i] / avg_loss[i]
                    rsi[i] = 100.0 - 100.0 / (1.0 + rs)

        ema200 = self._ema(close, self.trend_ema)
        atr = self._atr(high, low, close, self.atr_period)

        direction = np.zeros(n, dtype=int)
        for i in range(p + 1, n):
            if np.isnan(rsi[i]) or np.isnan(rsi[i - 1]):
                direction[i] = direction[i - 1]
                continue
            # Crosses up through oversold
            if rsi[i - 1] < self.oversold and rsi[i] >= self.oversold:
                direction[i] = 1
            # Crosses down through overbought
            elif rsi[i - 1] > self.overbought and rsi[i] <= self.overbought:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        # trade_direction: apply EMA trend filter
        trade_dir = direction.copy()
        for i in range(n):
            if np.isnan(ema200[i]):
                trade_dir[i] = 0
            elif direction[i] > 0 and close[i] <= ema200[i]:
                trade_dir[i] = 0
            elif direction[i] < 0 and close[i] >= ema200[i]:
                trade_dir[i] = 0

        return {
            'rsi': rsi, 'ema200': ema200,
            'direction': direction, 'trade_direction': trade_dir, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        ema = indicators['ema200'][start:]
        m = ~np.isnan(ema)
        if m.any():
            ax_price.plot(x[m], ema[m], color=colors['yellow'],
                          linewidth=1.0, label='200 EMA', zorder=2)
        if ax_sub is None:
            return
        rsi = indicators['rsi'][start:]
        mr = ~np.isnan(rsi)
        if mr.any():
            ax_sub.plot(x[mr], rsi[mr], color='#4fc3f7', linewidth=1.0,
                        label='RSI', zorder=3)
        ax_sub.axhline(y=self.overbought, color=colors['red'],
                       linewidth=0.7, linestyle='--', alpha=0.7)
        ax_sub.axhline(y=self.oversold, color=colors['green'],
                       linewidth=0.7, linestyle='--', alpha=0.7)
        ax_sub.axhline(y=50, color='#555', linewidth=0.5)
        ax_sub.fill_between(x, self.overbought, 100,
                            color=colors['red'], alpha=0.05)
        ax_sub.fill_between(x, 0, self.oversold,
                            color=colors['green'], alpha=0.05)
        ax_sub.set_ylim(0, 100)
        ax_sub.legend(loc='upper left', fontsize=7,
                      facecolor='#111', edgecolor='#333', labelcolor='#ccc')


# ---------------------------------------------------------------------------
# Strategy: Squeeze Momentum (Bollinger inside Keltner)
# ---------------------------------------------------------------------------

class SqueezeMomentumStrategy(BaseStrategy):
    name = "Squeeze Momentum"
    description = (
        "MOMENTUM strategy detecting volatility compression then expansion.\n\n"
        "HOW IT WORKS\n"
        "A 'squeeze' occurs when Bollinger Bands contract inside the "
        "Keltner Channels -- this means volatility is compressing and a "
        "big move is about to happen. When the squeeze releases (BB "
        "expand outside KC) the strategy enters in the direction of the "
        "momentum histogram. Red dots on the zero line = squeeze ON, "
        "green dots = squeeze OFF.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Squeeze releases AND momentum histogram is positive\n"
        "  SHORT : Squeeze releases AND momentum histogram is negative\n"
        "  (no entries during active squeeze -- direction is neutral)\n\n"
        "EXIT RULES\n"
        "  - Momentum flips to opposite direction\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Stocks consolidating before a big move\n"
        "  - Earnings run-ups, pre-breakout patterns\n"
        "  - Works on any timeframe; best with 1-5 min bars\n"
        "  - The longer the squeeze, the bigger the expected move\n\n"
        "PARAMETERS\n"
        "  BB Period : 20    KC Period : 20\n"
        "  BB StdDev : 2.0   KC Mult   : 1.5\n"
        "  ATR Period : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  2.0x  (momentum — captures the big move after the squeeze fires)\n\n"
        "CHART INDICATORS\n"
        "  - Blue dashed  : Bollinger Bands (price chart)\n"
        "  - Orange dotted : Keltner Channels (price chart)\n"
        "  - Sub-panel:\n"
        "    - Green/Red bars : Momentum histogram\n"
        "    - Red dots  : Squeeze is ON (compression)\n"
        "    - Green dots : Squeeze is OFF (expansion)"
    )
    needs_subpanel = True

    def __init__(self, bb_period=20, bb_std=2.0, kc_period=20,
                 kc_mult=1.5, atr_period=14):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        self.atr_period = atr_period
        self.min_bars = max(bb_period, kc_period) + 20

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        # Bollinger Bands
        bb_mid = self._sma(close, self.bb_period)
        bb_std_arr = self._rolling_std(close, self.bb_period)
        bb_upper = bb_mid + self.bb_std * bb_std_arr
        bb_lower = bb_mid - self.bb_std * bb_std_arr

        # Keltner Channels
        kc_mid = self._ema(close, self.kc_period)
        kc_atr = self._atr(high, low, close, self.kc_period)
        kc_upper = kc_mid + self.kc_mult * kc_atr
        kc_lower = kc_mid - self.kc_mult * kc_atr

        # Squeeze detection
        squeeze_on = np.zeros(n, dtype=bool)
        for i in range(n):
            if (not np.isnan(bb_lower[i]) and not np.isnan(kc_lower[i])):
                squeeze_on[i] = (bb_lower[i] > kc_lower[i] and
                                 bb_upper[i] < kc_upper[i])

        # Momentum: close minus midline of donchian(bb_period), smoothed
        momentum = np.full(n, np.nan)
        for i in range(self.bb_period, n):
            donchian_mid = (np.max(high[i - self.bb_period:i + 1]) +
                            np.min(low[i - self.bb_period:i + 1])) / 2.0
            avg_mid = (donchian_mid + bb_mid[i]) / 2.0 if not np.isnan(bb_mid[i]) else donchian_mid
            momentum[i] = close[i] - avg_mid

        atr = self._atr(high, low, close, self.atr_period)

        direction = np.zeros(n, dtype=int)
        was_squeezing = False
        for i in range(self.min_bars, n):
            if np.isnan(momentum[i]):
                direction[i] = direction[i - 1]
                continue
            if squeeze_on[i]:
                was_squeezing = True
                direction[i] = 0  # neutral during squeeze
            else:
                if was_squeezing:
                    # Squeeze just released
                    was_squeezing = False
                    direction[i] = 1 if momentum[i] > 0 else -1
                else:
                    # Momentum continuation
                    if momentum[i] > 0:
                        direction[i] = 1
                    elif momentum[i] < 0:
                        direction[i] = -1
                    else:
                        direction[i] = direction[i - 1]

        return {
            'bb_upper': bb_upper, 'bb_lower': bb_lower,
            'kc_upper': kc_upper, 'kc_lower': kc_lower,
            'squeeze_on': squeeze_on, 'momentum': momentum,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        bbu = indicators['bb_upper'][start:]
        bbl = indicators['bb_lower'][start:]
        kcu = indicators['kc_upper'][start:]
        kcl = indicators['kc_lower'][start:]
        mb = ~np.isnan(bbu)
        mk = ~np.isnan(kcu)
        if mb.any():
            ax_price.plot(x[mb], bbu[mb], color='#4fc3f7', linewidth=0.7,
                          linestyle='--', label='BB', zorder=2)
            ax_price.plot(x[mb], bbl[mb], color='#4fc3f7', linewidth=0.7,
                          linestyle='--', zorder=2)
        if mk.any():
            ax_price.plot(x[mk], kcu[mk], color='#ff9800', linewidth=0.7,
                          linestyle=':', label='KC', zorder=2)
            ax_price.plot(x[mk], kcl[mk], color='#ff9800', linewidth=0.7,
                          linestyle=':', zorder=2)
        if ax_sub is None:
            return
        mom = indicators['momentum'][start:]
        sq = indicators['squeeze_on'][start:]
        mm = ~np.isnan(mom)
        if mm.any():
            pos = np.where(mom > 0, mom, 0)
            neg = np.where(mom < 0, mom, 0)
            ax_sub.bar(x[mm], pos[mm], color=colors['green'], alpha=0.5, width=0.8)
            ax_sub.bar(x[mm], neg[mm], color=colors['red'], alpha=0.5, width=0.8)
        # Squeeze dots on zero line
        for i in range(len(sq)):
            c = colors['red'] if sq[i] else colors['green']
            ax_sub.plot(x[i], 0, marker='o', color=c, markersize=3, zorder=5)
        ax_sub.axhline(y=0, color='#555', linewidth=0.5)
        ax_sub.set_title('Squeeze Momentum', color='#888', fontsize=8, loc='left')


# ---------------------------------------------------------------------------
# Strategy: OBV Divergence
# ---------------------------------------------------------------------------

class OBVDivergenceStrategy(BaseStrategy):
    name = "OBV Divergence"
    description = (
        "VOLUME-BASED strategy detecting divergences between price and OBV.\n\n"
        "HOW IT WORKS\n"
        "On-Balance Volume (OBV) is a cumulative total that adds volume on "
        "up-closes and subtracts it on down-closes. A bullish divergence "
        "occurs when price makes a lower low but OBV makes a higher low "
        "(smart money accumulating). A bearish divergence occurs when price "
        "makes a higher high but OBV makes a lower high (distribution).\n\n"
        "ENTRY RULES\n"
        "  LONG  : Price makes lower low + OBV makes higher low\n"
        "          (bullish divergence = accumulation)\n"
        "  SHORT : Price makes higher high + OBV makes lower high\n"
        "          (bearish divergence = distribution)\n\n"
        "EXIT RULES\n"
        "  - Opposite divergence signal fires\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - High-volume stocks where volume leads price\n"
        "  - End of trends when smart money is repositioning\n"
        "  - Less effective on low-volume or meme stocks\n"
        "  - Works best when combined with support/resistance levels\n\n"
        "PARAMETERS\n"
        "  OBV EMA Period : 20 (smoothing for OBV display)\n"
        "  Lookback       : 14 (window for swing detection)\n"
        "  ATR Period     : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  1.5x  (divergence — tight stop protects against failed reversals)\n\n"
        "CHART INDICATORS (sub-panel)\n"
        "  - Blue line   : Raw OBV\n"
        "  - Orange line : OBV EMA (smoothed trend)"
    )
    needs_subpanel = True

    def __init__(self, obv_ema_period=20, lookback=14, atr_period=14):
        self.obv_ema_period = obv_ema_period
        self.lookback = lookback
        self.atr_period = atr_period
        self.min_bars = max(obv_ema_period, lookback) + 20

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        volume = np.array([getattr(b, 'volume', 0) for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        # OBV
        obv = np.zeros(n)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        obv_ema = self._ema(obv, self.obv_ema_period)
        atr = self._atr(high, low, close, self.atr_period)

        # Simple divergence detection using swing highs/lows
        lb = self.lookback
        direction = np.zeros(n, dtype=int)
        for i in range(lb * 2, n):
            window_c = close[i - lb:i + 1]
            window_o = obv[i - lb:i + 1]

            # Price making lower low, OBV making higher low = bullish divergence
            price_ll = (close[i] < np.min(close[i - lb:i]) and
                        close[i] < close[i - lb])
            obv_hl = obv[i] > np.min(obv[i - lb:i])

            # Price making higher high, OBV making lower high = bearish divergence
            price_hh = (close[i] > np.max(close[i - lb:i]) and
                        close[i] > close[i - lb])
            obv_lh = obv[i] < np.max(obv[i - lb:i])

            if price_ll and obv_hl:
                direction[i] = 1   # bullish divergence
            elif price_hh and obv_lh:
                direction[i] = -1  # bearish divergence
            else:
                direction[i] = direction[i - 1]

        return {
            'obv': obv, 'obv_ema': obv_ema,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        if ax_sub is None:
            return
        obv = indicators['obv'][start:]
        obv_ema = indicators['obv_ema'][start:]
        ax_sub.plot(x, obv, color='#4fc3f7', linewidth=1.0,
                    label='OBV', zorder=3)
        m = ~np.isnan(obv_ema)
        if m.any():
            ax_sub.plot(x[m], obv_ema[m], color='#ff9800', linewidth=1.0,
                        label='OBV EMA', zorder=3)
        ax_sub.legend(loc='upper left', fontsize=7,
                      facecolor='#111', edgecolor='#333', labelcolor='#ccc')


# ---------------------------------------------------------------------------
# Strategy: Ichimoku Cloud
# ---------------------------------------------------------------------------

class IchimokuCloudStrategy(BaseStrategy):
    name = "Ichimoku Cloud"
    description = (
        "TREND-FOLLOWING strategy using the Ichimoku Kinko Hyo system.\n\n"
        "HOW IT WORKS\n"
        "Ichimoku is a comprehensive system that shows support/resistance, "
        "trend direction, and momentum at a glance. The 'cloud' (Kumo) is "
        "formed by Senkou Span A and B projected 26 bars forward. A green "
        "cloud is bullish, a red cloud is bearish. The Tenkan-sen (fast) "
        "and Kijun-sen (slow) lines act like moving average crossovers.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Tenkan > Kijun AND price > cloud top\n"
        "          (all three conditions must align)\n"
        "  SHORT : Tenkan < Kijun AND price < cloud bottom\n\n"
        "EXIT RULES\n"
        "  - Any condition breaks (e.g., price enters cloud)\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Strongly trending markets with clear momentum\n"
        "  - Works well on higher timeframes (5-min bars used)\n"
        "  - The thicker the cloud, the stronger the support/resistance\n"
        "  - Avoid when price is inside the cloud (choppy, no edge)\n\n"
        "PARAMETERS\n"
        "  Tenkan    : 9   (fast conversion line)\n"
        "  Kijun     : 26  (slow base line)\n"
        "  Senkou B  : 52  (cloud span B)\n"
        "  Displacement : 26 (cloud projection forward)\n"
        "  ATR Period   : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  3.0x  (trend strategy — wide stop lets the cloud system work;\n"
        "         Ichimoku has its own exit when price enters cloud)\n\n"
        "CHART INDICATORS\n"
        "  - Blue line  : Tenkan-sen (fast)\n"
        "  - Red line   : Kijun-sen (slow)\n"
        "  - Green line : Chikou Span (lagging)\n"
        "  - Green cloud : Bullish (Span A > Span B)\n"
        "  - Red cloud   : Bearish (Span A < Span B)"
    )
    bar_size = '5 mins'
    duration = '10 D'

    def __init__(self, tenkan_period=9, kijun_period=26, senkou_b_period=52,
                 displacement=26, atr_period=14):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        self.atr_period = atr_period
        self.min_bars = senkou_b_period + displacement + 5

    def _donchian_mid(self, high, low, period):
        n = len(high)
        mid = np.full(n, np.nan)
        for i in range(period - 1, n):
            mid[i] = (np.max(high[i - period + 1:i + 1]) +
                      np.min(low[i - period + 1:i + 1])) / 2.0
        return mid

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        tenkan = self._donchian_mid(high, low, self.tenkan_period)
        kijun = self._donchian_mid(high, low, self.kijun_period)

        # Senkou A & B displaced forward
        senkou_a = np.full(n, np.nan)
        senkou_b = np.full(n, np.nan)
        raw_a = (tenkan + kijun) / 2.0
        raw_b = self._donchian_mid(high, low, self.senkou_b_period)
        d = self.displacement
        for i in range(n):
            if not np.isnan(raw_a[i]) and i + d < n:
                senkou_a[i + d] = raw_a[i]
            if not np.isnan(raw_b[i]) and i + d < n:
                senkou_b[i + d] = raw_b[i]

        # Chikou (lagging span) = close displaced back
        chikou = np.full(n, np.nan)
        for i in range(d, n):
            chikou[i - d] = close[i]

        atr = self._atr(high, low, close, self.atr_period)

        # Direction
        direction = np.zeros(n, dtype=int)
        for i in range(1, n):
            if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
                    np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])):
                direction[i] = direction[i - 1]
                continue
            cloud_top = max(senkou_a[i], senkou_b[i])
            cloud_bot = min(senkou_a[i], senkou_b[i])

            if (tenkan[i] > kijun[i] and close[i] > cloud_top):
                direction[i] = 1
            elif (tenkan[i] < kijun[i] and close[i] < cloud_bot):
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        return {
            'tenkan': tenkan, 'kijun': kijun,
            'senkou_a': senkou_a, 'senkou_b': senkou_b, 'chikou': chikou,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        tenkan = indicators['tenkan'][start:]
        kijun = indicators['kijun'][start:]
        sa = indicators['senkou_a'][start:]
        sb = indicators['senkou_b'][start:]
        chikou = indicators['chikou'][start:]

        mt = ~np.isnan(tenkan)
        mk = ~np.isnan(kijun)
        if mt.any():
            ax_price.plot(x[mt], tenkan[mt], color='#4fc3f7', linewidth=0.8,
                          label='Tenkan', zorder=2)
        if mk.any():
            ax_price.plot(x[mk], kijun[mk], color=colors['red'], linewidth=0.8,
                          label='Kijun', zorder=2)

        mc = ~np.isnan(chikou)
        if mc.any():
            ax_price.plot(x[mc], chikou[mc], color=colors['green'],
                          linewidth=0.6, linestyle=':', alpha=0.7,
                          label='Chikou', zorder=2)

        # Cloud fill
        both = ~np.isnan(sa) & ~np.isnan(sb)
        if both.any():
            ax_price.fill_between(
                x[both], sa[both], sb[both],
                where=sa[both] >= sb[both],
                color=colors['green'], alpha=0.10, zorder=1)
            ax_price.fill_between(
                x[both], sa[both], sb[both],
                where=sa[both] < sb[both],
                color=colors['red'], alpha=0.10, zorder=1)
            ax_price.plot(x[both], sa[both], color=colors['green'],
                          linewidth=0.5, alpha=0.5, zorder=1)
            ax_price.plot(x[both], sb[both], color=colors['red'],
                          linewidth=0.5, alpha=0.5, zorder=1)


# ---------------------------------------------------------------------------
# Strategy: Keltner Channel + ADX
# ---------------------------------------------------------------------------

class KeltnerADXStrategy(BaseStrategy):
    name = "Keltner + ADX"
    description = (
        "TREND-FOLLOWING strategy using Keltner Channel breakouts filtered "
        "by ADX trend strength.\n\n"
        "HOW IT WORKS\n"
        "Keltner Channels draw bands at N x ATR above and below an EMA. "
        "The ADX (Average Directional Index) measures trend strength on a "
        "0-100 scale. Signals only fire when ADX > 25, confirming a real "
        "trend and filtering out noise in choppy markets.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Close breaks above upper Keltner AND ADX > 25\n"
        "  SHORT : Close breaks below lower Keltner AND ADX > 25\n\n"
        "EXIT RULES\n"
        "  - Price returns inside the channel\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Strong trending markets with high ADX readings\n"
        "  - Momentum breakouts with institutional volume\n"
        "  - Avoids false breakouts in low-ADX choppy markets\n\n"
        "PARAMETERS\n"
        "  EMA Period   : 20\n"
        "  KC Multiplier : 2.0 (channel width)\n"
        "  ADX Period    : 14\n"
        "  ADX Threshold : 25 (minimum trend strength)\n"
        "  ATR Period    : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  2.0x  (trend breakout — captures the ADX-confirmed move)\n\n"
        "CHART INDICATORS\n"
        "  - Yellow line  : 20 EMA (channel midline)\n"
        "  - Green dashed : Upper Keltner band\n"
        "  - Red dashed   : Lower Keltner band\n"
        "  - Sub-panel    : ADX line with threshold"
    )
    needs_subpanel = True

    def __init__(self, ema_period=20, kc_mult=2.0, adx_period=14,
                 adx_threshold=25, atr_period=14):
        self.ema_period = ema_period
        self.kc_mult = kc_mult
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.min_bars = max(ema_period, adx_period * 2) + 10

    def _adx(self, high, low, close, period):
        n = len(close)
        adx = np.full(n, np.nan)
        if n < period * 2:
            return adx
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm[i] = up if (up > down and up > 0) else 0.0
            minus_dm[i] = down if (down > up and down > 0) else 0.0
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
        # Wilder smoothing
        atr_s = np.full(n, np.nan)
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        atr_s[period] = np.mean(tr[1:period + 1])
        sm_plus = np.mean(plus_dm[1:period + 1])
        sm_minus = np.mean(minus_dm[1:period + 1])
        for i in range(period + 1, n):
            atr_s[i] = (atr_s[i - 1] * (period - 1) + tr[i]) / period
            sm_plus = (sm_plus * (period - 1) + plus_dm[i]) / period
            sm_minus = (sm_minus * (period - 1) + minus_dm[i]) / period
            if atr_s[i] > 0:
                plus_di[i] = 100.0 * sm_plus / atr_s[i]
                minus_di[i] = 100.0 * sm_minus / atr_s[i]
        dx = np.full(n, np.nan)
        for i in range(period + 1, n):
            if (not np.isnan(plus_di[i]) and not np.isnan(minus_di[i])
                    and (plus_di[i] + minus_di[i]) > 0):
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        # Smooth DX into ADX
        valid_dx = [(i, dx[i]) for i in range(n) if not np.isnan(dx[i])]
        if len(valid_dx) >= period:
            start_idx = valid_dx[period - 1][0]
            adx[start_idx] = np.mean([v for _, v in valid_dx[:period]])
            vd_map = {i: v for i, v in valid_dx}
            for i in range(start_idx + 1, n):
                if not np.isnan(adx[i - 1]) and i in vd_map:
                    adx[i] = (adx[i - 1] * (period - 1) + vd_map[i]) / period
        return adx

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        ema = self._ema(close, self.ema_period)
        atr = self._atr(high, low, close, self.atr_period)
        kc_upper = ema + self.kc_mult * atr
        kc_lower = ema - self.kc_mult * atr
        adx = self._adx(high, low, close, self.adx_period)

        direction = np.zeros(n, dtype=int)
        for i in range(1, n):
            if (np.isnan(kc_upper[i]) or np.isnan(adx[i])):
                direction[i] = direction[i - 1]
                continue
            if close[i] > kc_upper[i] and adx[i] >= self.adx_threshold:
                direction[i] = 1
            elif close[i] < kc_lower[i] and adx[i] >= self.adx_threshold:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        return {
            'ema': ema, 'kc_upper': kc_upper, 'kc_lower': kc_lower,
            'adx': adx, 'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        ema = indicators['ema'][start:]
        upper = indicators['kc_upper'][start:]
        lower = indicators['kc_lower'][start:]
        me = ~np.isnan(ema)
        if me.any():
            ax_price.plot(x[me], ema[me], color=colors['yellow'],
                          linewidth=1.0, label='EMA', zorder=2)
            ax_price.plot(x[me], upper[me], color=colors['green'],
                          linewidth=0.8, linestyle='--', label='KC Upper', zorder=2)
            ax_price.plot(x[me], lower[me], color=colors['red'],
                          linewidth=0.8, linestyle='--', label='KC Lower', zorder=2)
            ax_price.fill_between(x[me], upper[me], lower[me],
                                  color='#4fc3f7', alpha=0.05, zorder=1)
        if ax_sub is None:
            return
        adx = indicators['adx'][start:]
        ma = ~np.isnan(adx)
        if ma.any():
            ax_sub.plot(x[ma], adx[ma], color='#4fc3f7', linewidth=1.2,
                        label='ADX', zorder=3)
        ax_sub.axhline(y=self.adx_threshold, color=colors['yellow'],
                       linewidth=0.7, linestyle='--', alpha=0.7)
        ax_sub.set_ylim(0, 60)
        ax_sub.set_title('ADX', color='#888', fontsize=8, loc='left')
        ax_sub.legend(loc='upper left', fontsize=7,
                      facecolor='#111', edgecolor='#333', labelcolor='#ccc')


# ---------------------------------------------------------------------------
# Strategy: Z-Score Mean Reversion
# ---------------------------------------------------------------------------

class ZScoreMeanReversionStrategy(BaseStrategy):
    name = "Z-Score Reversion"
    description = (
        "MEAN-REVERSION strategy using statistical z-score extremes.\n\n"
        "HOW IT WORKS\n"
        "Computes a rolling z-score: how many standard deviations price is "
        "from its moving average. When z-score drops below -2.0 the stock "
        "is statistically oversold; when above +2.0 it is overbought. More "
        "precise than Bollinger Bands because it normalizes for volatility.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Z-score drops below -2.0 (2 std devs below mean)\n"
        "  SHORT : Z-score rises above +2.0 (2 std devs above mean)\n\n"
        "EXIT RULES\n"
        "  - Z-score returns to 0 (mean reversion complete)\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Range-bound stocks with stable volatility\n"
        "  - After sharp over-extensions that snap back\n"
        "  - Avoid during regime changes or trend breakouts\n\n"
        "PARAMETERS\n"
        "  Lookback     : 20 (rolling window for mean/std)\n"
        "  Entry Z      : 2.0 (entry threshold)\n"
        "  Exit Z       : 0.0 (exit at mean)\n"
        "  ATR Period   : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  1.0x  (mean-reversion — tight safety net; strategy exits at\n"
        "         z-score 0 (mean), or use 0 to rely on z-score only)\n\n"
        "CHART INDICATORS\n"
        "  - Yellow line : 20-bar SMA (mean)\n"
        "  - Sub-panel   : Z-score with +/-2 thresholds"
    )
    needs_subpanel = True

    def __init__(self, lookback=20, entry_z=2.0, exit_z=0.0, atr_period=14):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.atr_period = atr_period
        self.min_bars = lookback + 10

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        sma = self._sma(close, self.lookback)
        std = self._rolling_std(close, self.lookback)
        atr = self._atr(high, low, close, self.atr_period)

        zscore = np.full(n, np.nan)
        for i in range(self.lookback - 1, n):
            if not np.isnan(sma[i]) and not np.isnan(std[i]) and std[i] > 0:
                zscore[i] = (close[i] - sma[i]) / std[i]

        direction = np.zeros(n, dtype=int)
        for i in range(1, n):
            if np.isnan(zscore[i]):
                direction[i] = direction[i - 1]
                continue
            if zscore[i] <= -self.entry_z:
                direction[i] = 1
            elif zscore[i] >= self.entry_z:
                direction[i] = -1
            elif direction[i - 1] == 1 and zscore[i] >= self.exit_z:
                direction[i] = 0
            elif direction[i - 1] == -1 and zscore[i] <= -self.exit_z:
                direction[i] = 0
            else:
                direction[i] = direction[i - 1]

        return {
            'sma': sma, 'zscore': zscore,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        sma = indicators['sma'][start:]
        m = ~np.isnan(sma)
        if m.any():
            ax_price.plot(x[m], sma[m], color=colors['yellow'],
                          linewidth=1.0, label='SMA', zorder=2)
        if ax_sub is None:
            return
        zs = indicators['zscore'][start:]
        mz = ~np.isnan(zs)
        if mz.any():
            ax_sub.plot(x[mz], zs[mz], color='#4fc3f7', linewidth=1.0,
                        label='Z-Score', zorder=3)
        ax_sub.axhline(y=self.entry_z, color=colors['red'],
                       linewidth=0.7, linestyle='--', alpha=0.7)
        ax_sub.axhline(y=-self.entry_z, color=colors['green'],
                       linewidth=0.7, linestyle='--', alpha=0.7)
        ax_sub.axhline(y=0, color='#555', linewidth=0.5)
        ax_sub.fill_between(x, self.entry_z, 4, color=colors['red'], alpha=0.05)
        ax_sub.fill_between(x, -4, -self.entry_z, color=colors['green'], alpha=0.05)
        ax_sub.set_ylim(-4, 4)
        ax_sub.set_title('Z-Score', color='#888', fontsize=8, loc='left')
        ax_sub.legend(loc='upper left', fontsize=7,
                      facecolor='#111', edgecolor='#333', labelcolor='#ccc')


# ---------------------------------------------------------------------------
# Strategy: Rate of Change Divergence
# ---------------------------------------------------------------------------

class ROCDivergenceStrategy(BaseStrategy):
    name = "ROC Divergence"
    description = (
        "MOMENTUM strategy comparing Rate of Change of price vs volume.\n\n"
        "HOW IT WORKS\n"
        "Rate of Change measures the percentage change over N bars. When "
        "price ROC and volume ROC diverge it signals exhaustion. A bullish "
        "divergence occurs when price ROC makes a lower low but volume ROC "
        "makes a higher low (buying pressure building). Bearish divergence "
        "is the opposite.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Price ROC < 0 AND volume ROC rising (bullish divergence)\n"
        "  SHORT : Price ROC > 0 AND volume ROC falling (bearish divergence)\n\n"
        "EXIT RULES\n"
        "  - ROC values re-align (divergence resolves)\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - End of strong trends where momentum is fading\n"
        "  - High-volume stocks where volume leads price\n"
        "  - Works well as a reversal signal at extremes\n\n"
        "PARAMETERS\n"
        "  ROC Period   : 12\n"
        "  Signal EMA   : 9 (smoothing for ROC)\n"
        "  ATR Period   : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  1.5x  (divergence — protects against failed reversal signals)\n\n"
        "CHART INDICATORS (sub-panel)\n"
        "  - Blue line   : Price ROC\n"
        "  - Orange line : Volume ROC\n"
        "  - Zero line   : Neutral"
    )
    needs_subpanel = True

    def __init__(self, roc_period=12, signal_period=9, atr_period=14):
        self.roc_period = roc_period
        self.signal_period = signal_period
        self.atr_period = atr_period
        self.min_bars = roc_period + signal_period + 10

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        volume = np.array([getattr(b, 'volume', 0) for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        p = self.roc_period
        price_roc = np.full(n, np.nan)
        vol_roc = np.full(n, np.nan)
        for i in range(p, n):
            if close[i - p] != 0:
                price_roc[i] = (close[i] - close[i - p]) / close[i - p] * 100.0
            vol_sum = np.sum(volume[i - p + 1:i + 1])
            vol_prev = np.sum(volume[i - 2 * p + 1:i - p + 1]) if i >= 2 * p else 1.0
            if vol_prev > 0:
                vol_roc[i] = (vol_sum - vol_prev) / vol_prev * 100.0

        price_roc_ema = self._ema(price_roc[~np.isnan(price_roc)], self.signal_period)
        vol_roc_ema = self._ema(vol_roc[~np.isnan(vol_roc)], self.signal_period)

        # Re-align to full arrays
        pr_smooth = np.full(n, np.nan)
        vr_smooth = np.full(n, np.nan)
        pr_idx = np.where(~np.isnan(price_roc))[0]
        vr_idx = np.where(~np.isnan(vol_roc))[0]
        if len(price_roc_ema) == len(pr_idx):
            pr_smooth[pr_idx] = price_roc_ema
        if len(vol_roc_ema) == len(vr_idx):
            vr_smooth[vr_idx] = vol_roc_ema

        atr = self._atr(high, low, close, self.atr_period)

        direction = np.zeros(n, dtype=int)
        for i in range(1, n):
            if np.isnan(pr_smooth[i]) or np.isnan(vr_smooth[i]):
                direction[i] = direction[i - 1]
                continue
            # Bullish divergence: price falling but volume rising
            if pr_smooth[i] < 0 and vr_smooth[i] > 0:
                direction[i] = 1
            # Bearish divergence: price rising but volume falling
            elif pr_smooth[i] > 0 and vr_smooth[i] < 0:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        return {
            'price_roc': pr_smooth, 'vol_roc': vr_smooth,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        if ax_sub is None:
            return
        pr = indicators['price_roc'][start:]
        vr = indicators['vol_roc'][start:]
        mp = ~np.isnan(pr)
        mv = ~np.isnan(vr)
        if mp.any():
            ax_sub.plot(x[mp], pr[mp], color='#4fc3f7', linewidth=1.0,
                        label='Price ROC', zorder=3)
        if mv.any():
            ax_sub.plot(x[mv], vr[mv], color='#ff9800', linewidth=1.0,
                        label='Vol ROC', zorder=3)
        ax_sub.axhline(y=0, color='#555', linewidth=0.5)
        ax_sub.set_title('ROC Divergence', color='#888', fontsize=8, loc='left')
        ax_sub.legend(loc='upper left', fontsize=7,
                      facecolor='#111', edgecolor='#333', labelcolor='#ccc')


# ---------------------------------------------------------------------------
# Strategy: Triple Screen (Elder)
# ---------------------------------------------------------------------------

class TripleScreenStrategy(BaseStrategy):
    name = "Triple Screen"
    description = (
        "MULTI-TIMEFRAME strategy based on Dr. Alexander Elder's system.\n\n"
        "HOW IT WORKS\n"
        "Uses three 'screens' (filters) that must all agree:\n"
        "  1. TREND  : Long EMA (100) determines the primary trend\n"
        "  2. MOMENTUM : MACD histogram rising/falling confirms timing\n"
        "  3. TRIGGER : Short EMA (13) crossover provides entry\n"
        "All three must align for a signal. This dramatically reduces "
        "false signals by requiring multi-timeframe confluence.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Price > 100 EMA AND MACD histogram rising AND\n"
        "          13 EMA > 26 EMA\n"
        "  SHORT : Price < 100 EMA AND MACD histogram falling AND\n"
        "          13 EMA < 26 EMA\n\n"
        "EXIT RULES\n"
        "  - Any screen breaks alignment\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Works in all market conditions (filters adapt)\n"
        "  - Fewer trades but higher conviction\n"
        "  - Best on trending stocks with clear momentum\n\n"
        "PARAMETERS\n"
        "  Trend EMA     : 100 (primary trend)\n"
        "  Fast EMA      : 13  (trigger)\n"
        "  Slow EMA      : 26  (trigger)\n"
        "  MACD Signal   : 9\n"
        "  ATR Period    : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  2.5x  (high-conviction trend — wide stop lets multi-screen\n"
        "         alignment play out fully)\n\n"
        "CHART INDICATORS\n"
        "  - Yellow line  : 100 EMA (trend)\n"
        "  - Blue line    : 13 EMA (fast trigger)\n"
        "  - Sub-panel    : MACD histogram"
    )
    needs_subpanel = True

    def __init__(self, trend_ema=100, fast_ema=13, slow_ema=26,
                 signal_ema=9, atr_period=14):
        self.trend_ema = trend_ema
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.signal_ema = signal_ema
        self.atr_period = atr_period
        self.min_bars = trend_ema + signal_ema + 5

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        trend = self._ema(close, self.trend_ema)
        fast = self._ema(close, self.fast_ema)
        slow = self._ema(close, self.slow_ema)
        atr = self._atr(high, low, close, self.atr_period)

        # MACD histogram
        macd_line = fast - slow
        valid_macd = macd_line[~np.isnan(macd_line)]
        sig_line_raw = self._ema(valid_macd, self.signal_ema)
        sig_line = np.full(n, np.nan)
        valid_idx = np.where(~np.isnan(macd_line))[0]
        if len(sig_line_raw) == len(valid_idx):
            sig_line[valid_idx] = sig_line_raw
        histogram = macd_line - sig_line

        direction = np.zeros(n, dtype=int)
        for i in range(2, n):
            if (np.isnan(trend[i]) or np.isnan(fast[i]) or
                    np.isnan(slow[i]) or np.isnan(histogram[i]) or
                    np.isnan(histogram[i - 1])):
                direction[i] = direction[i - 1]
                continue
            # Screen 1: Trend
            trend_bull = close[i] > trend[i]
            trend_bear = close[i] < trend[i]
            # Screen 2: Momentum (histogram rising/falling)
            mom_bull = histogram[i] > histogram[i - 1]
            mom_bear = histogram[i] < histogram[i - 1]
            # Screen 3: Trigger (fast/slow crossover)
            trigger_bull = fast[i] > slow[i]
            trigger_bear = fast[i] < slow[i]

            if trend_bull and mom_bull and trigger_bull:
                direction[i] = 1
            elif trend_bear and mom_bear and trigger_bear:
                direction[i] = -1
            else:
                direction[i] = 0

        return {
            'trend_ema': trend, 'fast_ema': fast, 'slow_ema': slow,
            'histogram': histogram,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        trend = indicators['trend_ema'][start:]
        fast = indicators['fast_ema'][start:]
        mt = ~np.isnan(trend)
        mf = ~np.isnan(fast)
        if mt.any():
            ax_price.plot(x[mt], trend[mt], color=colors['yellow'],
                          linewidth=1.2, label='100 EMA', zorder=2)
        if mf.any():
            ax_price.plot(x[mf], fast[mf], color='#4fc3f7',
                          linewidth=0.8, label='13 EMA', zorder=2)
        if ax_sub is None:
            return
        hist = indicators['histogram'][start:]
        mh = ~np.isnan(hist)
        if mh.any():
            pos = np.where(hist > 0, hist, 0)
            neg = np.where(hist < 0, hist, 0)
            ax_sub.bar(x[mh], pos[mh], color=colors['green'], alpha=0.5, width=0.8)
            ax_sub.bar(x[mh], neg[mh], color=colors['red'], alpha=0.5, width=0.8)
        ax_sub.axhline(y=0, color='#555', linewidth=0.5)
        ax_sub.set_title('MACD Histogram (Triple Screen)', color='#888',
                         fontsize=8, loc='left')


# ---------------------------------------------------------------------------
# Strategy: Stochastic RSI + VWAP
# ---------------------------------------------------------------------------

class StochRSIVWAPStrategy(BaseStrategy):
    name = "StochRSI + VWAP"
    description = (
        "MOMENTUM strategy combining Stochastic RSI with VWAP bias.\n\n"
        "HOW IT WORKS\n"
        "Stochastic RSI applies the Stochastic oscillator formula to RSI "
        "values instead of price, catching momentum shifts faster than raw "
        "RSI. VWAP provides intraday directional bias. Signals only fire "
        "when StochRSI crosses extreme levels AND VWAP confirms direction.\n\n"
        "ENTRY RULES\n"
        "  LONG  : StochRSI crosses up through 20 AND price > VWAP\n"
        "  SHORT : StochRSI crosses down through 80 AND price < VWAP\n\n"
        "EXIT RULES\n"
        "  - StochRSI reaches opposite extreme\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Intraday momentum plays with clear VWAP bias\n"
        "  - Catches reversals faster than standard RSI\n"
        "  - Best during active trading hours with volume\n\n"
        "PARAMETERS\n"
        "  RSI Period     : 14\n"
        "  Stoch Period   : 14\n"
        "  Smooth K       : 3\n"
        "  Overbought     : 80\n"
        "  Oversold       : 20\n"
        "  ATR Period     : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  1.5x  (momentum — moderate stop; strategy exits at opposite\n"
        "         StochRSI extreme, trail adds protection)\n\n"
        "CHART INDICATORS\n"
        "  - Blue line    : VWAP (price chart)\n"
        "  - Sub-panel    : StochRSI %K with 80/20 levels"
    )
    needs_subpanel = True

    def __init__(self, rsi_period=14, stoch_period=14, smooth_k=3,
                 overbought=80, oversold=20, atr_period=14):
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.smooth_k = smooth_k
        self.overbought = overbought
        self.oversold = oversold
        self.atr_period = atr_period
        self.min_bars = rsi_period + stoch_period + smooth_k + 10

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        volume = np.array([getattr(b, 'volume', 0) for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        # RSI
        p = self.rsi_period
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        rsi = np.full(n, np.nan)
        if n > p:
            avg_gain = np.full(n, np.nan)
            avg_loss = np.full(n, np.nan)
            avg_gain[p] = np.mean(gain[1:p + 1])
            avg_loss[p] = np.mean(loss[1:p + 1])
            for i in range(p + 1, n):
                avg_gain[i] = (avg_gain[i - 1] * (p - 1) + gain[i]) / p
                avg_loss[i] = (avg_loss[i - 1] * (p - 1) + loss[i]) / p
            for i in range(p, n):
                if avg_loss[i] == 0:
                    rsi[i] = 100.0
                else:
                    rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])

        # Stochastic of RSI
        sp = self.stoch_period
        stoch_rsi = np.full(n, np.nan)
        for i in range(p + sp - 1, n):
            window = rsi[i - sp + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= sp:
                lo = np.min(valid)
                hi = np.max(valid)
                if hi - lo > 0:
                    stoch_rsi[i] = (rsi[i] - lo) / (hi - lo) * 100.0
                else:
                    stoch_rsi[i] = 50.0

        # Smooth %K
        valid_sr = stoch_rsi[~np.isnan(stoch_rsi)]
        smooth = self._sma(valid_sr, self.smooth_k) if len(valid_sr) >= self.smooth_k else valid_sr
        stoch_k = np.full(n, np.nan)
        sr_idx = np.where(~np.isnan(stoch_rsi))[0]
        if len(smooth) == len(sr_idx):
            stoch_k[sr_idx] = smooth

        # VWAP
        _ET = ZoneInfo('America/New_York')
        vwap = np.full(n, np.nan)
        typical = (high + low + close) / 3.0
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_date = None
        for i in range(n):
            try:
                dt = bars[i].date
                if hasattr(dt, 'astimezone'):
                    dt = dt.astimezone(_ET)
                cur_date = dt.date() if hasattr(dt, 'date') else None
            except Exception:
                cur_date = prev_date
            if cur_date != prev_date:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_date = cur_date
            vol = max(volume[i], 1.0)
            cum_tp_vol += typical[i] * vol
            cum_vol += vol
            vwap[i] = cum_tp_vol / cum_vol

        atr = self._atr(high, low, close, self.atr_period)

        direction = np.zeros(n, dtype=int)
        for i in range(1, n):
            if (np.isnan(stoch_k[i]) or np.isnan(stoch_k[i - 1])
                    or np.isnan(vwap[i])):
                direction[i] = direction[i - 1]
                continue
            # Bullish: StochRSI crosses up through oversold + above VWAP
            if (stoch_k[i - 1] < self.oversold and stoch_k[i] >= self.oversold
                    and close[i] > vwap[i]):
                direction[i] = 1
            # Bearish: StochRSI crosses down through overbought + below VWAP
            elif (stoch_k[i - 1] > self.overbought and stoch_k[i] <= self.overbought
                  and close[i] < vwap[i]):
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        return {
            'stoch_k': stoch_k, 'vwap': vwap,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        vwap = indicators['vwap'][start:]
        mv = ~np.isnan(vwap)
        if mv.any():
            ax_price.plot(x[mv], vwap[mv], color='#4fc3f7', linewidth=1.2,
                          label='VWAP', zorder=2)
        if ax_sub is None:
            return
        sk = indicators['stoch_k'][start:]
        ms = ~np.isnan(sk)
        if ms.any():
            ax_sub.plot(x[ms], sk[ms], color='#4fc3f7', linewidth=1.0,
                        label='StochRSI %K', zorder=3)
        ax_sub.axhline(y=self.overbought, color=colors['red'],
                       linewidth=0.7, linestyle='--', alpha=0.7)
        ax_sub.axhline(y=self.oversold, color=colors['green'],
                       linewidth=0.7, linestyle='--', alpha=0.7)
        ax_sub.axhline(y=50, color='#555', linewidth=0.5)
        ax_sub.fill_between(x, self.overbought, 100,
                            color=colors['red'], alpha=0.05)
        ax_sub.fill_between(x, 0, self.oversold,
                            color=colors['green'], alpha=0.05)
        ax_sub.set_ylim(0, 100)
        ax_sub.set_title('Stochastic RSI', color='#888', fontsize=8, loc='left')
        ax_sub.legend(loc='upper left', fontsize=7,
                      facecolor='#111', edgecolor='#333', labelcolor='#ccc')


# ---------------------------------------------------------------------------
# Strategy: Volatility Breakout (Connors)
# ---------------------------------------------------------------------------

class VolatilityBreakoutStrategy(BaseStrategy):
    name = "Volatility Breakout"
    description = (
        "BREAKOUT strategy entering when today's range expands beyond the "
        "average range.\n\n"
        "HOW IT WORKS\n"
        "Computes the average true range over N bars. When the current "
        "bar's range exceeds a multiple of the average range, it signals "
        "a volatility expansion. The direction is determined by whether "
        "the breakout bar closes in its upper or lower half.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Bar range > 1.5x avg range AND close in upper half\n"
        "  SHORT : Bar range > 1.5x avg range AND close in lower half\n\n"
        "EXIT RULES\n"
        "  - Volatility contracts (range < average range)\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - After periods of low volatility (squeeze-like)\n"
        "  - Earnings reactions, news events, gap days\n"
        "  - Avoid during steady, low-volume drift\n\n"
        "PARAMETERS\n"
        "  ATR Period    : 14 (for average range)\n"
        "  Range Mult    : 1.5 (breakout threshold)\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  2.0x  (breakout — captures the volatility expansion move)\n\n"
        "CHART INDICATORS\n"
        "  - Sub-panel : Current range vs average range threshold"
    )
    needs_subpanel = True

    def __init__(self, atr_period=14, range_mult=1.5):
        self.atr_period = atr_period
        self.range_mult = range_mult
        self.min_bars = atr_period + 10

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        atr = self._atr(high, low, close, self.atr_period)
        bar_range = high - low
        threshold = atr * self.range_mult

        direction = np.zeros(n, dtype=int)
        for i in range(self.atr_period, n):
            if np.isnan(atr[i]) or atr[i] <= 0:
                direction[i] = direction[i - 1]
                continue
            if bar_range[i] > threshold[i]:
                mid = (high[i] + low[i]) / 2.0
                if close[i] >= mid:
                    direction[i] = 1  # bullish expansion
                else:
                    direction[i] = -1  # bearish expansion
            else:
                direction[i] = direction[i - 1]

        return {
            'bar_range': bar_range, 'threshold': threshold,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        if ax_sub is None:
            return
        br = indicators['bar_range'][start:]
        th = indicators['threshold'][start:]
        mt = ~np.isnan(th)
        ax_sub.bar(x, br, color='#4fc3f7', alpha=0.5, width=0.8, label='Range')
        if mt.any():
            ax_sub.plot(x[mt], th[mt], color=colors['red'], linewidth=1.0,
                        linestyle='--', label='Threshold', zorder=3)
        ax_sub.set_title('Volatility Breakout', color='#888', fontsize=8, loc='left')
        ax_sub.legend(loc='upper left', fontsize=7,
                      facecolor='#111', edgecolor='#333', labelcolor='#ccc')


# ---------------------------------------------------------------------------
# Strategy: ATR Channel Breakout
# ---------------------------------------------------------------------------

class ATRChannelBreakoutStrategy(BaseStrategy):
    name = "ATR Channel"
    description = (
        "BREAKOUT strategy with dynamic volatility-adaptive channels.\n\n"
        "HOW IT WORKS\n"
        "Draws upper and lower channels at N x ATR above and below an EMA. "
        "Unlike fixed Donchian channels, ATR channels automatically widen "
        "in volatile markets and narrow in calm markets. Entry occurs on "
        "a channel breakout.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Close breaks above EMA + 2.5x ATR\n"
        "  SHORT : Close breaks below EMA - 2.5x ATR\n\n"
        "EXIT RULES\n"
        "  - Price returns inside the channel\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Volatile stocks with strong breakouts\n"
        "  - Adapts automatically to changing volatility\n"
        "  - Wider channels = fewer but more reliable signals\n\n"
        "PARAMETERS\n"
        "  EMA Period    : 20\n"
        "  Channel Mult  : 2.5 (ATR multiplier for channel width)\n"
        "  ATR Period    : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  2.5x  (breakout — wide stop for volatility-adaptive channels)\n\n"
        "CHART INDICATORS\n"
        "  - Yellow line  : 20 EMA (midline)\n"
        "  - Green dashed : Upper channel\n"
        "  - Red dashed   : Lower channel\n"
        "  - Blue shading : Channel range"
    )

    def __init__(self, ema_period=20, channel_mult=2.5, atr_period=14):
        self.ema_period = ema_period
        self.channel_mult = channel_mult
        self.atr_period = atr_period
        self.min_bars = max(ema_period, atr_period) + 10

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        ema = self._ema(close, self.ema_period)
        atr = self._atr(high, low, close, self.atr_period)
        upper = ema + self.channel_mult * atr
        lower = ema - self.channel_mult * atr

        direction = np.zeros(n, dtype=int)
        for i in range(1, n):
            if np.isnan(upper[i]) or np.isnan(lower[i]):
                direction[i] = direction[i - 1]
                continue
            if close[i] > upper[i]:
                direction[i] = 1
            elif close[i] < lower[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        return {
            'ema': ema, 'upper': upper, 'lower': lower,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        ema = indicators['ema'][start:]
        upper = indicators['upper'][start:]
        lower = indicators['lower'][start:]
        m = ~np.isnan(ema)
        if m.any():
            ax_price.plot(x[m], ema[m], color=colors['yellow'],
                          linewidth=1.0, label='EMA', zorder=2)
            ax_price.plot(x[m], upper[m], color=colors['green'],
                          linewidth=0.8, linestyle='--', label='Upper', zorder=2)
            ax_price.plot(x[m], lower[m], color=colors['red'],
                          linewidth=0.8, linestyle='--', label='Lower', zorder=2)
            ax_price.fill_between(x[m], upper[m], lower[m],
                                  color='#4fc3f7', alpha=0.05, zorder=1)


# ---------------------------------------------------------------------------
# Strategy: Pivot Point Bounce
# ---------------------------------------------------------------------------

class PivotPointBounceStrategy(BaseStrategy):
    name = "Pivot Point Bounce"
    description = (
        "MEAN-REVERSION strategy trading bounces off daily pivot levels.\n\n"
        "HOW IT WORKS\n"
        "Computes classic floor trader pivot points from the prior day's "
        "high, low, close: Pivot = (H+L+C)/3, then R1/R2 resistance and "
        "S1/S2 support levels. When price touches S1/S2 a long bounce is "
        "signalled; when price touches R1/R2 a short reversal is signalled.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Close drops to or below S1 support\n"
        "  SHORT : Close rises to or above R1 resistance\n\n"
        "EXIT RULES\n"
        "  - Price returns to the pivot (mean)\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Intraday trading (pivots reset daily)\n"
        "  - Range-bound days where price respects levels\n"
        "  - Avoid on gap days or strong trend days\n\n"
        "PARAMETERS\n"
        "  ATR Period : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  1.0x  (mean-reversion — tight stop; strategy exits at pivot,\n"
        "         or use 0 to rely on pivot levels only)\n\n"
        "CHART INDICATORS\n"
        "  - Yellow dashed : Pivot (P)\n"
        "  - Green dashed  : Support S1, S2\n"
        "  - Red dashed    : Resistance R1, R2"
    )

    def __init__(self, atr_period=14):
        self.atr_period = atr_period
        self.min_bars = 60

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < 2:
            return self._empty(n)

        _ET = ZoneInfo('America/New_York')
        atr = self._atr(high, low, close, self.atr_period)

        # Parse dates
        bar_dates = []
        for i in range(n):
            try:
                dt = bars[i].date
                if hasattr(dt, 'astimezone'):
                    dt = dt.astimezone(_ET)
                bar_dates.append(dt.date() if hasattr(dt, 'date') else None)
            except Exception:
                bar_dates.append(None)

        # Build daily H/L/C
        day_data = {}
        for i in range(n):
            d = bar_dates[i]
            if d is None:
                continue
            if d not in day_data:
                day_data[d] = {'high': high[i], 'low': low[i], 'close': close[i]}
            else:
                day_data[d]['high'] = max(day_data[d]['high'], high[i])
                day_data[d]['low'] = min(day_data[d]['low'], low[i])
                day_data[d]['close'] = close[i]

        sorted_days = sorted(day_data.keys())

        # Compute pivot levels per day (from PRIOR day)
        pivot = np.full(n, np.nan)
        r1 = np.full(n, np.nan)
        r2 = np.full(n, np.nan)
        s1 = np.full(n, np.nan)
        s2 = np.full(n, np.nan)

        for i in range(n):
            d = bar_dates[i]
            if d is None:
                continue
            idx = sorted_days.index(d) if d in sorted_days else -1
            if idx <= 0:
                continue
            prev_day = sorted_days[idx - 1]
            pd = day_data[prev_day]
            pp = (pd['high'] + pd['low'] + pd['close']) / 3.0
            pivot[i] = pp
            r1[i] = 2 * pp - pd['low']
            s1[i] = 2 * pp - pd['high']
            r2[i] = pp + (pd['high'] - pd['low'])
            s2[i] = pp - (pd['high'] - pd['low'])

        direction = np.zeros(n, dtype=int)
        for i in range(1, n):
            if np.isnan(pivot[i]) or np.isnan(s1[i]) or np.isnan(r1[i]):
                direction[i] = direction[i - 1]
                continue
            if close[i] <= s1[i]:
                direction[i] = 1  # bounce long off support
            elif close[i] >= r1[i]:
                direction[i] = -1  # reject short off resistance
            elif direction[i - 1] == 1 and close[i] >= pivot[i]:
                direction[i] = 0  # exit at pivot
            elif direction[i - 1] == -1 and close[i] <= pivot[i]:
                direction[i] = 0  # exit at pivot
            else:
                direction[i] = direction[i - 1]

        return {
            'pivot': pivot, 'r1': r1, 'r2': r2, 's1': s1, 's2': s2,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        pivot = indicators['pivot'][start:]
        r1 = indicators['r1'][start:]
        r2 = indicators['r2'][start:]
        s1 = indicators['s1'][start:]
        s2 = indicators['s2'][start:]
        mp = ~np.isnan(pivot)
        if mp.any():
            ax_price.plot(x[mp], pivot[mp], color=colors['yellow'],
                          linewidth=0.8, linestyle='--', label='Pivot', zorder=2)
            ax_price.plot(x[mp], r1[mp], color=colors['red'],
                          linewidth=0.6, linestyle=':', label='R1', zorder=2)
            ax_price.plot(x[mp], r2[mp], color=colors['red'],
                          linewidth=0.6, linestyle=':', alpha=0.5, label='R2', zorder=2)
            ax_price.plot(x[mp], s1[mp], color=colors['green'],
                          linewidth=0.6, linestyle=':', label='S1', zorder=2)
            ax_price.plot(x[mp], s2[mp], color=colors['green'],
                          linewidth=0.6, linestyle=':', alpha=0.5, label='S2', zorder=2)


# ---------------------------------------------------------------------------
# Strategy: Inside Bar Breakout
# ---------------------------------------------------------------------------

class InsideBarBreakoutStrategy(BaseStrategy):
    name = "Inside Bar Breakout"
    description = (
        "BREAKOUT strategy detecting inside bars (consolidation) then "
        "trading the breakout.\n\n"
        "HOW IT WORKS\n"
        "An 'inside bar' occurs when the current bar's high is below the "
        "prior bar's high AND the current low is above the prior low. This "
        "means price is consolidating. When the next bar breaks out above "
        "the inside bar's mother bar high, go long. Below the mother bar "
        "low, go short.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Close breaks above the mother bar high\n"
        "  SHORT : Close breaks below the mother bar low\n\n"
        "EXIT RULES\n"
        "  - Opposite breakout occurs\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - After strong moves that pause briefly\n"
        "  - Near key support/resistance levels\n"
        "  - The tighter the inside bar, the bigger the expected move\n"
        "  - Works on any timeframe\n\n"
        "PARAMETERS\n"
        "  ATR Period : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  1.5x  (breakout — moderate stop; inside bar range is tight\n"
        "         so the trail locks in gains quickly)\n\n"
        "CHART INDICATORS\n"
        "  - Purple markers : Inside bars detected\n"
        "  - Green/Red dashed : Mother bar high/low levels"
    )

    def __init__(self, atr_period=14):
        self.atr_period = atr_period
        self.min_bars = 20

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        atr = self._atr(high, low, close, self.atr_period)

        # Detect inside bars
        is_inside = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if high[i] <= high[i - 1] and low[i] >= low[i - 1]:
                is_inside[i] = True

        # Track mother bar levels and breakout direction
        mother_high = np.full(n, np.nan)
        mother_low = np.full(n, np.nan)
        direction = np.zeros(n, dtype=int)
        active_mh = np.nan
        active_ml = np.nan

        for i in range(2, n):
            if is_inside[i - 1]:
                # Previous bar was inside; its mother is i-2
                active_mh = high[i - 2]
                active_ml = low[i - 2]

            if not np.isnan(active_mh):
                mother_high[i] = active_mh
                mother_low[i] = active_ml
                if close[i] > active_mh:
                    direction[i] = 1
                    active_mh = np.nan
                    active_ml = np.nan
                elif close[i] < active_ml:
                    direction[i] = -1
                    active_mh = np.nan
                    active_ml = np.nan
                else:
                    direction[i] = direction[i - 1]
            else:
                direction[i] = direction[i - 1]

        return {
            'is_inside': is_inside, 'mother_high': mother_high,
            'mother_low': mother_low,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        inside = indicators['is_inside'][start:]
        mh = indicators['mother_high'][start:]
        ml = indicators['mother_low'][start:]
        closes = np.array([0.0])  # placeholder

        # Mark inside bars
        for i in range(len(inside)):
            if inside[i]:
                ax_price.axvline(x=x[i], color='#e040fb', linewidth=0.5,
                                 alpha=0.4, zorder=1)

        # Mother bar levels
        mmh = ~np.isnan(mh)
        mml = ~np.isnan(ml)
        if mmh.any():
            ax_price.plot(x[mmh], mh[mmh], color=colors['green'],
                          linewidth=0.7, linestyle='--', alpha=0.6,
                          label='Mother High', zorder=2)
        if mml.any():
            ax_price.plot(x[mml], ml[mml], color=colors['red'],
                          linewidth=0.7, linestyle='--', alpha=0.6,
                          label='Mother Low', zorder=2)


# ---------------------------------------------------------------------------
# Strategy: Heikin-Ashi Trend
# ---------------------------------------------------------------------------

class HeikinAshiTrendStrategy(BaseStrategy):
    name = "Heikin-Ashi Trend"
    description = (
        "TREND-FOLLOWING strategy using smoothed Heikin-Ashi candles.\n\n"
        "HOW IT WORKS\n"
        "Heikin-Ashi candles are computed from averaged OHLC values, "
        "which filters out noise and makes trends much clearer. A bullish "
        "HA candle has close > open (green), bearish has close < open (red). "
        "Entry occurs on color change (trend reversal). A 50 EMA provides "
        "additional trend confirmation.\n\n"
        "ENTRY RULES\n"
        "  LONG  : HA candle turns bullish (HA close > HA open)\n"
        "          AND price > 50 EMA\n"
        "  SHORT : HA candle turns bearish (HA close < HA open)\n"
        "          AND price < 50 EMA\n\n"
        "EXIT RULES\n"
        "  - HA candle changes color against position\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Trending markets with clean directional moves\n"
        "  - Smooths out whipsaws compared to regular candles\n"
        "  - Very clean visual trend identification\n\n"
        "PARAMETERS\n"
        "  EMA Filter   : 50\n"
        "  ATR Period   : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  2.0x  (trend — lets the smoothed HA candles ride the move;\n"
        "         strategy exits on HA color change + EMA filter)\n\n"
        "CHART INDICATORS\n"
        "  - Yellow line : 50 EMA\n"
        "  - Sub-panel   : Heikin-Ashi candle direction bars"
    )
    needs_subpanel = True

    def __init__(self, ema_period=50, atr_period=14):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.min_bars = ema_period + 5

    def compute(self, bars) -> dict:
        o = np.array([b.open for b in bars], dtype=float)
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        # Heikin-Ashi
        ha_close = (o + high + low + close) / 4.0
        ha_open = np.empty(n)
        ha_open[0] = (o[0] + close[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(low, np.minimum(ha_open, ha_close))

        # HA direction: +1 bullish, -1 bearish
        ha_dir = np.where(ha_close > ha_open, 1, -1)

        ema = self._ema(close, self.ema_period)
        atr = self._atr(high, low, close, self.atr_period)

        # Direction with EMA filter
        direction = np.zeros(n, dtype=int)
        trade_dir = np.zeros(n, dtype=int)
        for i in range(1, n):
            direction[i] = ha_dir[i]
            if np.isnan(ema[i]):
                trade_dir[i] = 0
            elif ha_dir[i] == 1 and close[i] > ema[i]:
                trade_dir[i] = 1
            elif ha_dir[i] == -1 and close[i] < ema[i]:
                trade_dir[i] = -1
            else:
                trade_dir[i] = 0

        return {
            'ha_open': ha_open, 'ha_close': ha_close,
            'ha_high': ha_high, 'ha_low': ha_low,
            'ha_dir': ha_dir, 'ema': ema,
            'direction': direction, 'trade_direction': trade_dir, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        ema = indicators['ema'][start:]
        me = ~np.isnan(ema)
        if me.any():
            ax_price.plot(x[me], ema[me], color=colors['yellow'],
                          linewidth=1.0, label='50 EMA', zorder=2)
        if ax_sub is None:
            return
        ha_dir = indicators['ha_dir'][start:]
        for i in range(len(ha_dir)):
            c = colors['green'] if ha_dir[i] == 1 else colors['red']
            ax_sub.bar(x[i], ha_dir[i], color=c, alpha=0.6, width=0.8)
        ax_sub.axhline(y=0, color='#555', linewidth=0.5)
        ax_sub.set_ylim(-1.5, 1.5)
        ax_sub.set_title('Heikin-Ashi Direction', color='#888',
                         fontsize=8, loc='left')


# ---------------------------------------------------------------------------
# Strategy: Linear Regression Channel
# ---------------------------------------------------------------------------

class LinRegChannelStrategy(BaseStrategy):
    name = "LinReg Channel"
    description = (
        "MEAN-REVERSION strategy using a linear regression channel.\n\n"
        "HOW IT WORKS\n"
        "Fits a linear regression line to the last N closing prices, then "
        "draws channels at +/- 2 standard deviations from the regression "
        "line. When price deviates below the lower channel it signals a "
        "reversion long; above the upper channel signals a reversion short. "
        "More adaptive than Bollinger Bands because it accounts for trend slope.\n\n"
        "ENTRY RULES\n"
        "  LONG  : Close drops below lower regression channel\n"
        "  SHORT : Close rises above upper regression channel\n\n"
        "EXIT RULES\n"
        "  - Price returns to the regression line\n"
        "  - ATR trailing stop hit\n\n"
        "BEST CONDITIONS\n"
        "  - Stocks with a clear trend but occasional overextensions\n"
        "  - Better than Bollinger in trending markets\n"
        "  - Avoid during regime changes or breakouts\n\n"
        "PARAMETERS\n"
        "  Lookback     : 50 (regression window)\n"
        "  Std Mult     : 2.0 (channel width)\n"
        "  ATR Period   : 14\n\n"
        "RECOMMENDED ATR TRAIL\n"
        "  1.0x  (mean-reversion — tight safety net; strategy exits at\n"
        "         regression line, or use 0 to rely on channel only)\n\n"
        "CHART INDICATORS\n"
        "  - Yellow line  : Regression line (best fit)\n"
        "  - Green dashed : Lower channel (-2 std)\n"
        "  - Red dashed   : Upper channel (+2 std)\n"
        "  - Blue shading : Channel range"
    )

    def __init__(self, lookback=50, std_mult=2.0, atr_period=14):
        self.lookback = lookback
        self.std_mult = std_mult
        self.atr_period = atr_period
        self.min_bars = lookback + 10

    def compute(self, bars) -> dict:
        high = np.array([b.high for b in bars], dtype=float)
        low = np.array([b.low for b in bars], dtype=float)
        close = np.array([b.close for b in bars], dtype=float)
        n = len(close)
        if n < self.min_bars:
            return self._empty(n)

        atr = self._atr(high, low, close, self.atr_period)

        reg_line = np.full(n, np.nan)
        reg_upper = np.full(n, np.nan)
        reg_lower = np.full(n, np.nan)
        lb = self.lookback

        x_reg = np.arange(lb, dtype=float)
        for i in range(lb - 1, n):
            window = close[i - lb + 1:i + 1]
            # Linear regression: y = mx + b
            x_mean = (lb - 1) / 2.0
            y_mean = np.mean(window)
            cov = np.sum((x_reg - x_mean) * (window - y_mean))
            var = np.sum((x_reg - x_mean) ** 2)
            if var == 0:
                reg_line[i] = y_mean
                reg_upper[i] = y_mean
                reg_lower[i] = y_mean
                continue
            slope = cov / var
            intercept = y_mean - slope * x_mean
            fitted = slope * x_reg + intercept
            reg_line[i] = fitted[-1]  # value at current bar
            residuals = window - fitted
            std = np.std(residuals, ddof=0)
            reg_upper[i] = reg_line[i] + self.std_mult * std
            reg_lower[i] = reg_line[i] - self.std_mult * std

        direction = np.zeros(n, dtype=int)
        for i in range(1, n):
            if np.isnan(reg_upper[i]) or np.isnan(reg_lower[i]):
                direction[i] = direction[i - 1]
                continue
            if close[i] <= reg_lower[i]:
                direction[i] = 1
            elif close[i] >= reg_upper[i]:
                direction[i] = -1
            elif direction[i - 1] == 1 and close[i] >= reg_line[i]:
                direction[i] = 0
            elif direction[i - 1] == -1 and close[i] <= reg_line[i]:
                direction[i] = 0
            else:
                direction[i] = direction[i - 1]

        return {
            'reg_line': reg_line, 'reg_upper': reg_upper,
            'reg_lower': reg_lower,
            'direction': direction, 'atr': atr,
        }

    def plot_indicators(self, ax_price, ax_sub, indicators, x, start, colors):
        reg = indicators['reg_line'][start:]
        upper = indicators['reg_upper'][start:]
        lower = indicators['reg_lower'][start:]
        m = ~np.isnan(reg)
        if m.any():
            ax_price.plot(x[m], reg[m], color=colors['yellow'],
                          linewidth=1.0, label='Regression', zorder=2)
            ax_price.plot(x[m], upper[m], color=colors['red'],
                          linewidth=0.8, linestyle='--', label='Upper', zorder=2)
            ax_price.plot(x[m], lower[m], color=colors['green'],
                          linewidth=0.8, linestyle='--', label='Lower', zorder=2)
            ax_price.fill_between(x[m], upper[m], lower[m],
                                  color='#4fc3f7', alpha=0.05, zorder=1)


# ---------------------------------------------------------------------------
# Strategy Registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    'orb': OpeningRangeBreakoutStrategy,
    'squeeze': SqueezeMomentumStrategy,
    'vwap': VWAPReversionStrategy,
    'supertrend': SupertrendStrategy,
    'macd': MACDCrossoverStrategy,
    'rsi': RSIOverboughtOversoldStrategy,
    'dual_ma': DualMovingAverageStrategy,
    'donchian': DonchianBreakoutStrategy,
    'bollinger': BollingerBounceStrategy,
    'atr_trail': ATRTrailingStopStrategy,
    'obv': OBVDivergenceStrategy,
    'ichimoku': IchimokuCloudStrategy,
    'keltner_adx': KeltnerADXStrategy,
    'zscore': ZScoreMeanReversionStrategy,
    'roc_div': ROCDivergenceStrategy,
    'triple_screen': TripleScreenStrategy,
    'stoch_rsi_vwap': StochRSIVWAPStrategy,
    'vol_breakout': VolatilityBreakoutStrategy,
    'atr_channel': ATRChannelBreakoutStrategy,
    'pivot_bounce': PivotPointBounceStrategy,
    'inside_bar': InsideBarBreakoutStrategy,
    'heikin_ashi': HeikinAshiTrendStrategy,
    'linreg': LinRegChannelStrategy,
}
STRATEGY_NAMES = {cls.name: key for key, cls in STRATEGY_REGISTRY.items()}


# ---------------------------------------------------------------------------
# Trading Engine
# ---------------------------------------------------------------------------

class TradingEngine:
    """Manages IB connection, data subscriptions, and order execution."""

    IDLE = 'IDLE'
    WATCHING = 'WATCHING'   # connected + streaming data, no trading
    RUNNING = 'RUNNING'
    PAUSED = 'PAUSED'
    _ET = ZoneInfo('America/New_York')

    def __init__(self, on_bar_update=None, on_log=None, strategy=None):
        self.ib = IB()
        self.contract = None
        self.bars = None
        self._ticker = None  # real-time market data ticker
        self.position = 0  # tracked locally, verified against IB
        self.quantity = 100
        self.state = self.IDLE
        self.strategy = strategy or OpeningRangeBreakoutStrategy()
        # ATR trailing stop
        self.trail_mult = 0.0       # multiplier for ATR trailing stop
        self.trail_active = False    # is trailing stop tracking?
        self.trail_high = 0.0       # high-water mark (longs)
        self.trail_low = float('inf')  # low-water mark (shorts)
        self.trail_stop = 0.0       # current stop price

        # Profit target
        self.profit_mult = 0.0      # multiplier for profit target (0 = disabled)
        self.profit_target = 0.0    # active profit target price (0 = inactive)

        # Trade tracking for win rate
        self.entry_price = 0.0      # price at which current position was opened
        self.entry_side = ''        # 'long' or 'short'
        self.wins = 0
        self.losses = 0
        self.trade_history = []
        self.live_markers = []      # live entry/exit markers for chart arrows

        self._on_bar_update = on_bar_update or (lambda *a: None)
        self._on_log = on_log or (lambda m: None)
        self._on_trade_update = lambda: None  # GUI callback for win rate

        self._load_trade_history()

    # -- trade history persistence --------------------------------------------

    def _load_trade_history(self):
        """Restore trade history from disk."""
        try:
            if TRADE_HISTORY_FILE.exists():
                with open(TRADE_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if not isinstance(loaded, list):
                    raise ValueError("trade history must be a list")

                clean_history = []
                for trade in loaded:
                    if not isinstance(trade, dict):
                        continue
                    if not {'entry_price', 'exit_price', 'side', 'pnl', 'qty', 'result', 'exit_type', 'timestamp'} <= set(trade):
                        continue
                    clean_history.append({
                        'entry_price': float(trade['entry_price']),
                        'exit_price': float(trade['exit_price']),
                        'side': str(trade['side']),
                        'pnl': float(trade['pnl']),
                        'qty': int(trade['qty']),
                        'result': str(trade['result']),
                        'exit_type': str(trade['exit_type']),
                        'timestamp': str(trade['timestamp']),
                    })

                self.trade_history = clean_history[-MAX_TRADE_HISTORY_ENTRIES:]
                self.wins = sum(1 for t in self.trade_history if t['result'] == 'WIN')
                self.losses = sum(1 for t in self.trade_history if t['result'] == 'LOSS')
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            self.trade_history = []
            self.wins = 0
            self.losses = 0

    def _save_trade_history(self):
        """Persist trade history to disk."""
        tmp_path = None
        try:
            TRADE_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix='trade_history_',
                suffix='.json',
                dir=str(TRADE_HISTORY_FILE.parent)
            )
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self.trade_history[-MAX_TRADE_HISTORY_ENTRIES:], f, indent=2)
            os.replace(tmp_path, TRADE_HISTORY_FILE)
        except OSError:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # -- strategy switching --------------------------------------------------

    def set_strategy(self, strategy_key: str):
        """Swap active strategy. Only when IDLE or WATCHING."""
        if self.state in (self.RUNNING, self.PAUSED):
            self._on_log("Stop trading before changing strategy.")
            return False
        cls = STRATEGY_REGISTRY.get(strategy_key)
        if cls is None:
            self._on_log(f"Unknown strategy: {strategy_key}")
            return False
        old_bar_size = self.strategy.bar_size
        self.strategy = cls()
        self._on_log(f"Strategy → {self.strategy.name}")
        # Re-subscribe if bar size changed and we're connected
        if (self.strategy.bar_size != old_bar_size and
                self.contract is not None and self.ib.isConnected()):
            self.subscribe(self.contract.symbol)
        elif self.bars and len(self.bars) > 0:
            indicators = self.strategy.compute(self.bars)
            self._on_bar_update(self.bars, indicators)
        return True

    # -- connection ----------------------------------------------------------

    def connect(self, host='127.0.0.1', port=7497, client_id=None) -> bool:
        if self.ib.isConnected():
            return True
        if client_id is None:
            client_id = random.randint(10, 999)
        try:
            self.ib.connect(host, port, clientId=client_id, timeout=0)
            self.ib.disconnectedEvent += self._handle_disconnect
            self.ib.errorEvent += self._handle_error
            self._on_log(f"Connected to TWS  {host}:{port}")
            return True
        except ConnectionRefusedError:
            self._on_log(
                "Connection refused — is TWS running with API enabled "
                "on port 7497?")
            return False
        except Exception as exc:
            self._on_log(f"Connection failed: {exc}")
            return False

    def disconnect(self):
        if self.ib.isConnected():
            try:
                self.ib.disconnect()
            except Exception:
                pass

    # -- subscribe (data only, no trading) -----------------------------------

    def subscribe(self, symbol: str) -> bool:
        """Connect to TWS and start streaming bars (chart only, no trades)."""
        symbol = symbol.strip().upper()
        if not symbol:
            self._on_log("Enter a ticker symbol.")
            return False

        if not self.ib.isConnected():
            if not self.connect():
                return False

        # Cancel previous subscriptions if any
        if self._ticker is not None:
            try:
                self.ib.cancelMktData(self.contract)
            except Exception:
                pass
            self._ticker = None
        if self.bars is not None:
            try:
                self.ib.cancelHistoricalData(self.bars)
            except Exception:
                pass

        self.contract = Stock(symbol, 'SMART', 'USD')
        qualified = self.ib.qualifyContracts(self.contract)
        if not qualified:
            self._on_log(f"Could not qualify contract for {symbol}")
            return False

        bs = self.strategy.bar_size
        dur = self.strategy.duration
        self._on_log(f"Subscribing to {symbol} {bs} bars ...")
        self.bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime='',
            durationStr=dur,
            barSizeSetting=bs,
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
            keepUpToDate=True,
        )
        self.bars.updateEvent += self._on_bars_updated
        # Subscribe to real-time tick data for instant price updates
        self._ticker = self.ib.reqMktData(self.contract, '233', False, False)
        if self.state == self.IDLE:
            self.state = self.WATCHING
        self._on_log(f"Streaming {symbol} {bs}  (loaded {len(self.bars)} bars)")

        # Render chart immediately with whatever data we have
        self._initial_chart_render()
        return True

    def _initial_chart_render(self):
        """Draw chart right after subscribe, even if not enough bars for EMA."""
        if not self.bars or len(self.bars) == 0:
            return
        indicators = self.strategy.compute(self.bars)
        self._on_bar_update(self.bars, indicators)

    # -- historical fetch (for backtesting) ------------------------------------

    def fetch_history(self, symbol: str, duration: str,
                      bar_size: str) -> list:
        """Fetch a one-shot block of historical bars (not streaming).
        Returns list of BarData or empty list on failure."""
        symbol = symbol.strip().upper()
        if not symbol:
            self._on_log("Enter a ticker symbol.")
            return []

        if not self.ib.isConnected():
            if not self.connect():
                return []

        contract = Stock(symbol, 'SMART', 'USD')
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            self._on_log(f"Could not qualify contract for {symbol}")
            return []

        self._on_log(f"Fetching {symbol} {bar_size} bars "
                     f"({duration}) for backtest ...")
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
            )
        except Exception as exc:
            self._on_log(f"Historical fetch failed: {exc}")
            return []

        self._on_log(f"Fetched {len(bars)} bars for backtest.")
        return bars

    # -- start / pause / stop ------------------------------------------------

    def start(self, symbol: str):
        symbol = symbol.strip().upper()
        if not symbol:
            self._on_log("Enter a ticker symbol.")
            return False

        if not self.ib.isConnected():
            if not self.connect():
                return False

        if self.state == self.PAUSED:
            self.state = self.RUNNING
            self._on_log("Resumed.")
            return True

        # If already watching the same symbol, just upgrade to RUNNING
        if (self.state == self.WATCHING and self.contract and
                self.contract.symbol == symbol and self.bars is not None):
            self.position = self._get_ib_position()
            self.state = self.RUNNING
            self._on_log(f"Strategy RUNNING on {symbol}  "
                         f"(position={self.position})")
            self._evaluate_immediate_entry()
            return True

        # Otherwise do a fresh subscribe + run
        if not self.subscribe(symbol):
            return False

        self.position = self._get_ib_position()
        self.state = self.RUNNING
        self._on_log(f"Strategy RUNNING on {symbol}  "
                     f"(loaded {len(self.bars)} bars, "
                     f"position={self.position})")
        self._evaluate_immediate_entry()
        return True

    def pause(self):
        if self.state == self.RUNNING:
            self.state = self.PAUSED
            self._on_log("Strategy PAUSED — holding current position.")

    def stop(self):
        """Cancel all orders, liquidate position, keep data streaming."""
        was_running = self.state in (self.RUNNING, self.PAUSED)
        self._on_log("STOP — cancelling orders and liquidating ...")
        self._clear_trail()

        # Cancel all open orders
        try:
            self.ib.reqGlobalCancel()
        except Exception:
            pass

        # Liquidate
        if was_running:
            self._liquidate()

        # Go back to WATCHING (keep chart alive) instead of IDLE
        if self.bars is not None and self.ib.isConnected():
            self.state = self.WATCHING
            self._on_log("Stopped trading. Chart still streaming.")
        else:
            self.state = self.IDLE
            self._on_log("Stopped.")

    # -- internal callbacks --------------------------------------------------

    def _on_bars_updated(self, bars, has_new_bar):
        if not bars or len(bars) == 0:
            return

        indicators = self.strategy.compute(bars)
        self._on_bar_update(bars, indicators)

        # Only evaluate signals when enough data for all indicators
        min_bars = self.strategy.min_bars
        if (len(bars) >= min_bars and has_new_bar
                and self.state == self.RUNNING):
            self._evaluate_signal(bars, indicators)

    def _to_et(self, dt):
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self._ET)
        return dt.astimezone(self._ET)

    def _bar_key(self, dt):
        et = self._to_et(dt)
        if et is None:
            return None
        return et.replace(second=0, microsecond=0)

    def _entry_allowed(self, bar_time):
        et = self._to_et(bar_time)
        if et is None:
            return True
        # Regular trading hours only
        if et.weekday() >= 5:
            return False
        if et.hour < 9 or (et.hour == 9 and et.minute < 30) or et.hour >= 16:
            return False
        return True

    def _evaluate_immediate_entry(self):
        """On Start, enter immediately if conditions are already met."""
        if not self.bars or self.position != 0:
            return
        min_bars = self.strategy.min_bars
        if len(self.bars) < min_bars:
            self._on_log("Waiting for enough bars to evaluate ...")
            return

        bar_time = self.bars[-1].date
        if not self._entry_allowed(bar_time):
            self._on_log("Market closed — waiting for next signal during RTH.")
            return

        indicators = self.strategy.compute(self.bars)
        trade_dir = indicators.get('trade_direction', indicators['direction'])
        close_price = self.bars[-1].close
        curr_dir = trade_dir[-1]

        if curr_dir > 0:
            self._on_log("Conditions already bullish — entering long now.")
            self._handle_signal('BUY', close_price, bar_time)
        elif curr_dir < 0:
            self._on_log("Conditions already bearish — entering short now.")
            self._handle_signal('SELL', close_price, bar_time)
        else:
            self._on_log("No entry conditions met — waiting for next signal.")

    def _evaluate_signal(self, bars, indicators):
        trade_dir = indicators.get('trade_direction', indicators['direction'])
        atr_arr = indicators['atr']
        close_price = bars[-1].close
        bar_time = bars[-1].date

        if len(trade_dir) < 2:
            return

        prev_dir = trade_dir[-2]
        curr_dir = trade_dir[-1]
        curr_atr = atr_arr[-1] if not np.isnan(atr_arr[-1]) else 0.0

        # --- ATR trailing stop check (first priority) ---
        trail_signal = self._update_trail(close_price, curr_atr)
        if trail_signal:
            self._handle_signal(trail_signal, close_price, bar_time)
            return

        # --- Profit target check (second priority) ---
        if self.profit_target > 0 and self.position != 0:
            if self.position > 0 and close_price >= self.profit_target:
                self._on_log(f"Profit target hit @ {self.profit_target:.2f}")
                self._handle_signal('PT_EXIT_LONG', close_price, bar_time)
                return
            elif self.position < 0 and close_price <= self.profit_target:
                self._on_log(f"Profit target hit @ {self.profit_target:.2f}")
                self._handle_signal('PT_EXIT_SHORT', close_price, bar_time)
                return

        ref_dir = prev_dir

        # --- Entry signals (direction flip) ---
        if ref_dir <= 0 and curr_dir > 0:
            if self._entry_allowed(bar_time):
                self._handle_signal('BUY', close_price, bar_time)
                # Set profit target from strategy if available
                self._set_profit_target(indicators)
            else:
                self._on_log("Entry blocked during market closed.")
        elif ref_dir >= 0 and curr_dir < 0:
            if self._entry_allowed(bar_time):
                self._handle_signal('SELL', close_price, bar_time)
                self._set_profit_target(indicators)
            else:
                self._on_log("Entry blocked during market closed.")
        # --- Exit on direction reversal ---
        elif self.position > 0 and curr_dir < 0:
            self._handle_signal('EXIT_LONG', close_price, bar_time)
        elif self.position < 0 and curr_dir > 0:
            self._handle_signal('EXIT_SHORT', close_price, bar_time)

    def _set_profit_target(self, indicators):
        """Set engine profit target from strategy indicators if available."""
        if self.profit_mult <= 0:
            self.profit_target = 0.0
            return
        pt_arr = indicators.get('profit_target')
        if pt_arr is not None:
            pt = pt_arr[-1]
            if not np.isnan(pt):
                self.profit_target = pt
                self._on_log(f"Profit target set @ {pt:.2f}")
                return
        self.profit_target = 0.0

    def _handle_signal(self, signal: str, price: float, bar_time=None):
        self._on_log(f"SIGNAL  {signal}  @ {price:.2f}")

        # Record marker for chart arrows
        self.live_markers.append({
            'bar_time': bar_time, 'price': price, 'signal': signal,
        })

        try:
            if signal == 'BUY':
                if self.position < 0:
                    # Closing short — score it
                    self._score_trade(price)
                    self._place_order('BUY', abs(self.position),
                                      tag='flatten short')
                self._place_order('BUY', self.quantity,
                                  tag='entry long')
                self.entry_price = price
                self.entry_side = 'long'
                self._reset_trail(price, side='long')
            elif signal == 'SELL':
                if self.position > 0:
                    # Closing long — score it
                    self._score_trade(price)
                    self._place_order('SELL', self.position,
                                      tag='flatten long')
                self._place_order('SELL', self.quantity,
                                  tag='entry short')
                self.entry_price = price
                self.entry_side = 'short'
                self._reset_trail(price, side='short')
            elif signal in ('EXIT_LONG', 'TRAIL_EXIT_LONG',
                            'PT_EXIT_LONG') and self.position > 0:
                self._score_trade(price)
                tag = 'profit target' if signal == 'PT_EXIT_LONG' else 'exit long'
                self._place_order('SELL', self.position, tag=tag)
                self._clear_trail()
            elif signal in ('EXIT_SHORT', 'TRAIL_EXIT_SHORT',
                            'PT_EXIT_SHORT') and self.position < 0:
                self._score_trade(price)
                tag = 'profit target' if signal == 'PT_EXIT_SHORT' else 'exit short'
                self._place_order('BUY', abs(self.position), tag=tag)
                self._clear_trail()
        finally:
            pass

    def _score_trade(self, exit_price: float, exit_type: str = 'signal'):
        """Record win or loss when closing a position."""
        if self.entry_price <= 0:
            return
        if self.entry_side == 'long':
            pnl = exit_price - self.entry_price
        elif self.entry_side == 'short':
            pnl = self.entry_price - exit_price
        else:
            return
        result = 'WIN' if pnl > 0 else 'LOSS'
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.trade_history.append({
            'entry_price': round(self.entry_price, 2),
            'exit_price': round(exit_price, 2),
            'side': self.entry_side,
            'pnl': round(pnl, 2),
            'qty': self.quantity,
            'result': result,
            'exit_type': exit_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
        self._save_trade_history()
        total = self.wins + self.losses
        wr = self.wins / total * 100
        self._on_log(f"Trade closed: {result}  "
                     f"({self.wins}W / {self.losses}L  "
                     f"WR={wr:.0f}%)")
        self.entry_price = 0.0
        self.entry_side = ''
        self._on_trade_update()

    # -- ATR trailing stop ---------------------------------------------------

    def _reset_trail(self, entry_price: float, side: str):
        """Start tracking trailing stop after a new entry."""
        self.trail_active = True
        if side == 'long':
            self.trail_high = entry_price
            self.trail_low = float('inf')
        else:
            self.trail_low = entry_price
            self.trail_high = 0.0
        self.trail_stop = 0.0

    def _clear_trail(self):
        """Reset trailing stop and profit target after exit."""
        self.trail_active = False
        self.trail_high = 0.0
        self.trail_low = float('inf')
        self.trail_stop = 0.0
        self.profit_target = 0.0

    def _update_trail(self, close_price: float, atr: float):
        """Update trailing stop level. Returns 'TRAIL_EXIT_LONG',
        'TRAIL_EXIT_SHORT', or None."""
        if not self.trail_active or np.isnan(atr) or self.trail_mult <= 0:
            return None

        trail_dist = self.trail_mult * atr

        if self.position > 0:
            # Long: track highest close, stop below
            if close_price > self.trail_high:
                self.trail_high = close_price
            self.trail_stop = self.trail_high - trail_dist
            if close_price <= self.trail_stop:
                self._on_log(
                    f"TRAIL STOP hit  (stop={self.trail_stop:.2f}, "
                    f"price={close_price:.2f})")
                return 'TRAIL_EXIT_LONG'
        elif self.position < 0:
            # Short: track lowest close, stop above
            if close_price < self.trail_low:
                self.trail_low = close_price
            self.trail_stop = self.trail_low + trail_dist
            if close_price >= self.trail_stop:
                self._on_log(
                    f"TRAIL STOP hit  (stop={self.trail_stop:.2f}, "
                    f"price={close_price:.2f})")
                return 'TRAIL_EXIT_SHORT'

        return None

    def _place_order(self, action: str, qty: int, tag: str = ''):
        if qty <= 0 or self.contract is None:
            return
        order = MarketOrder(action, qty)
        order.tif = 'DAY'
        trade = self.ib.placeOrder(self.contract, order)
        self._on_log(f"ORDER  {action} {qty}  ({tag})")

        def _on_fill(trade):
            self.position = self._get_ib_position()
            fill = trade.fills[-1]
            self._on_log(
                f"FILLED  {fill.execution.side} {fill.execution.shares} "
                f"@ {fill.execution.avgPrice:.2f}  "
                f"(pos={self.position})")

        trade.filledEvent += _on_fill

        def _on_cancel(trade):
            self._on_log(f"ORDER CANCELLED  {action} {qty}")

        trade.cancelledEvent += _on_cancel

    def _liquidate(self):
        """Flatten any position in the active contract."""
        pos = self._get_ib_position()
        if pos == 0:
            self._on_log("No position to liquidate.")
            return
        # Score the trade before flattening
        if self.entry_price > 0 and self.bars:
            self._score_trade(self.bars[-1].close, exit_type='liquidate')
        action = 'SELL' if pos > 0 else 'BUY'
        qty = abs(pos)
        order = MarketOrder(action, qty)
        order.tif = 'DAY'
        self._on_log(f"LIQUIDATE  {action} {qty}")
        self.ib.placeOrder(self.contract, order)
        self.position = 0

    def _get_ib_position(self) -> int:
        """Query IB for current position in the active contract."""
        if self.contract is None:
            return 0
        for p in self.ib.positions():
            if (p.contract.symbol == self.contract.symbol and
                    p.contract.secType == self.contract.secType):
                return int(p.position)
        return 0

    # -- error / disconnect --------------------------------------------------

    def _handle_disconnect(self):
        self._on_log("DISCONNECTED from TWS.")
        self.state = self.IDLE
        self.bars = None

    def _handle_error(self, reqId, errorCode, errorString, contract):
        # Informational / noisy codes — suppress
        ignore = {2104, 2106, 2108, 2158, 2119, 162}
        if errorCode in ignore:
            return
        self._on_log(f"IB Error [{errorCode}] {errorString}")


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class App(tk.Tk):
    BG = '#1a1a2e'
    BG2 = '#16213e'
    FG = '#e0e0e0'
    ACCENT = '#0f3460'
    GREEN = '#00e676'
    RED = '#ff1744'
    YELLOW = '#ffd600'

    def __init__(self):
        super().__init__()
        self.title("AlgoTrader  |  IBKR Paper Trading")
        self.configure(bg=self.BG)
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.engine = TradingEngine(
            on_bar_update=self._update_chart,
            on_log=self._log,
        )
        self.engine._on_trade_update = self._refresh_winrate
        self._bt_running = False  # guard against overlapping backtests
        self._tws_time_offset = timedelta(0)  # synced on connect

        self._build_gui()
        self._set_button_states()
        self._pump_asyncio()

        # Show persisted win rate on startup
        self.after(300, self._refresh_winrate)

        # Auto-connect and start streaming on launch
        self.after(200, self._auto_connect)

    # -- GUI construction ----------------------------------------------------

    def _build_gui(self):
        # --- Top control bar ---
        top = tk.Frame(self, bg=self.BG2, padx=10, pady=8)
        top.pack(fill=tk.X)

        tk.Label(top, text="Ticker:", bg=self.BG2, fg=self.FG,
                 font=("Consolas", 11)).pack(side=tk.LEFT)
        self.ticker_var = tk.StringVar(value="TSLA")
        self.ticker_entry = tk.Entry(
            top, textvariable=self.ticker_var, width=10,
            font=("Consolas", 12), bg='#0a0a1a', fg=self.GREEN,
            insertbackground=self.GREEN, relief=tk.FLAT)
        self.ticker_entry.pack(side=tk.LEFT, padx=(5, 10))
        self.ticker_entry.bind('<Return>', self._on_ticker_submit)

        tk.Label(top, text="Qty:", bg=self.BG2, fg=self.FG,
                 font=("Consolas", 11)).pack(side=tk.LEFT)
        self.qty_var = tk.StringVar(value="100")
        self.qty_entry = tk.Entry(
            top, textvariable=self.qty_var, width=6,
            font=("Consolas", 12), bg='#0a0a1a', fg=self.GREEN,
            insertbackground=self.GREEN, relief=tk.FLAT)
        self.qty_entry.pack(side=tk.LEFT, padx=(5, 10))
        self.qty_entry.bind('<Return>', self._on_qty_change)
        self.qty_entry.bind('<FocusOut>', self._on_qty_change)

        tk.Label(top, text="Trail:", bg=self.BG2, fg=self.FG,
                 font=("Consolas", 11)).pack(side=tk.LEFT)
        self.trail_var = tk.StringVar(value="0.0")
        self.trail_entry = tk.Entry(
            top, textvariable=self.trail_var, width=4,
            font=("Consolas", 12), bg='#0a0a1a', fg=self.GREEN,
            insertbackground=self.GREEN, relief=tk.FLAT)
        self.trail_entry.pack(side=tk.LEFT, padx=(5, 2))
        self.trail_entry.bind('<Return>', self._on_trail_change)
        self.trail_entry.bind('<FocusOut>', self._on_trail_change)
        tk.Label(top, text="xATR", bg=self.BG2, fg='#888',
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(top, text="PT:", bg=self.BG2, fg=self.FG,
                 font=("Consolas", 11)).pack(side=tk.LEFT)
        self.pt_var = tk.StringVar(value="0.0")
        self.pt_entry = tk.Entry(
            top, textvariable=self.pt_var, width=4,
            font=("Consolas", 12), bg='#0a0a1a', fg='#00bcd4',
            insertbackground='#00bcd4', relief=tk.FLAT)
        self.pt_entry.pack(side=tk.LEFT, padx=(5, 2))
        self.pt_entry.bind('<Return>', self._on_pt_change)
        self.pt_entry.bind('<FocusOut>', self._on_pt_change)
        tk.Label(top, text="xR", bg=self.BG2, fg='#888',
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 15))

        btn_cfg = dict(font=("Consolas", 10, "bold"), width=8,
                       relief=tk.FLAT, padx=8, pady=4, cursor="hand2")
        self.start_btn = tk.Button(
            top, text="Start", bg=self.GREEN, fg='#000',
            command=self._on_start, **btn_cfg)
        self.start_btn.pack(side=tk.LEFT, padx=3)

        self.pause_btn = tk.Button(
            top, text="Pause", bg=self.YELLOW, fg='#000',
            command=self._on_pause, **btn_cfg)
        self.pause_btn.pack(side=tk.LEFT, padx=3)

        self.stop_btn = tk.Button(
            top, text="Stop", bg=self.RED, fg='#fff',
            command=self._on_stop, **btn_cfg)
        self.stop_btn.pack(side=tk.LEFT, padx=3)

        self.bt_btn = tk.Button(
            top, text="Backtest", bg='#7c4dff', fg='#fff',
            command=self._on_backtest, **btn_cfg)
        self.bt_btn.pack(side=tk.LEFT, padx=(12, 3))

        self.status_var = tk.StringVar(value="IDLE")
        self.status_label = tk.Label(
            top, textvariable=self.status_var, bg=self.BG2,
            fg=self.YELLOW, font=("Consolas", 11, "bold"))
        self.status_label.pack(side=tk.RIGHT)

        # Live price (right side of top bar)
        self.price_var = tk.StringVar(value="--")
        self.price_label = tk.Label(
            top, textvariable=self.price_var, bg=self.BG2,
            fg='#4fc3f7', font=("Consolas", 12, "bold"))
        self.price_label.pack(side=tk.RIGHT, padx=(0, 15))
        self._prev_price = 0.0

        # --- Strategy selector bar ---
        strat_bar = tk.Frame(self, bg=self.BG2, padx=10, pady=3)
        strat_bar.pack(fill=tk.X)

        tk.Label(strat_bar, text="Strategy:", bg=self.BG2, fg=self.FG,
                 font=("Consolas", 11)).pack(side=tk.LEFT)
        self.strategy_var = tk.StringVar(value=self.engine.strategy.name)
        strategy_names = list(STRATEGY_NAMES.keys())
        self.strategy_menu = tk.OptionMenu(
            strat_bar, self.strategy_var, *strategy_names,
            command=self._on_strategy_change)
        self.strategy_menu.config(
            bg='#0a0a1a', fg=self.GREEN, font=("Consolas", 10),
            activebackground='#333', activeforeground=self.GREEN,
            highlightthickness=0, relief=tk.FLAT)
        self.strategy_menu['menu'].config(
            bg='#16213e', fg='#e0e0e0', activebackground='#7c4dff',
            font=("Consolas", 10))
        self.strategy_menu.pack(side=tk.LEFT, padx=(5, 0))

        self.info_btn = tk.Button(
            strat_bar, text="Info", bg='#0f3460', fg='#4fc3f7',
            font=("Consolas", 9, "bold"), width=4, relief=tk.FLAT,
            cursor="hand2", command=self._show_strategy_info)
        self.info_btn.pack(side=tk.LEFT, padx=(8, 0))

        # --- Account info bar ---
        acct = tk.Frame(self, bg='#0d1b2a', padx=10, pady=5)
        acct.pack(fill=tk.X)

        lbl_cfg = dict(bg='#0d1b2a', font=("Consolas", 10))

        tk.Label(acct, text="Cash:", fg='#888', **lbl_cfg).pack(side=tk.LEFT)
        self.cash_var = tk.StringVar(value="--")
        tk.Label(acct, textvariable=self.cash_var, fg='#4fc3f7',
                 **lbl_cfg).pack(side=tk.LEFT, padx=(2, 15))

        tk.Label(acct, text="Net Liq:", fg='#888', **lbl_cfg).pack(side=tk.LEFT)
        self.netliq_var = tk.StringVar(value="--")
        tk.Label(acct, textvariable=self.netliq_var, fg='#4fc3f7',
                 **lbl_cfg).pack(side=tk.LEFT, padx=(2, 15))

        tk.Label(acct, text="Unrl P&L:", fg='#888', **lbl_cfg).pack(side=tk.LEFT)
        self.upnl_var = tk.StringVar(value="--")
        self.upnl_label = tk.Label(acct, textvariable=self.upnl_var,
                                    fg=self.FG, **lbl_cfg)
        self.upnl_label.pack(side=tk.LEFT, padx=(2, 15))

        tk.Label(acct, text="Day P&L:", fg='#888', **lbl_cfg).pack(side=tk.LEFT)
        self.daypnl_var = tk.StringVar(value="--")
        self.daypnl_label = tk.Label(acct, textvariable=self.daypnl_var,
                                      fg=self.FG, **lbl_cfg)
        self.daypnl_label.pack(side=tk.LEFT, padx=(2, 15))

        tk.Label(acct, text="Pos:", fg='#888', **lbl_cfg).pack(side=tk.LEFT)
        self.pos_var = tk.StringVar(value="0")
        tk.Label(acct, textvariable=self.pos_var, fg=self.YELLOW,
                 **lbl_cfg).pack(side=tk.LEFT, padx=(2, 15))

        tk.Label(acct, text="Trades:", fg='#888', **lbl_cfg).pack(side=tk.LEFT)
        self.trades_var = tk.StringVar(value="0W / 0L")
        tk.Label(acct, textvariable=self.trades_var, fg=self.FG,
                 **lbl_cfg).pack(side=tk.LEFT, padx=(2, 15))

        tk.Label(acct, text="Win Rate:", fg='#888', **lbl_cfg).pack(side=tk.LEFT)
        self.winrate_var = tk.StringVar(value="--")
        self.winrate_label = tk.Label(acct, textvariable=self.winrate_var,
                                       fg=self.FG, **lbl_cfg)
        self.winrate_label.pack(side=tk.LEFT, padx=(2, 0))

        # Live TWS clock (right-aligned)
        self.clock_var = tk.StringVar(value="--:--:--")
        self.clock_label = tk.Label(
            acct, textvariable=self.clock_var, fg='#4fc3f7',
            font=("Consolas", 11, "bold"), bg='#0d1b2a')
        self.clock_label.pack(side=tk.RIGHT, padx=(10, 0))
        tk.Label(acct, text="ET", fg='#888',
                 font=("Consolas", 9), bg='#0d1b2a').pack(side=tk.RIGHT)

        # --- Chart ---
        chart_frame = tk.Frame(self, bg=self.BG)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))

        self.fig = Figure(figsize=(10, 4.5), dpi=100,
                          facecolor=self.BG, edgecolor=self.BG)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Log area ---
        log_frame = tk.Frame(self, bg=self.BG)
        log_frame.pack(fill=tk.X, padx=5, pady=5)

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(
            log_frame, height=8, bg='#0a0a1a', fg=self.FG,
            font=("Consolas", 9), wrap=tk.WORD, state=tk.DISABLED,
            yscrollcommand=scrollbar.set, relief=tk.FLAT,
            insertbackground=self.FG)
        self.log_text.pack(fill=tk.X, expand=True)
        scrollbar.config(command=self.log_text.yview)

        # Log tag colours
        self.log_text.tag_configure('buy', foreground=self.GREEN)
        self.log_text.tag_configure('sell', foreground=self.RED)
        self.log_text.tag_configure('info', foreground=self.YELLOW)

    def _style_axes(self):
        self.ax.set_facecolor('#0a0a1a')
        for spine in self.ax.spines.values():
            spine.set_color('#333')
        self.ax.tick_params(colors='#888', labelsize=8)
        self.ax.grid(True, color='#222', linewidth=0.5)

    # -- Auto-connect -------------------------------------------------------

    def _auto_connect(self):
        """Connect to TWS and start streaming the default ticker on launch."""
        symbol = self.ticker_var.get().strip().upper()
        if symbol:
            self._log("Auto-connecting to TWS ...")
            self.engine.subscribe(symbol)
        self._set_button_states()
        # Sync to TWS server clock (one-shot)
        self._sync_tws_clock()
        # Start real-time account streaming
        self._start_account_stream()

    def _on_ticker_submit(self, _event=None):
        """Re-subscribe when the user changes the ticker and hits Enter."""
        symbol = self.ticker_var.get().strip().upper()
        if not symbol:
            return
        if self.engine.state in (TradingEngine.RUNNING, TradingEngine.PAUSED):
            self._log("Stop trading first before changing ticker.")
            return
        self.engine.subscribe(symbol)
        self._set_button_states()

    def _on_qty_change(self, _event=None):
        """Update engine quantity when user changes the Qty field."""
        raw = self.qty_var.get().strip()
        try:
            qty = int(raw)
            if qty < 1:
                raise ValueError
            self.engine.quantity = qty
        except ValueError:
            self.qty_var.set(str(self.engine.quantity))

    def _on_trail_change(self, _event=None):
        """Update engine trail multiplier when user changes the Trail field."""
        raw = self.trail_var.get().strip()
        try:
            mult = float(raw)
            if mult < 0:
                raise ValueError
            self.engine.trail_mult = mult
        except ValueError:
            self.trail_var.set(str(self.engine.trail_mult))

    def _on_pt_change(self, _event=None):
        """Update engine profit target multiplier when user changes PT field."""
        raw = self.pt_var.get().strip()
        try:
            mult = float(raw)
            if mult < 0:
                raise ValueError
            self.engine.profit_mult = mult
        except ValueError:
            self.pt_var.set(str(self.engine.profit_mult))

    def _on_strategy_change(self, selected_name):
        """Handle strategy dropdown selection."""
        key = STRATEGY_NAMES.get(selected_name)
        if key:
            self.engine.set_strategy(key)

    def _show_strategy_info(self):
        """Show a popup window with detailed info about the current strategy."""
        strat = self.engine.strategy
        win = tk.Toplevel(self)
        win.title(f"Strategy Info  —  {strat.name}")
        win.configure(bg=self.BG)
        win.geometry("620x560")
        win.minsize(500, 400)

        # Header
        header = tk.Frame(win, bg=self.BG2, padx=15, pady=10)
        header.pack(fill=tk.X)
        tk.Label(header, text=strat.name, bg=self.BG2, fg=self.GREEN,
                 font=("Consolas", 14, "bold")).pack(anchor='w')
        meta = (f"Bar Size: {strat.bar_size}    |    "
                f"Min Bars: {strat.min_bars}    |    "
                f"Sub-panel: {'Yes' if strat.needs_subpanel else 'No'}")
        tk.Label(header, text=meta, bg=self.BG2, fg='#888',
                 font=("Consolas", 9)).pack(anchor='w', pady=(4, 0))

        # Description text
        txt_frame = tk.Frame(win, bg=self.BG)
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(txt_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        txt = tk.Text(
            txt_frame, bg='#0a0a1a', fg=self.FG,
            font=("Consolas", 10), wrap=tk.WORD, relief=tk.FLAT,
            yscrollcommand=scrollbar.set, padx=12, pady=10,
            insertbackground=self.FG, state=tk.NORMAL)
        txt.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=txt.yview)

        # Tag styles for section headers
        txt.tag_configure('heading', foreground=self.GREEN,
                          font=("Consolas", 11, "bold"))
        txt.tag_configure('param', foreground='#4fc3f7')
        txt.tag_configure('rule_long', foreground=self.GREEN)
        txt.tag_configure('rule_short', foreground=self.RED)

        desc = getattr(strat, 'description', 'No description available.')
        for line in desc.split('\n'):
            stripped = line.strip()
            if stripped in ('HOW IT WORKS', 'ENTRY RULES', 'EXIT RULES',
                            'BEST CONDITIONS', 'PARAMETERS', 'TIPS',
                            'CHART INDICATORS', 'CHART INDICATORS (sub-panel)'):
                txt.insert(tk.END, '\n' + stripped + '\n', 'heading')
            elif stripped.startswith('LONG'):
                txt.insert(tk.END, line + '\n', 'rule_long')
            elif stripped.startswith('SHORT'):
                txt.insert(tk.END, line + '\n', 'rule_short')
            elif ':' in stripped and any(
                    stripped.startswith(p) for p in (
                        'ATR', 'EMA', 'BB ', 'KC ', 'RSI', 'Fast',
                        'Slow', 'Signal', 'Channel', 'Std', 'OBV',
                        'Range', 'Trend', 'Entry', 'Lookback',
                        'Tenkan', 'Kijun', 'Senkou', 'Displacement',
                        'Overbought', 'Oversold')):
                txt.insert(tk.END, line + '\n', 'param')
            else:
                txt.insert(tk.END, line + '\n')

        txt.config(state=tk.DISABLED)

        # Close button
        btn_frame = tk.Frame(win, bg=self.BG, pady=8)
        btn_frame.pack(fill=tk.X)
        tk.Button(btn_frame, text="Close", bg='#0f3460', fg='#e0e0e0',
                  font=("Consolas", 10, "bold"), width=10, relief=tk.FLAT,
                  cursor="hand2", command=win.destroy).pack()

    def _rebuild_strategy_menu(self):
        """Rebuild the strategy dropdown with current STRATEGY_NAMES."""
        menu = self.strategy_menu['menu']
        menu.delete(0, tk.END)
        for name in STRATEGY_NAMES:
            menu.add_command(
                label=name,
                command=lambda n=name: (self.strategy_var.set(n),
                                        self._on_strategy_change(n)))

    # -- TWS clock sync -------------------------------------------------------

    def _sync_tws_clock(self):
        """Call reqCurrentTime once to compute offset from local clock."""
        ib = self.engine.ib
        if not ib.isConnected():
            return
        try:
            tws_dt = ib.reqCurrentTime()        # UTC-aware datetime
            local_dt = datetime.now(timezone.utc)
            self._tws_time_offset = tws_dt.replace(tzinfo=timezone.utc) - local_dt
            self._log("TWS clock synced  "
                      f"(offset {self._tws_time_offset.total_seconds():+.0f}s)")
        except Exception as exc:
            self._log(f"TWS clock sync failed: {exc}")
        # Start the 1-second clock tick
        self._tick_clock()

    _ET = ZoneInfo('America/New_York')

    def _tws_now(self) -> datetime:
        """Current TWS/exchange time (US-Eastern) without a network call."""
        utc_now = datetime.now(timezone.utc) + self._tws_time_offset
        return utc_now.astimezone(self._ET)

    def _tick_clock(self):
        """Master 1-second loop: clock, balances, position, win rate."""
        now = self._tws_now()
        self.clock_var.set(now.strftime('%I:%M:%S %p').lstrip('0'))

        # Colour: green during market hours, dim when closed
        wd = now.weekday()
        h = now.hour
        if wd < 5 and (h > 9 or (h == 9 and now.minute >= 30)) and h < 16:
            self.clock_label.config(fg=self.GREEN)
        else:
            self.clock_label.config(fg='#888')

        # Refresh account balances, position, and win rate every tick
        try:
            self._refresh_balances()
        except Exception:
            pass
        self._refresh_winrate()

        self.after(1000, self._tick_clock)

    # -- Account streaming & polling -----------------------------------------

    def _start_account_stream(self):
        """Subscribe to real-time PnL streaming and poll slower for balances."""
        ib = self.engine.ib
        if not ib.isConnected():
            return

        # Stream real-time PnL (pushes updates as they happen)
        acct = ib.managedAccounts()[0] if ib.managedAccounts() else ''
        if acct:
            self._pnl_sub = ib.reqPnL(acct)
            ib.pnlEvent += self._on_pnl_update

        # Real-time tick price updates (fires on every tick)
        ib.pendingTickersEvent += self._on_pending_tickers

    def _on_pnl_update(self, pnl):
        """Called by IB in real-time when PnL changes."""
        try:
            if pnl.unrealizedPnL is not None:
                num = pnl.unrealizedPnL
                self.upnl_var.set(f"${num:,.2f}")
                self.upnl_label.config(
                    fg=self.GREEN if num >= 0 else self.RED)
            if pnl.realizedPnL is not None:
                num = pnl.realizedPnL
                self.daypnl_var.set(f"${num:,.2f}")
                self.daypnl_label.config(
                    fg=self.GREEN if num >= 0 else self.RED)
        except Exception:
            pass

    def _on_pending_tickers(self, tickers):
        """Called by IB on every tick — update live price instantly."""
        ticker = self.engine._ticker
        if ticker is None:
            return
        for t in tickers:
            if t.contract == self.engine.contract:
                price = t.marketPrice()
                if price != price:  # nan check
                    price = t.last
                if price != price or price <= 0:
                    return
                self.price_var.set(f"{price:.2f}")
                # Green if up, red if down vs previous tick
                if price > self._prev_price:
                    self.price_label.config(fg=self.GREEN)
                elif price < self._prev_price:
                    self.price_label.config(fg=self.RED)
                self._prev_price = price
                return

    def _refresh_balances(self):
        ib = self.engine.ib
        if not ib.isConnected():
            return

        acct_vals = ib.accountSummary()
        for av in acct_vals:
            tag = av.tag
            val = av.value
            try:
                num = float(val)
                formatted = f"${num:,.2f}"
            except (ValueError, TypeError):
                continue

            if tag == 'CashBalance' or tag == 'TotalCashValue':
                self.cash_var.set(formatted)
            elif tag == 'NetLiquidation':
                self.netliq_var.set(formatted)

        # Current position in active contract
        pos = self.engine._get_ib_position()
        self.pos_var.set(str(pos))

    def _refresh_winrate(self):
        """Update the win rate display from engine trade counters."""
        w = self.engine.wins
        l = self.engine.losses
        total = w + l
        self.trades_var.set(f"{w}W / {l}L")
        if total > 0:
            wr = w / total * 100
            self.winrate_var.set(f"{wr:.0f}%")
            self.winrate_label.config(
                fg=self.GREEN if wr >= 50 else self.RED)
        else:
            self.winrate_var.set("--")
            self.winrate_label.config(fg=self.FG)

    # -- Button handlers -----------------------------------------------------

    def _on_start(self):
        # Sync quantity and trail multiplier from GUI before starting
        self._on_qty_change()
        self._on_trail_change()
        if self.engine.state == TradingEngine.PAUSED:
            self.engine.state = TradingEngine.RUNNING
            self._log("Resumed.")
        else:
            symbol = self.ticker_var.get()
            self.engine.start(symbol)
        self._set_button_states()

    def _on_pause(self):
        self.engine.pause()
        self._set_button_states()

    def _on_stop(self):
        self.engine.stop()
        self._set_button_states()

    def _on_backtest(self):
        """Show a popup menu to choose backtest period."""
        bs = self.engine.strategy.bar_size
        menu = tk.Menu(self, tearoff=0, bg='#16213e', fg='#e0e0e0',
                       activebackground='#7c4dff', activeforeground='#fff',
                       font=("Consolas", 10))
        menu.add_command(label=f"2 Days  ({bs} bars)",
                         command=lambda: self._launch_backtest(
                             '2 D', bs, '2 Day'))
        menu.add_command(label=f"1 Week  ({bs} bars)",
                         command=lambda: self._launch_backtest(
                             '1 W', bs, '1 Week'))
        menu.add_command(label=f"1 Month  ({bs} bars)",
                         command=lambda: self._launch_backtest(
                             '1 M', bs, '1 Month'))

        # Position menu below the Backtest button
        try:
            bx = self.bt_btn.winfo_rootx()
            by = self.bt_btn.winfo_rooty() + self.bt_btn.winfo_height()
            menu.tk_popup(bx, by)
        except Exception:
            menu.tk_popup(
                self.winfo_pointerx(), self.winfo_pointery())

    def _launch_backtest(self, duration: str, bar_size: str, label: str):
        """Fetch data for the chosen period and run the backtest."""
        if self._bt_running:
            self._log("Backtest already running — please wait.")
            return
        self._bt_running = True
        self.bt_btn.config(state=tk.DISABLED)

        self._on_qty_change()
        self._on_trail_change()
        symbol = self.ticker_var.get().strip().upper()
        if not symbol:
            self._log("Enter a ticker symbol first.")
            self._bt_running = False
            self._set_button_states()
            return

        try:
            need = self.engine.strategy.min_bars + 10
            if (duration == self.engine.strategy.duration
                    and self.engine.bars
                    and len(self.engine.bars) >= need):
                # Reuse already-loaded live bars when duration matches
                self._run_backtest(self.engine.bars, label, bar_size)
            else:
                # Fetch fresh historical data from TWS
                bars = self.engine.fetch_history(symbol, duration, bar_size)
                need = self.engine.strategy.min_bars + 10
                if not bars or len(bars) < need:
                    self._log(
                        f"Not enough bars for backtest (got "
                        f"{len(bars) if bars else 0}, need {need}+). "
                        f"Try a shorter period or wait for market hours.")
                    return
                self._run_backtest(bars, label, bar_size)
        finally:
            self._bt_running = False
            self._set_button_states()

    # -- Backtester ----------------------------------------------------------

    def _run_backtest(self, bars, period_label='2 Day', bar_size='1 min'):
        """Replay all loaded bars through the strategy and show results."""
        strategy = self.engine.strategy
        indicators = strategy.compute(bars)
        dirs = indicators.get('trade_direction', indicators['direction'])
        atr_arr = indicators['atr']
        closes = np.array([b.close for b in bars], dtype=float)
        n = len(closes)

        # Diagnostic: count non-zero direction signals
        nonzero = int(np.count_nonzero(dirs))
        self._log(f"[BT] {n} bars, {nonzero} non-zero signals, "
                  f"min_bar={strategy.min_bars}")

        qty = self.engine.quantity
        trail_mult = self.engine.trail_mult
        profit_mult = self.engine.profit_mult
        min_bar = strategy.min_bars

        # Profit target array from strategy (if available)
        pt_arr = indicators.get('profit_target')

        # Simulation state
        pos = 0            # +1 long, -1 short, 0 flat
        entry_price = 0.0
        trail_high = 0.0
        trail_low = float('inf')
        trail_stop = 0.0
        bt_profit_target = 0.0  # active profit target price

        trades = []        # list of dicts per completed round-trip
        equity = [0.0]     # cumulative P&L curve (one per bar from min_bar)

        for i in range(min_bar, n):
            curr_close = closes[i]
            prev_dir = dirs[i - 1]
            curr_dir = dirs[i]
            curr_atr = atr_arr[i]
            pnl_this_bar = 0.0
            exited = False

            if np.isnan(curr_atr):
                equity.append(equity[-1])
                continue

            # --- Trailing stop check ---
            if pos != 0 and trail_mult > 0 and curr_atr > 0:
                trail_dist = trail_mult * curr_atr
                if pos == 1:
                    if curr_close > trail_high:
                        trail_high = curr_close
                    trail_stop = trail_high - trail_dist
                    if curr_close <= trail_stop:
                        pnl_this_bar = (curr_close - entry_price) * qty
                        trades.append({
                            'side': 'LONG', 'entry': entry_price,
                            'exit': curr_close, 'pnl': pnl_this_bar,
                            'exit_type': 'TRAIL', 'bar': i,
                        })
                        pos = 0
                        bt_profit_target = 0.0
                        exited = True
                elif pos == -1:
                    if curr_close < trail_low:
                        trail_low = curr_close
                    trail_stop = trail_low + trail_dist
                    if curr_close >= trail_stop:
                        pnl_this_bar = (entry_price - curr_close) * qty
                        trades.append({
                            'side': 'SHORT', 'entry': entry_price,
                            'exit': curr_close, 'pnl': pnl_this_bar,
                            'exit_type': 'TRAIL', 'bar': i,
                        })
                        pos = 0
                        bt_profit_target = 0.0
                        exited = True

            # --- Profit target check ---
            if not exited and pos != 0 and bt_profit_target > 0:
                if pos == 1 and curr_close >= bt_profit_target:
                    pnl_this_bar = (curr_close - entry_price) * qty
                    trades.append({
                        'side': 'LONG', 'entry': entry_price,
                        'exit': curr_close, 'pnl': pnl_this_bar,
                        'exit_type': 'PT', 'bar': i,
                    })
                    pos = 0
                    bt_profit_target = 0.0
                    exited = True
                elif pos == -1 and curr_close <= bt_profit_target:
                    pnl_this_bar = (entry_price - curr_close) * qty
                    trades.append({
                        'side': 'SHORT', 'entry': entry_price,
                        'exit': curr_close, 'pnl': pnl_this_bar,
                        'exit_type': 'PT', 'bar': i,
                    })
                    pos = 0
                    bt_profit_target = 0.0
                    exited = True

            if not exited:
                flip_bull = prev_dir <= 0 and curr_dir > 0
                flip_bear = prev_dir >= 0 and curr_dir < 0

                # Exit on direction reversal
                if pos == 1 and curr_dir < 0:
                    pnl_this_bar = (curr_close - entry_price) * qty
                    trades.append({
                        'side': 'LONG', 'entry': entry_price,
                        'exit': curr_close, 'pnl': pnl_this_bar,
                        'exit_type': 'REVERSAL', 'bar': i,
                    })
                    pos = 0
                    bt_profit_target = 0.0
                    exited = True
                elif pos == -1 and curr_dir > 0:
                    pnl_this_bar = (entry_price - curr_close) * qty
                    trades.append({
                        'side': 'SHORT', 'entry': entry_price,
                        'exit': curr_close, 'pnl': pnl_this_bar,
                        'exit_type': 'REVERSAL', 'bar': i,
                    })
                    pos = 0
                    bt_profit_target = 0.0
                    exited = True

                # Entry signals (only when flat, direction flip)
                if pos == 0 and flip_bull:
                    pos = 1
                    entry_price = curr_close
                    trail_high = curr_close
                    trail_low = float('inf')
                    trail_stop = 0.0
                    # Set profit target from strategy indicators
                    bt_profit_target = 0.0
                    if profit_mult > 0 and pt_arr is not None:
                        pt_val = pt_arr[i]
                        if not np.isnan(pt_val):
                            bt_profit_target = pt_val
                elif pos == 0 and flip_bear:
                    pos = -1
                    entry_price = curr_close
                    trail_low = curr_close
                    trail_high = 0.0
                    trail_stop = 0.0
                    bt_profit_target = 0.0
                    if profit_mult > 0 and pt_arr is not None:
                        pt_val = pt_arr[i]
                        if not np.isnan(pt_val):
                            bt_profit_target = pt_val

            equity.append(equity[-1] + pnl_this_bar)

        # Close any open position at last bar
        if pos == 1:
            pnl = (closes[-1] - entry_price) * qty
            trades.append({
                'side': 'LONG', 'entry': entry_price,
                'exit': closes[-1], 'pnl': pnl,
                'exit_type': 'OPEN', 'bar': n - 1,
            })
            equity[-1] += pnl
        elif pos == -1:
            pnl = (entry_price - closes[-1]) * qty
            trades.append({
                'side': 'SHORT', 'entry': entry_price,
                'exit': closes[-1], 'pnl': pnl,
                'exit_type': 'OPEN', 'bar': n - 1,
            })
            equity[-1] += pnl

        self._show_backtest_results(bars, trades, equity, min_bar,
                                     period_label, bar_size)

    def _show_backtest_results(self, bars, trades, equity, start_bar,
                               period_label='2 Day', bar_size='1 min'):
        """Popup window with backtest stats and equity curve."""
        # --- Compute stats ---
        total_trades = len(trades)
        if total_trades == 0:
            self._log("Backtest complete: 0 trades generated.")
            return

        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        win_rate = len(wins) / total_trades * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = (sum(wins) / abs(sum(losses))
                         if losses and sum(losses) != 0 else float('inf'))

        # Max drawdown from equity curve
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = dd.min()

        # Time range
        try:
            t0 = bars[start_bar].date.strftime('%m/%d %H:%M')
            t1 = bars[-1].date.strftime('%m/%d %H:%M')
        except Exception:
            t0, t1 = str(start_bar), str(len(bars) - 1)

        symbol = self.ticker_var.get().strip().upper()

        # --- Build popup window ---
        win = tk.Toplevel(self)
        strat_name = self.engine.strategy.name
        win.title(f"Backtest  —  {strat_name}  |  {symbol}  {period_label}  ({bar_size} bars)")
        win.configure(bg=self.BG)
        win.geometry("780x620")
        win.minsize(600, 500)

        # Stats text
        stats_frame = tk.Frame(win, bg=self.BG, padx=10, pady=8)
        stats_frame.pack(fill=tk.X)

        pnl_color = self.GREEN if total_pnl >= 0 else self.RED
        stats_lines = (
            f"Symbol: {symbol}   |   "
            f"Period: {period_label}  ({t0} -> {t1})   |   "
            f"Bar Size: {bar_size}\n"
            f"Bars: {len(bars)}   |   "
            f"Qty: {self.engine.quantity}   |   "
            f"Trail: {self.engine.trail_mult}x ATR   |   "
            f"PT: {self.engine.profit_mult}x Range\n"
            f"Trades: {total_trades}   |   "
            f"Win Rate: {win_rate:.1f}%   |   "
            f"Profit Factor: {profit_factor:.2f}\n"
            f"Avg Win: ${avg_win:,.2f}   |   "
            f"Avg Loss: ${avg_loss:,.2f}   |   "
            f"Max Drawdown: ${max_dd:,.2f}\n"
        )
        tk.Label(stats_frame, text=stats_lines, bg=self.BG, fg=self.FG,
                 font=("Consolas", 10), justify=tk.LEFT).pack(anchor='w')

        pnl_text = f"Net P&L:  ${total_pnl:,.2f}"
        tk.Label(stats_frame, text=pnl_text, bg=self.BG, fg=pnl_color,
                 font=("Consolas", 14, "bold")).pack(anchor='w', pady=(2, 0))

        # --- Equity curve chart ---
        fig = Figure(figsize=(7.5, 3.0), dpi=100,
                     facecolor=self.BG, edgecolor=self.BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0a0a1a')
        for spine in ax.spines.values():
            spine.set_color('#333')
        ax.tick_params(colors='#888', labelsize=7)
        ax.grid(True, color='#222', linewidth=0.5)

        eq_x = np.arange(len(equity))
        ax.fill_between(eq_x, equity, 0,
                         where=np.array(equity) >= 0,
                         color=self.GREEN, alpha=0.15)
        ax.fill_between(eq_x, equity, 0,
                         where=np.array(equity) < 0,
                         color=self.RED, alpha=0.15)
        ax.plot(eq_x, equity, color='#4fc3f7', linewidth=1.2)
        ax.axhline(y=0, color='#555', linewidth=0.5)
        ax.set_title(f"Equity Curve  —  {period_label}  ({bar_size})",
                      color='#ccc', fontsize=10, loc='left')
        ax.set_ylabel("P&L ($)", color='#888', fontsize=8)
        fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        canvas.draw()

        # --- Trade log table ---
        table_frame = tk.Frame(win, bg=self.BG)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        scrollbar = tk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        trade_text = tk.Text(
            table_frame, height=10, bg='#0a0a1a', fg=self.FG,
            font=("Consolas", 9), wrap=tk.NONE, state=tk.DISABLED,
            yscrollcommand=scrollbar.set, relief=tk.FLAT)
        trade_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=trade_text.yview)

        trade_text.tag_configure('win', foreground=self.GREEN)
        trade_text.tag_configure('lose', foreground=self.RED)
        trade_text.tag_configure('header', foreground=self.YELLOW)

        header = (f"{'#':>3}  {'Side':<6} {'Entry':>9} {'Exit':>9} "
                  f"{'P&L':>10}  {'Type':<9} {'Bar':>5}\n")
        sep = "-" * 60 + "\n"

        trade_text.config(state=tk.NORMAL)
        trade_text.insert(tk.END, header, 'header')
        trade_text.insert(tk.END, sep, 'header')
        for idx, t in enumerate(trades, 1):
            tag = 'win' if t['pnl'] > 0 else 'lose'
            line = (f"{idx:>3}  {t['side']:<6} "
                    f"${t['entry']:>8.2f} ${t['exit']:>8.2f} "
                    f"${t['pnl']:>+9.2f}  {t['exit_type']:<9} "
                    f"{t['bar']:>5}\n")
            trade_text.insert(tk.END, line, tag)
        trade_text.config(state=tk.DISABLED)

        self._log(f"Backtest [{period_label} {bar_size}]: "
                  f"{total_trades} trades, "
                  f"P&L=${total_pnl:,.2f}, "
                  f"Win={win_rate:.1f}%")

    def _set_button_states(self):
        state = self.engine.state
        self.status_var.set(state)
        has_bars = (self.engine.bars is not None
                    and len(self.engine.bars) >= self.engine.strategy.min_bars)
        can_change = state in (TradingEngine.IDLE, TradingEngine.WATCHING)
        self.strategy_menu.config(state=tk.NORMAL if can_change else tk.DISABLED)
        if state == TradingEngine.IDLE:
            self.start_btn.config(state=tk.NORMAL, text="Start")
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.bt_btn.config(state=tk.NORMAL if has_bars else tk.DISABLED)
            self.ticker_entry.config(state=tk.NORMAL)
            self.status_label.config(fg=self.FG)
        elif state == TradingEngine.WATCHING:
            self.start_btn.config(state=tk.NORMAL, text="Start")
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.bt_btn.config(state=tk.NORMAL if has_bars else tk.DISABLED)
            self.ticker_entry.config(state=tk.NORMAL)
            self.status_label.config(fg='#4fc3f7')  # light blue
        elif state == TradingEngine.RUNNING:
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.bt_btn.config(state=tk.DISABLED)
            self.ticker_entry.config(state=tk.DISABLED)
            self.status_label.config(fg=self.GREEN)
        elif state == TradingEngine.PAUSED:
            self.start_btn.config(state=tk.NORMAL, text="Resume")
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.bt_btn.config(state=tk.NORMAL if has_bars else tk.DISABLED)
            self.ticker_entry.config(state=tk.DISABLED)
            self.status_label.config(fg=self.YELLOW)

    # -- Chart ---------------------------------------------------------------

    def _update_chart(self, bars, indicators):
        if not bars or len(bars) == 0:
            return

        strategy = self.engine.strategy
        has_subpanel = getattr(strategy, 'needs_subpanel', False)

        # Manage sub-panel axes
        if has_subpanel:
            self.fig.clear()
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
            self.ax = self.fig.add_subplot(gs[0])
            ax_sub = self.fig.add_subplot(gs[1], sharex=self.ax)
            self._style_axes()
            ax_sub.set_facecolor('#0a0a1a')
            for spine in ax_sub.spines.values():
                spine.set_color('#333')
            ax_sub.tick_params(colors='#888', labelsize=7)
            ax_sub.grid(True, color='#222', linewidth=0.5)
        else:
            self.ax.clear()
            self._style_axes()
            ax_sub = None

        # Show last 100 bars
        window = 100
        n = len(bars)
        start = max(0, n - window)
        windowed = bars[start:]
        closes = np.array([b.close for b in windowed])
        x = np.arange(len(closes))

        # Extract timestamps for x-axis labels
        bar_times = []
        for b in windowed:
            try:
                bar_times.append(b.date)
            except Exception:
                bar_times.append(None)

        # Price line (always)
        self.ax.plot(x, closes, color='#ffffff', linewidth=1.0,
                     label='Close', zorder=3)

        # Delegate indicator overlays to the strategy
        colors = {'green': self.GREEN, 'red': self.RED, 'yellow': self.YELLOW}
        strategy.plot_indicators(self.ax, ax_sub, indicators, x, start, colors)

        # --- Generic trade markers using trade_direction ---
        trade_dir = indicators.get('trade_direction', indicators['direction'])
        dirs = trade_dir[start:]
        atr_arr = indicators['atr'][start:]
        trail_mult = self.engine.trail_mult
        sim_pos = 0
        sim_trail_high = 0.0
        sim_trail_low = float('inf')
        sim_trail_stop = 0.0
        sim_trail_stops = np.full(len(closes), np.nan)

        for i in range(1, len(dirs)):
            curr_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 0.0
            flip_bull = dirs[i - 1] <= 0 and dirs[i] > 0
            flip_bear = dirs[i - 1] >= 0 and dirs[i] < 0
            trail_hit = False

            if sim_pos != 0 and trail_mult > 0 and curr_atr > 0:
                trail_dist = trail_mult * curr_atr
                if sim_pos > 0:
                    if closes[i] > sim_trail_high:
                        sim_trail_high = closes[i]
                    sim_trail_stop = sim_trail_high - trail_dist
                    sim_trail_stops[i] = sim_trail_stop
                    if closes[i] <= sim_trail_stop:
                        trail_hit = True
                elif sim_pos < 0:
                    if closes[i] < sim_trail_low:
                        sim_trail_low = closes[i]
                    sim_trail_stop = sim_trail_low + trail_dist
                    sim_trail_stops[i] = sim_trail_stop
                    if closes[i] >= sim_trail_stop:
                        trail_hit = True

            if trail_hit:
                va = 'top' if sim_pos > 0 else 'bottom'
                self.ax.annotate('TRAIL', xy=(x[i], closes[i]),
                                 fontsize=7, color='#e040fb',
                                 fontweight='bold', ha='center',
                                 va=va, zorder=5)
                self.ax.plot(x[i], closes[i], marker='d',
                             color='#e040fb', markersize=7, zorder=5)
                sim_pos = 0
                sim_trail_high = 0.0
                sim_trail_low = float('inf')
                sim_trail_stop = 0.0
            elif flip_bull:
                self.ax.annotate('BUY', xy=(x[i], closes[i]),
                                 fontsize=7, color=self.GREEN,
                                 fontweight='bold', ha='center',
                                 va='bottom', zorder=5)
                self.ax.plot(x[i], closes[i], marker='^',
                             color=self.GREEN, markersize=8, zorder=5)
                if sim_pos < 0:
                    sim_pos = 0
                sim_pos = 1
                sim_trail_high = closes[i]
                sim_trail_low = float('inf')
                sim_trail_stop = 0.0
            elif flip_bear:
                self.ax.annotate('SELL', xy=(x[i], closes[i]),
                                 fontsize=7, color=self.RED,
                                 fontweight='bold', ha='center',
                                 va='top', zorder=5)
                self.ax.plot(x[i], closes[i], marker='v',
                             color=self.RED, markersize=8, zorder=5)
                if sim_pos > 0:
                    sim_pos = 0
                sim_pos = -1
                sim_trail_low = closes[i]
                sim_trail_high = 0.0
                sim_trail_stop = 0.0
            elif sim_pos > 0 and dirs[i] < 0:
                self.ax.annotate('EXIT', xy=(x[i], closes[i]),
                                 fontsize=7, color='#ff9800',
                                 fontweight='bold', ha='center',
                                 va='top', zorder=5)
                self.ax.plot(x[i], closes[i], marker='x',
                             color='#ff9800', markersize=7, zorder=5)
                sim_pos = 0
                sim_trail_high = 0.0
                sim_trail_low = float('inf')
                sim_trail_stop = 0.0
            elif sim_pos < 0 and dirs[i] > 0:
                self.ax.annotate('EXIT', xy=(x[i], closes[i]),
                                 fontsize=7, color='#ff9800',
                                 fontweight='bold', ha='center',
                                 va='bottom', zorder=5)
                self.ax.plot(x[i], closes[i], marker='x',
                             color='#ff9800', markersize=7, zorder=5)
                sim_pos = 0
                sim_trail_high = 0.0
                sim_trail_low = float('inf')
                sim_trail_stop = 0.0

        # --- Simulated trailing stop line ---
        mask_trail = ~np.isnan(sim_trail_stops)
        if mask_trail.any():
            self.ax.plot(x[mask_trail], sim_trail_stops[mask_trail],
                         color='#e040fb', linewidth=1.0, linestyle=':',
                         alpha=0.7, label='Trail Stop', zorder=2)

        # --- Live trailing stop level ---
        if (self.engine.trail_active and self.engine.trail_stop > 0 and
                self.engine.state == TradingEngine.RUNNING):
            self.ax.axhline(y=self.engine.trail_stop, color='#e040fb',
                            linewidth=1.0, linestyle='--', alpha=0.8)
            self.ax.text(x[-1] + 1, self.engine.trail_stop,
                         f'STOP {self.engine.trail_stop:.2f}',
                         color='#e040fb', fontsize=7, va='center')

        # --- Live trade arrows (actual engine entries/exits) ---
        for mk in self.engine.live_markers:
            bt = mk['bar_time']
            # Find matching bar index in the visible window
            idx = None
            for j, t in enumerate(bar_times):
                if t is not None and bt is not None and t == bt:
                    idx = j
                    break
            if idx is None:
                continue
            sig = mk['signal']
            px = mk['price']
            if sig == 'BUY':
                self.ax.annotate('', xy=(x[idx], px),
                                 xytext=(x[idx], px - (closes.max() - closes.min()) * 0.08),
                                 arrowprops=dict(arrowstyle='->', color=self.GREEN,
                                                 lw=2.5), zorder=10)
                self.ax.scatter(x[idx], px, marker='^', color=self.GREEN,
                                s=120, edgecolors='white', linewidths=0.8, zorder=11)
            elif sig == 'SELL':
                self.ax.annotate('', xy=(x[idx], px),
                                 xytext=(x[idx], px + (closes.max() - closes.min()) * 0.08),
                                 arrowprops=dict(arrowstyle='->', color=self.RED,
                                                 lw=2.5), zorder=10)
                self.ax.scatter(x[idx], px, marker='v', color=self.RED,
                                s=120, edgecolors='white', linewidths=0.8, zorder=11)
            elif 'EXIT' in sig or 'TRAIL' in sig:
                self.ax.scatter(x[idx], px, marker='x', color='#ff9800',
                                s=120, linewidths=2.5, zorder=11)
                self.ax.annotate('EXIT', xy=(x[idx], px),
                                 xytext=(x[idx], px + (closes.max() - closes.min()) * 0.05),
                                 fontsize=8, color='#ff9800', fontweight='bold',
                                 ha='center', va='bottom', zorder=10)

        # Current price annotation
        last_price = closes[-1]
        self.ax.axhline(y=last_price, color='#4fc3f7', linewidth=0.5,
                         linestyle='--', alpha=0.6)
        self.ax.text(x[-1] + 1, last_price, f'{last_price:.2f}',
                     color='#4fc3f7', fontsize=8, va='center')

        # --- X-axis time labels ---
        target_ax = ax_sub if ax_sub is not None else self.ax
        num_ticks = min(8, len(x))
        tick_indices = np.linspace(0, len(x) - 1, num_ticks, dtype=int)
        tick_labels = []
        prev_date_str = ''
        for idx in tick_indices:
            bt = bar_times[idx]
            if bt is None:
                tick_labels.append('')
                continue
            try:
                date_str = bt.strftime('%m/%d')
                time_str = bt.strftime('%H:%M')
                if date_str != prev_date_str:
                    tick_labels.append(f'{date_str}\n{time_str}')
                    prev_date_str = date_str
                else:
                    tick_labels.append(time_str)
            except Exception:
                tick_labels.append(str(bt)[-5:])
        target_ax.set_xticks(x[tick_indices])
        target_ax.set_xticklabels(tick_labels, fontsize=7, color='#888')
        if ax_sub is not None:
            self.ax.tick_params(labelbottom=False)

        # Last bar timestamp for title
        last_bar = bars[-1]
        try:
            bar_time = last_bar.date.strftime('%a %m/%d %H:%M')
        except Exception:
            bar_time = str(last_bar.date)

        now = self._tws_now()
        weekday = now.weekday()
        hour = now.hour
        if weekday >= 5:
            market_tag = "  [MARKET CLOSED - weekend]"
        elif hour < 9 or (hour == 9 and now.minute < 30) or hour >= 16:
            market_tag = "  [MARKET CLOSED]"
        else:
            market_tag = "  [LIVE]"

        self.ax.legend(loc='upper left', fontsize=8,
                       facecolor='#111', edgecolor='#333',
                       labelcolor='#ccc')
        self.ax.set_title(
            f"{self.ticker_var.get().upper()}  {self.engine.strategy.bar_size}  |  "
            f"Last: {last_price:.2f}  |  {bar_time}{market_tag}",
            color='#ccc', fontsize=10, loc='left')
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw_idle()

    # -- Log -----------------------------------------------------------------

    def _log(self, msg: str):
        timestamp = self._tws_now().strftime('%H:%M:%S')
        line = f"[{timestamp}]  {msg}\n"

        tag = 'info'
        msg_upper = msg.upper()
        if 'BUY' in msg_upper or 'LONG' in msg_upper:
            tag = 'buy'
        elif 'SELL' in msg_upper or 'SHORT' in msg_upper or 'LIQUIDAT' in msg_upper:
            tag = 'sell'

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, line, tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    # -- Asyncio pump --------------------------------------------------------

    def _pump_asyncio(self):
        """Process pending asyncio callbacks alongside Tkinter mainloop."""
        loop = asyncio.get_event_loop()
        loop.stop()
        loop.run_forever()
        self.after(50, self._pump_asyncio)

    # -- Cleanup -------------------------------------------------------------

    def destroy(self):
        if self.engine.ib.isConnected():
            self.engine.stop()
            self.engine.disconnect()
        super().destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
