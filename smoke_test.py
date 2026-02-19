"""
Comprehensive smoke test for all strategies and engine logic.
Run: python smoke_test.py
"""

import sys
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# Import everything from app.py
# ---------------------------------------------------------------------------
from app import (
    BaseStrategy, SupertrendStrategy, DualMovingAverageStrategy,
    DonchianBreakoutStrategy, BollingerBounceStrategy,
    VWAPReversionStrategy, OpeningRangeBreakoutStrategy,
    ATRTrailingStopStrategy, MACDCrossoverStrategy,
    RSIOverboughtOversoldStrategy, SqueezeMomentumStrategy,
    OBVDivergenceStrategy, IchimokuCloudStrategy,
    KeltnerADXStrategy, ZScoreMeanReversionStrategy,
    ROCDivergenceStrategy, TripleScreenStrategy,
    StochRSIVWAPStrategy, VolatilityBreakoutStrategy,
    ATRChannelBreakoutStrategy, PivotPointBounceStrategy,
    InsideBarBreakoutStrategy, HeikinAshiTrendStrategy,
    LinRegChannelStrategy,
    STRATEGY_REGISTRY, STRATEGY_NAMES, TradingEngine,
)

ET = ZoneInfo('America/New_York')
PASS = 0
FAIL = 0


def log_pass(name):
    global PASS
    PASS += 1
    print(f"  PASS  {name}")


def log_fail(name, err):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name}  -->  {err}")


# ---------------------------------------------------------------------------
# Synthetic bar data generators
# ---------------------------------------------------------------------------

def make_bars(n, base_price=100.0, volatility=1.5, start_time=None,
              bar_minutes=1, with_volume=True, trend='up'):
    """Generate n synthetic OHLCV bars as SimpleNamespace objects."""
    if start_time is None:
        start_time = datetime(2025, 6, 2, 9, 30, tzinfo=ET)  # Monday 9:30 ET

    bars = []
    price = base_price
    for i in range(n):
        dt = start_time + timedelta(minutes=i * bar_minutes)
        # Generate realistic OHLC
        if trend == 'up':
            drift = 0.02
        elif trend == 'down':
            drift = -0.02
        elif trend == 'sine':
            drift = np.sin(i / 20.0) * 0.3
        else:
            drift = 0.0

        change = drift + (np.random.randn() * volatility * 0.01)
        o = price
        c = price * (1 + change)
        h = max(o, c) * (1 + abs(np.random.randn()) * 0.002)
        l = min(o, c) * (1 - abs(np.random.randn()) * 0.002)
        vol = int(10000 + np.random.randint(0, 5000)) if with_volume else 0

        bars.append(SimpleNamespace(
            date=dt, open=o, high=h, low=l, close=c,
            volume=vol, barCount=0, average=0.0,
        ))
        price = c
    return bars


def make_multiday_bars(n_days=3, bars_per_day=130):
    """Generate bars spanning multiple days for pivot/VWAP strategies."""
    bars = []
    price = 100.0
    for day in range(n_days):
        # Monday=0, so start on Monday June 2
        start = datetime(2025, 6, 2 + day, 9, 30, tzinfo=ET)
        for i in range(bars_per_day):
            dt = start + timedelta(minutes=i)
            change = np.sin(i / 30.0) * 0.3 + np.random.randn() * 0.1
            o = price
            c = price + change
            h = max(o, c) + abs(np.random.randn()) * 0.2
            l = min(o, c) - abs(np.random.randn()) * 0.2
            bars.append(SimpleNamespace(
                date=dt, open=o, high=h, low=l, close=c,
                volume=10000 + np.random.randint(0, 5000),
                barCount=0, average=0.0,
            ))
            price = c
    return bars


def make_orb_bars(n=200):
    """Generate bars that include an opening range and a breakout."""
    start = datetime(2025, 6, 2, 9, 30, tzinfo=ET)
    bars = []
    price = 100.0
    for i in range(n):
        dt = start + timedelta(minutes=i)
        minutes_since_open = i

        if minutes_since_open < 30:
            # Opening range: tight, bounded 99-101
            change = np.random.randn() * 0.001
            price = 100.0 + np.random.randn() * 0.3
            price = max(99.0, min(101.0, price))
        elif minutes_since_open == 30:
            # Breakout bar
            price = 102.0
        else:
            # Post-breakout drift up
            price = price + np.random.randn() * 0.1 + 0.05

        o = price - 0.1
        c = price
        h = max(o, c) + 0.2
        l = min(o, c) - 0.2
        bars.append(SimpleNamespace(
            date=dt, open=o, high=h, low=l, close=c,
            volume=10000, barCount=0, average=0.0,
        ))
    return bars


# ---------------------------------------------------------------------------
# Test: Strategy registry is complete
# ---------------------------------------------------------------------------

def test_registry():
    print("\n=== Strategy Registry ===")
    expected_keys = [
        'orb', 'squeeze', 'vwap', 'supertrend', 'macd', 'rsi',
        'dual_ma', 'donchian', 'bollinger', 'atr_trail', 'obv', 'ichimoku',
        'keltner_adx', 'zscore', 'roc_div', 'triple_screen',
        'stoch_rsi_vwap', 'vol_breakout', 'atr_channel', 'pivot_bounce',
        'inside_bar', 'heikin_ashi', 'linreg',
    ]
    for k in expected_keys:
        if k in STRATEGY_REGISTRY:
            log_pass(f"Registry has '{k}'")
        else:
            log_fail(f"Registry has '{k}'", "MISSING")

    # Check STRATEGY_NAMES maps all display names back
    for name, key in STRATEGY_NAMES.items():
        if key in STRATEGY_REGISTRY:
            log_pass(f"STRATEGY_NAMES['{name}'] -> '{key}'")
        else:
            log_fail(f"STRATEGY_NAMES mapping", f"'{key}' not in registry")


# ---------------------------------------------------------------------------
# Test: Each strategy computes without error and returns valid outputs
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [
    ('supertrend', SupertrendStrategy, 500, {'trend': 'up'}),
    ('dual_ma', DualMovingAverageStrategy, 500, {'bar_minutes': 5, 'trend': 'sine'}),
    ('donchian', DonchianBreakoutStrategy, 200, {'bar_minutes': 5, 'trend': 'up'}),
    ('bollinger', BollingerBounceStrategy, 200, {'bar_minutes': 5, 'trend': 'sine'}),
    ('vwap', VWAPReversionStrategy, 200, {'trend': 'sine'}),
    ('orb', OpeningRangeBreakoutStrategy, None, {}),  # special data
    ('atr_trail', ATRTrailingStopStrategy, 200, {'trend': 'up'}),
    ('macd', MACDCrossoverStrategy, 200, {'trend': 'sine'}),
    ('rsi', RSIOverboughtOversoldStrategy, 500, {'trend': 'sine'}),
    ('squeeze', SqueezeMomentumStrategy, 200, {'trend': 'sine'}),
    ('obv', OBVDivergenceStrategy, 200, {'trend': 'sine'}),
    ('ichimoku', IchimokuCloudStrategy, 300, {'bar_minutes': 5, 'trend': 'up'}),
    # --- New strategies ---
    ('keltner_adx', KeltnerADXStrategy, 200, {'trend': 'up'}),
    ('zscore', ZScoreMeanReversionStrategy, 200, {'trend': 'sine'}),
    ('roc_div', ROCDivergenceStrategy, 200, {'trend': 'sine'}),
    ('triple_screen', TripleScreenStrategy, 300, {'trend': 'up'}),
    ('stoch_rsi_vwap', StochRSIVWAPStrategy, 200, {'trend': 'sine'}),
    ('vol_breakout', VolatilityBreakoutStrategy, 200, {'trend': 'up'}),
    ('atr_channel', ATRChannelBreakoutStrategy, 200, {'trend': 'up'}),
    ('pivot_bounce', PivotPointBounceStrategy, None, {}),  # needs multi-day data
    ('inside_bar', InsideBarBreakoutStrategy, 200, {'trend': 'sine'}),
    ('heikin_ashi', HeikinAshiTrendStrategy, 200, {'trend': 'up'}),
    ('linreg', LinRegChannelStrategy, 200, {'trend': 'sine'}),
]


def test_strategy_compute():
    print("\n=== Strategy compute() ===")
    for key, cls, n_bars, bar_kwargs in ALL_STRATEGIES:
        strat = cls()
        name = strat.name

        # Generate appropriate bars
        if key == 'orb':
            bars = make_orb_bars(200)
        elif key == 'pivot_bounce':
            bars = make_multiday_bars(n_days=3, bars_per_day=130)
        else:
            bars = make_bars(n_bars, **bar_kwargs)

        n = len(bars)

        try:
            result = strat.compute(bars)
        except Exception as e:
            log_fail(f"{name} compute()", f"Exception: {e}")
            traceback.print_exc()
            continue

        # Validate return type
        if not isinstance(result, dict):
            log_fail(f"{name} compute()", f"returned {type(result)}, expected dict")
            continue

        # Must have 'direction' and 'atr'
        for required_key in ('direction', 'atr'):
            if required_key not in result:
                log_fail(f"{name} has '{required_key}'", "MISSING from result dict")
            else:
                arr = result[required_key]
                if not isinstance(arr, np.ndarray):
                    log_fail(f"{name} '{required_key}' type",
                             f"expected ndarray, got {type(arr)}")
                elif len(arr) != n:
                    log_fail(f"{name} '{required_key}' length",
                             f"expected {n}, got {len(arr)}")
                else:
                    log_pass(f"{name} '{required_key}' shape={arr.shape}")

        # Direction values must be in {-1, 0, 1}
        dirs = result.get('direction')
        if dirs is not None and isinstance(dirs, np.ndarray):
            unique = set(np.unique(dirs))
            if unique <= {-1, 0, 1}:
                log_pass(f"{name} direction values in {{-1,0,1}}")
            else:
                log_fail(f"{name} direction values",
                         f"unexpected values: {unique}")

        # If trade_direction exists, same checks
        if 'trade_direction' in result:
            td = result['trade_direction']
            if len(td) != n:
                log_fail(f"{name} trade_direction length",
                         f"expected {n}, got {len(td)}")
            else:
                log_pass(f"{name} trade_direction shape={td.shape}")
            unique_td = set(np.unique(td))
            if unique_td <= {-1, 0, 1}:
                log_pass(f"{name} trade_direction values in {{-1,0,1}}")
            else:
                log_fail(f"{name} trade_direction values",
                         f"unexpected values: {unique_td}")

        # ATR should have NaN padding then valid values
        atr = result.get('atr')
        if atr is not None and isinstance(atr, np.ndarray):
            valid_atr = atr[~np.isnan(atr)]
            if len(valid_atr) > 0 and np.all(valid_atr >= 0):
                log_pass(f"{name} ATR has valid non-negative values")
            elif len(valid_atr) == 0:
                log_fail(f"{name} ATR", "all NaN — no valid ATR values")
            else:
                log_fail(f"{name} ATR", "negative ATR values found")

        # Check direction has non-zero values (signals exist)
        if dirs is not None:
            nonzero = int(np.count_nonzero(dirs))
            if nonzero > 0:
                log_pass(f"{name} has {nonzero} non-zero signals")
            else:
                log_fail(f"{name} signals",
                         "0 non-zero direction values (no signals generated)")

        log_pass(f"{name} compute() completed OK")


# ---------------------------------------------------------------------------
# Test: Each strategy's plot_indicators doesn't crash
# ---------------------------------------------------------------------------

def test_strategy_plot():
    print("\n=== Strategy plot_indicators() ===")
    from matplotlib.figure import Figure

    for key, cls, n_bars, bar_kwargs in ALL_STRATEGIES:
        strat = cls()
        name = strat.name

        if key == 'orb':
            bars = make_orb_bars(200)
        elif key == 'pivot_bounce':
            bars = make_multiday_bars(n_days=3, bars_per_day=130)
        else:
            bars = make_bars(n_bars, **bar_kwargs)

        result = strat.compute(bars)
        n = len(bars)
        start = max(0, n - 100)
        x = np.arange(n - start)

        fig = Figure(figsize=(8, 4))
        ax_price = fig.add_subplot(111)
        ax_sub = fig.add_subplot(111) if strat.needs_subpanel else None

        colors = {'green': '#00e676', 'red': '#ff1744', 'yellow': '#ffd600'}

        try:
            strat.plot_indicators(ax_price, ax_sub, result, x, start, colors)
            log_pass(f"{name} plot_indicators() OK")
        except Exception as e:
            log_fail(f"{name} plot_indicators()", f"Exception: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Test: Strategies with edge cases (too few bars, empty data)
# ---------------------------------------------------------------------------

def test_edge_cases():
    print("\n=== Edge Cases ===")
    for key, cls, _, _ in ALL_STRATEGIES:
        strat = cls()
        name = strat.name

        # Test with very few bars (less than min_bars)
        few_bars = make_bars(5)
        try:
            result = strat.compute(few_bars)
            if isinstance(result, dict) and 'direction' in result and 'atr' in result:
                log_pass(f"{name} handles few bars (n=5)")
            else:
                log_fail(f"{name} few bars", "missing required keys")
        except Exception as e:
            log_fail(f"{name} few bars", f"Exception: {e}")
            traceback.print_exc()

        # Test with exactly min_bars
        min_b = strat.min_bars
        exact_bars = make_bars(min_b)
        try:
            result = strat.compute(exact_bars)
            if isinstance(result, dict) and 'direction' in result:
                log_pass(f"{name} handles exactly min_bars (n={min_b})")
            else:
                log_fail(f"{name} exact min_bars", "missing required keys")
        except Exception as e:
            log_fail(f"{name} exact min_bars", f"Exception: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Test: BaseStrategy shared helpers
# ---------------------------------------------------------------------------

def test_base_helpers():
    print("\n=== BaseStrategy Helpers ===")
    base = BaseStrategy()

    # EMA
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    ema = base._ema(data, 3)
    assert len(ema) == 10
    assert np.isnan(ema[0]) and np.isnan(ema[1])
    assert not np.isnan(ema[2])
    log_pass("_ema() basic")

    # SMA
    sma = base._sma(data, 3)
    assert len(sma) == 10
    assert np.isnan(sma[0]) and np.isnan(sma[1])
    assert abs(sma[2] - 2.0) < 1e-10  # mean(1,2,3)=2
    log_pass("_sma() basic")

    # ATR
    high = np.array([2, 3, 4, 5, 6], dtype=float)
    low = np.array([1, 2, 3, 4, 5], dtype=float)
    close = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=float)
    atr = base._atr(high, low, close, 3)
    assert len(atr) == 5
    assert np.isnan(atr[0]) and np.isnan(atr[1])
    assert not np.isnan(atr[2])
    assert atr[2] > 0
    log_pass("_atr() basic")

    # Rolling std
    std = base._rolling_std(data, 3)
    assert len(std) == 10
    assert np.isnan(std[0]) and np.isnan(std[1])
    assert not np.isnan(std[2])
    log_pass("_rolling_std() basic")

    # _empty
    empty = base._empty(5)
    assert len(empty['direction']) == 5
    assert len(empty['atr']) == 5
    assert np.all(empty['direction'] == 0)
    log_pass("_empty() basic")

    # Edge: period larger than data
    ema_short = base._ema(np.array([1.0, 2.0]), 10)
    assert np.all(np.isnan(ema_short))
    log_pass("_ema() with period > data length")

    sma_short = base._sma(np.array([1.0, 2.0]), 10)
    assert np.all(np.isnan(sma_short))
    log_pass("_sma() with period > data length")


# ---------------------------------------------------------------------------
# Test: TradingEngine signal evaluation logic (mocked)
# ---------------------------------------------------------------------------

def test_engine_signal_logic():
    print("\n=== TradingEngine Signal Logic ===")

    log_messages = []

    def mock_log(msg):
        log_messages.append(msg)

    for key, cls, n_bars, bar_kwargs in ALL_STRATEGIES:
        strat = cls()
        name = strat.name

        engine = TradingEngine(
            on_bar_update=lambda *a: None,
            on_log=mock_log,
            strategy=strat,
        )

        if key == 'orb':
            bars = make_orb_bars(200)
        elif key == 'pivot_bounce':
            bars = make_multiday_bars(n_days=3, bars_per_day=130)
        else:
            bars = make_bars(n_bars or 200, **bar_kwargs)

        indicators = strat.compute(bars)
        trade_dir = indicators.get('trade_direction', indicators['direction'])

        # Verify engine's _evaluate_signal doesn't crash
        # We need to set up minimal engine state
        engine.bars = bars
        engine.state = TradingEngine.RUNNING
        engine.contract = SimpleNamespace(symbol='TEST', secType='STK')
        engine.position = 0
        engine.quantity = 100

        # Mock _place_order to avoid actual IB calls
        placed_orders = []
        def mock_place_order(action, qty, tag=''):
            placed_orders.append({'action': action, 'qty': qty, 'tag': tag})
        engine._place_order = mock_place_order

        # Mock _get_ib_position
        engine._get_ib_position = lambda: 0

        log_messages.clear()
        try:
            engine._evaluate_signal(bars, indicators)
            log_pass(f"{name} _evaluate_signal() OK")
        except Exception as e:
            log_fail(f"{name} _evaluate_signal()", f"Exception: {e}")
            traceback.print_exc()

    # Test evaluate_immediate_entry
    print()
    for key, cls, n_bars, bar_kwargs in ALL_STRATEGIES:
        strat = cls()
        name = strat.name

        engine = TradingEngine(
            on_bar_update=lambda *a: None,
            on_log=mock_log,
            strategy=strat,
        )

        if key == 'orb':
            bars = make_orb_bars(200)
        elif key == 'pivot_bounce':
            bars = make_multiday_bars(n_days=3, bars_per_day=130)
        else:
            bars = make_bars(n_bars or 200, **bar_kwargs)

        engine.bars = bars
        engine.state = TradingEngine.RUNNING
        engine.contract = SimpleNamespace(symbol='TEST', secType='STK')
        engine.position = 0
        engine.quantity = 100

        placed_orders = []
        engine._place_order = lambda action, qty, tag='': placed_orders.append(
            {'action': action, 'qty': qty, 'tag': tag})
        engine._get_ib_position = lambda: 0

        log_messages.clear()
        try:
            engine._evaluate_immediate_entry()
            log_pass(f"{name} _evaluate_immediate_entry() OK")
        except Exception as e:
            log_fail(f"{name} _evaluate_immediate_entry()", f"Exception: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Test: Engine _entry_allowed
# ---------------------------------------------------------------------------

def test_entry_allowed():
    print("\n=== _entry_allowed() ===")
    engine = TradingEngine(on_log=lambda m: None)

    # During market hours (Monday 10:00 ET)
    mkt_open = datetime(2025, 6, 2, 10, 0, tzinfo=ET)
    assert engine._entry_allowed(mkt_open) == True
    log_pass("_entry_allowed during market hours = True")

    # Before market open (Monday 9:29 ET)
    pre_open = datetime(2025, 6, 2, 9, 29, tzinfo=ET)
    assert engine._entry_allowed(pre_open) == False
    log_pass("_entry_allowed before 9:30 = False")

    # After market close (Monday 16:00 ET)
    post_close = datetime(2025, 6, 2, 16, 0, tzinfo=ET)
    assert engine._entry_allowed(post_close) == False
    log_pass("_entry_allowed after 16:00 = False")

    # Weekend (Saturday)
    weekend = datetime(2025, 6, 7, 12, 0, tzinfo=ET)
    assert engine._entry_allowed(weekend) == False
    log_pass("_entry_allowed on weekend = False")

    # Exactly 9:30 ET
    exact_open = datetime(2025, 6, 2, 9, 30, tzinfo=ET)
    assert engine._entry_allowed(exact_open) == True
    log_pass("_entry_allowed at exactly 9:30 = True")

    # 15:59 (last minute)
    last_min = datetime(2025, 6, 2, 15, 59, tzinfo=ET)
    assert engine._entry_allowed(last_min) == True
    log_pass("_entry_allowed at 15:59 = True")


# ---------------------------------------------------------------------------
# Test: Engine strategy switching
# ---------------------------------------------------------------------------

def test_strategy_switching():
    print("\n=== Strategy Switching ===")
    log_msgs = []
    engine = TradingEngine(
        on_bar_update=lambda *a: None,
        on_log=lambda m: log_msgs.append(m),
    )

    # Can switch when IDLE
    for key in STRATEGY_REGISTRY:
        result = engine.set_strategy(key)
        if result:
            log_pass(f"Switch to '{key}' when IDLE")
        else:
            log_fail(f"Switch to '{key}' when IDLE", "returned False")

    # Cannot switch when RUNNING
    engine.state = TradingEngine.RUNNING
    result = engine.set_strategy('macd')
    if not result:
        log_pass("Switch blocked when RUNNING")
    else:
        log_fail("Switch blocked when RUNNING", "should return False")

    # Cannot switch when PAUSED
    engine.state = TradingEngine.PAUSED
    result = engine.set_strategy('rsi')
    if not result:
        log_pass("Switch blocked when PAUSED")
    else:
        log_fail("Switch blocked when PAUSED", "should return False")

    # Invalid strategy key
    engine.state = TradingEngine.IDLE
    result = engine.set_strategy('nonexistent_strategy')
    if not result:
        log_pass("Switch to invalid key returns False")
    else:
        log_fail("Switch to invalid key", "should return False")


# ---------------------------------------------------------------------------
# Test: Trailing stop logic
# ---------------------------------------------------------------------------

def test_trailing_stop():
    print("\n=== Trailing Stop ===")
    engine = TradingEngine(on_log=lambda m: None)
    engine.trail_mult = 2.0

    # Long position: price rises then falls below stop
    engine.position = 1
    engine._reset_trail(100.0, 'long')
    assert engine.trail_active == True
    log_pass("Trail active after reset")

    # Price rises
    result = engine._update_trail(105.0, 1.0)  # ATR=1, stop=105-2=103
    assert result is None
    assert engine.trail_high == 105.0
    log_pass("Trail updates high watermark")

    # Price drops but above stop
    result = engine._update_trail(103.5, 1.0)  # stop=105-2=103
    assert result is None
    log_pass("No trail exit above stop")

    # Price drops below stop
    result = engine._update_trail(102.5, 1.0)  # stop=105-2=103, 102.5<=103
    assert result == 'TRAIL_EXIT_LONG'
    log_pass("Trail exit long triggered")

    # Short position: price falls then rises above stop
    engine.position = -1
    engine._reset_trail(100.0, 'short')

    result = engine._update_trail(95.0, 1.0)  # stop=95+2=97
    assert result is None
    assert engine.trail_low == 95.0
    log_pass("Trail updates low watermark for short")

    result = engine._update_trail(97.5, 1.0)  # stop=95+2=97, 97.5>=97
    assert result == 'TRAIL_EXIT_SHORT'
    log_pass("Trail exit short triggered")

    # Clear trail
    engine._clear_trail()
    assert engine.trail_active == False
    log_pass("Trail cleared")

    # No trail with mult=0
    engine.trail_mult = 0
    engine.position = 1
    engine._reset_trail(100.0, 'long')
    result = engine._update_trail(50.0, 1.0)
    assert result is None
    log_pass("Trail disabled when mult=0")


# ---------------------------------------------------------------------------
# Test: Score trade (win/loss tracking)
# ---------------------------------------------------------------------------

def test_score_trade():
    print("\n=== Score Trade ===")
    log_msgs = []
    engine = TradingEngine(on_log=lambda m: log_msgs.append(m))
    engine._save_trade_history = lambda: None  # don't write files
    # Reset counters (may be loaded from disk)
    engine.wins = 0
    engine.losses = 0
    engine.trade_history = []

    # Long win
    engine.entry_price = 100.0
    engine.entry_side = 'long'
    engine._score_trade(110.0)
    assert engine.wins == 1 and engine.losses == 0
    log_pass("Long win scored correctly")

    # Short win
    engine.entry_price = 100.0
    engine.entry_side = 'short'
    engine._score_trade(90.0)
    assert engine.wins == 2 and engine.losses == 0
    log_pass("Short win scored correctly")

    # Long loss
    engine.entry_price = 100.0
    engine.entry_side = 'long'
    engine._score_trade(95.0)
    assert engine.wins == 2 and engine.losses == 1
    log_pass("Long loss scored correctly")

    # Short loss
    engine.entry_price = 100.0
    engine.entry_side = 'short'
    engine._score_trade(105.0)
    assert engine.wins == 2 and engine.losses == 2
    log_pass("Short loss scored correctly")

    # No entry price (should be no-op)
    engine.entry_price = 0.0
    engine.entry_side = 'long'
    prev_w, prev_l = engine.wins, engine.losses
    engine._score_trade(100.0)
    assert engine.wins == prev_w and engine.losses == prev_l
    log_pass("Score trade with no entry_price is no-op")


# ---------------------------------------------------------------------------
# Test: Handle signal flow
# ---------------------------------------------------------------------------

def test_handle_signal():
    print("\n=== _handle_signal() ===")
    log_msgs = []
    engine = TradingEngine(on_log=lambda m: log_msgs.append(m))
    engine._save_trade_history = lambda: None
    engine.contract = SimpleNamespace(symbol='TEST', secType='STK')
    engine.quantity = 100
    engine.position = 0

    orders = []
    engine._place_order = lambda action, qty, tag='': orders.append(
        {'action': action, 'qty': qty, 'tag': tag})
    engine._get_ib_position = lambda: 0

    dt = datetime(2025, 6, 2, 10, 0, tzinfo=ET)

    # BUY signal from flat
    orders.clear()
    engine.position = 0
    engine._handle_signal('BUY', 100.0, dt)
    assert len(orders) == 1
    assert orders[0]['action'] == 'BUY'
    assert orders[0]['qty'] == 100
    log_pass("BUY from flat")

    # SELL signal from long (should flatten + entry)
    orders.clear()
    engine.position = 100
    engine.entry_price = 100.0
    engine.entry_side = 'long'
    engine._handle_signal('SELL', 105.0, dt)
    assert len(orders) == 2
    assert orders[0]['action'] == 'SELL' and orders[0]['tag'] == 'flatten long'
    assert orders[1]['action'] == 'SELL' and orders[1]['tag'] == 'entry short'
    log_pass("SELL from long (flatten + entry)")

    # BUY signal from short (should flatten + entry)
    orders.clear()
    engine.position = -100
    engine.entry_price = 105.0
    engine.entry_side = 'short'
    engine._handle_signal('BUY', 100.0, dt)
    assert len(orders) == 2
    assert orders[0]['action'] == 'BUY' and orders[0]['tag'] == 'flatten short'
    assert orders[1]['action'] == 'BUY' and orders[1]['tag'] == 'entry long'
    log_pass("BUY from short (flatten + entry)")

    # EXIT_LONG
    orders.clear()
    engine.position = 100
    engine.entry_price = 100.0
    engine.entry_side = 'long'
    engine._handle_signal('EXIT_LONG', 105.0, dt)
    assert len(orders) == 1
    assert orders[0]['action'] == 'SELL' and orders[0]['tag'] == 'exit long'
    log_pass("EXIT_LONG")

    # EXIT_SHORT
    orders.clear()
    engine.position = -100
    engine.entry_price = 100.0
    engine.entry_side = 'short'
    engine._handle_signal('EXIT_SHORT', 95.0, dt)
    assert len(orders) == 1
    assert orders[0]['action'] == 'BUY' and orders[0]['tag'] == 'exit short'
    log_pass("EXIT_SHORT")

    # TRAIL_EXIT_LONG
    orders.clear()
    engine.position = 100
    engine.entry_price = 100.0
    engine.entry_side = 'long'
    engine._handle_signal('TRAIL_EXIT_LONG', 95.0, dt)
    assert len(orders) == 1
    log_pass("TRAIL_EXIT_LONG")

    # TRAIL_EXIT_SHORT
    orders.clear()
    engine.position = -100
    engine.entry_price = 100.0
    engine.entry_side = 'short'
    engine._handle_signal('TRAIL_EXIT_SHORT', 105.0, dt)
    assert len(orders) == 1
    log_pass("TRAIL_EXIT_SHORT")

    # PT_EXIT_LONG
    orders.clear()
    engine.position = 100
    engine.entry_price = 100.0
    engine.entry_side = 'long'
    engine._handle_signal('PT_EXIT_LONG', 105.0, dt)
    assert len(orders) == 1
    assert orders[0]['action'] == 'SELL' and orders[0]['tag'] == 'profit target'
    log_pass("PT_EXIT_LONG")

    # PT_EXIT_SHORT
    orders.clear()
    engine.position = -100
    engine.entry_price = 100.0
    engine.entry_side = 'short'
    engine._handle_signal('PT_EXIT_SHORT', 95.0, dt)
    assert len(orders) == 1
    assert orders[0]['action'] == 'BUY' and orders[0]['tag'] == 'profit target'
    log_pass("PT_EXIT_SHORT")


# ---------------------------------------------------------------------------
# Test: ORB profit target computation
# ---------------------------------------------------------------------------

def test_orb_profit_target():
    print("\n=== ORB Profit Target ===")
    strat = OpeningRangeBreakoutStrategy(profit_mult=1.5)
    bars = make_orb_bars(200)
    result = strat.compute(bars)

    # profit_target key must exist
    assert 'profit_target' in result
    log_pass("ORB compute() returns 'profit_target' key")

    pt = result['profit_target']
    assert len(pt) == len(bars)
    log_pass("profit_target array has correct length")

    # Find breakout bar (first non-zero direction)
    dirs = result['direction']
    orb_h = result['orb_high']
    orb_l = result['orb_low']
    breakout_idx = None
    for i in range(len(dirs)):
        if dirs[i] != 0 and (i == 0 or dirs[i - 1] == 0):
            breakout_idx = i
            break

    if breakout_idx is not None:
        rh = orb_h[breakout_idx]
        rl = orb_l[breakout_idx]
        rw = rh - rl
        if dirs[breakout_idx] == 1:
            expected_pt = rh + 1.5 * rw
        else:
            expected_pt = rl - 1.5 * rw

        actual_pt = pt[breakout_idx]
        assert not np.isnan(actual_pt), "PT should be set at breakout bar"
        assert abs(actual_pt - expected_pt) < 1e-6, \
            f"PT={actual_pt:.4f} != expected={expected_pt:.4f}"
        log_pass(f"Profit target value correct at breakout (dir={dirs[breakout_idx]}, "
                 f"PT={actual_pt:.2f})")

        # PT should be carried forward after breakout
        valid_pts = pt[breakout_idx:]
        valid_count = int(np.count_nonzero(~np.isnan(valid_pts)))
        assert valid_count > 1
        log_pass(f"Profit target carried forward ({valid_count} bars)")
    else:
        log_fail("ORB profit target", "no breakout found in test data")

    # Engine profit target fields exist
    engine = TradingEngine(on_log=lambda m: None)
    assert hasattr(engine, 'profit_mult')
    assert hasattr(engine, 'profit_target')
    assert engine.profit_mult == 0.0
    assert engine.profit_target == 0.0
    log_pass("Engine has profit_mult and profit_target fields (default 0.0)")


# ---------------------------------------------------------------------------
# Test: No pending_order references remain
# ---------------------------------------------------------------------------

def test_no_pending_order():
    print("\n=== No pending_order ===")
    engine = TradingEngine(on_log=lambda m: None)
    if hasattr(engine, 'pending_order'):
        log_fail("pending_order removed", "attribute still exists on engine")
    else:
        log_pass("pending_order attribute removed from engine")

    # Check source code doesn't contain it
    import inspect
    src = inspect.getsource(TradingEngine)
    if 'pending_order' in src:
        log_fail("pending_order in source", "still referenced in TradingEngine source")
    else:
        log_pass("pending_order not in TradingEngine source")


# ---------------------------------------------------------------------------
# Test: Each strategy has correct attributes
# ---------------------------------------------------------------------------

def test_strategy_attributes():
    print("\n=== Strategy Attributes ===")
    for key, cls, _, _ in ALL_STRATEGIES:
        strat = cls()
        name = strat.name

        # Has required attributes
        for attr in ('name', 'min_bars', 'needs_subpanel', 'bar_size', 'duration'):
            if hasattr(strat, attr):
                log_pass(f"{name} has '{attr}'={getattr(strat, attr)}")
            else:
                log_fail(f"{name} has '{attr}'", "MISSING")

        # min_bars is positive
        if strat.min_bars > 0:
            log_pass(f"{name} min_bars={strat.min_bars} > 0")
        else:
            log_fail(f"{name} min_bars", f"min_bars={strat.min_bars} should be > 0")


# ---------------------------------------------------------------------------
# Test: Backtest simulation (full replay for each strategy)
# ---------------------------------------------------------------------------

def test_backtest_simulation():
    print("\n=== Backtest Simulation (full replay) ===")
    for key, cls, n_bars, bar_kwargs in ALL_STRATEGIES:
        strat = cls()
        name = strat.name

        if key == 'orb':
            bars = make_orb_bars(200)
        elif key == 'pivot_bounce':
            bars = make_multiday_bars(n_days=3, bars_per_day=130)
        else:
            bars = make_bars(n_bars or 200, **bar_kwargs)

        indicators = strat.compute(bars)
        dirs = indicators.get('trade_direction', indicators['direction'])
        atr_arr = indicators['atr']
        closes = np.array([b.close for b in bars], dtype=float)
        n = len(closes)
        min_bar = strat.min_bars
        trail_mult = 2.0

        pos = 0
        entry_price = 0.0
        trail_high = 0.0
        trail_low = float('inf')
        trades = []

        try:
            for i in range(min_bar, n):
                curr_close = closes[i]
                prev_dir = dirs[i - 1]
                curr_dir = dirs[i]
                curr_atr = atr_arr[i]

                if np.isnan(curr_atr):
                    continue

                # Trailing stop check
                if pos != 0 and trail_mult > 0 and curr_atr > 0:
                    trail_dist = trail_mult * curr_atr
                    if pos == 1:
                        if curr_close > trail_high:
                            trail_high = curr_close
                        trail_stop = trail_high - trail_dist
                        if curr_close <= trail_stop:
                            pnl = (curr_close - entry_price)
                            trades.append(pnl)
                            pos = 0
                            continue
                    elif pos == -1:
                        if curr_close < trail_low:
                            trail_low = curr_close
                        trail_stop = trail_low + trail_dist
                        if curr_close >= trail_stop:
                            pnl = (entry_price - curr_close)
                            trades.append(pnl)
                            pos = 0
                            continue

                flip_bull = prev_dir <= 0 and curr_dir > 0
                flip_bear = prev_dir >= 0 and curr_dir < 0

                if pos == 1 and curr_dir < 0:
                    trades.append(curr_close - entry_price)
                    pos = 0
                elif pos == -1 and curr_dir > 0:
                    trades.append(entry_price - curr_close)
                    pos = 0

                if pos == 0 and flip_bull:
                    pos = 1
                    entry_price = curr_close
                    trail_high = curr_close
                    trail_low = float('inf')
                elif pos == 0 and flip_bear:
                    pos = -1
                    entry_price = curr_close
                    trail_low = curr_close
                    trail_high = 0.0

            log_pass(f"{name} backtest: {len(trades)} trades, "
                     f"net={sum(trades):.2f}")
        except Exception as e:
            log_fail(f"{name} backtest simulation", f"Exception: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Test: _update_chart-like flow (indicator -> chart markers)
# ---------------------------------------------------------------------------

def test_chart_marker_generation():
    """Verify that direction flips produce correct marker signals."""
    print("\n=== Chart Marker Generation ===")
    for key, cls, n_bars, bar_kwargs in ALL_STRATEGIES:
        strat = cls()
        name = strat.name

        if key == 'orb':
            bars = make_orb_bars(200)
        elif key == 'pivot_bounce':
            bars = make_multiday_bars(n_days=3, bars_per_day=130)
        else:
            bars = make_bars(n_bars or 200, **bar_kwargs)

        indicators = strat.compute(bars)
        trade_dir = indicators.get('trade_direction', indicators['direction'])
        n = len(bars)
        start = max(0, n - 100)
        dirs = trade_dir[start:]

        buys = 0
        sells = 0
        for i in range(1, len(dirs)):
            if dirs[i - 1] <= 0 and dirs[i] > 0:
                buys += 1
            elif dirs[i - 1] >= 0 and dirs[i] < 0:
                sells += 1

        log_pass(f"{name} chart markers: {buys} BUY, {sells} SELL flips")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)  # Reproducible results

    print("=" * 65)
    print("  C10 AlgoTrader Smoke Test")
    print("=" * 65)

    test_registry()
    test_base_helpers()
    test_strategy_attributes()
    test_strategy_compute()
    test_strategy_plot()
    test_edge_cases()
    test_engine_signal_logic()
    test_entry_allowed()
    test_strategy_switching()
    test_trailing_stop()
    test_score_trade()
    test_handle_signal()
    test_orb_profit_target()
    test_no_pending_order()
    test_backtest_simulation()
    test_chart_marker_generation()

    print("\n" + "=" * 65)
    print(f"  RESULTS:  {PASS} passed,  {FAIL} failed")
    print("=" * 65)

    if FAIL > 0:
        sys.exit(1)
    else:
        print("\n  All tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
