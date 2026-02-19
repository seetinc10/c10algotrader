# C10 AlgoTrader (IBKR TWS)

Desktop algorithmic trading app for IBKR paper/live trading.

This project provides:
- A Tkinter-based GUI (`app.py`)
- A shared trading engine with order/risk controls
- Multiple pluggable strategies (not just Supertrend)
- A comprehensive smoke test suite (`smoke_test.py`)

## Requirements

- Python 3.10+
- IBKR Trader Workstation (TWS) or IB Gateway
- API access enabled in TWS/Gateway

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## IBKR Setup (TWS)

1. Open TWS and log in (paper account recommended for testing).
2. Go to `File > Global Configuration > API > Settings`.
3. Enable socket/API clients.
4. Use port `7497` for paper trading (`7496` is typically live).
5. Keep localhost access enabled.
6. Disable read-only mode if you want order placement.

## Run

```bash
python app.py
```

## Basic Workflow

1. Enter a symbol (example: `AAPL`).
2. Select a strategy from the strategy dropdown.
3. Configure quantity/risk options in the UI.
4. Click `Start` to connect, load bars, and begin evaluation.
5. Use `Pause` to halt new signal handling temporarily.
6. Use `Stop` to flatten/cancel and disconnect safely.

## Included Strategies

Current registry includes:

- `orb` (Opening Range Breakout)
- `squeeze`
- `vwap`
- `supertrend`
- `macd`
- `rsi`
- `dual_ma`
- `donchian`
- `bollinger`
- `atr_trail`
- `obv`
- `ichimoku`
- `keltner_adx`
- `zscore`
- `roc_div`
- `triple_screen`
- `stoch_rsi_vwap`
- `vol_breakout`
- `atr_channel`
- `pivot_bounce`
- `inside_bar`
- `heikin_ashi`
- `linreg`

## Testing

Run the smoke test suite:

```bash
python smoke_test.py
```

The smoke test validates:
- Strategy registry integrity
- Strategy compute/plot interfaces
- Engine signal handling and state transitions
- Trailing stop/profit-target behavior
- Backtest-style replay sanity checks

## Project Files

- `app.py`: GUI, trading engine, and strategy implementations
- `smoke_test.py`: broad regression/sanity tests
- `requirements.txt`: Python dependencies
- `trade_history.json`: persisted trade outcome history

## Safety Notes

- Start in paper trading.
- Verify symbol permissions and market data subscriptions.
- Keep position size small while validating behavior.
- Confirm TWS API permissions before live order testing.
