# ibkr_py_wzr

Utilities for downloading data, backtesting strategies and running live trading
sessions with Interactive Brokers.  The package is built on top of
[`ib_insync`](https://ib-insync.readthedocs.io/) and is organised into three
major parts:

1. **Data centre** – `IBKRDataCenter` wraps `ib_insync.IB` to fetch 1 minute US
   equity data.
2. **Backtester** – `Backtester` runs vectorised simulations of strategies that
   implement the simple `Strategy` protocol.  An example moving average cross
   strategy is bundled with the project.
3. **Online trading** – `IBKRTradingService` polls IBKR for new data and submits
   market orders whenever the supplied strategy's signal changes.

## Installation

Create a virtual environment and install the dependencies listed in
`requirements.txt` (not provided here).  The code expects `ib_insync`, `pandas`
and `pyarrow` (for parquet export) to be available.

```bash
pip install ib-insync pandas pyarrow
```

## Usage

### Downloading historical data

```python
from pathlib import Path

from ibkr_py_wzr import IBKRDataCenter

data_center = IBKRDataCenter()
data_center.connect()

# Download the last five days of one minute bars for Apple and persist to disk.
df = data_center.download_intraday_bars(
    "AAPL",
    duration="5 D",
    save_to=Path("data/aapl_1m.parquet"),
)

data_center.disconnect()
```

The returned `DataFrame` is indexed by timezone aware timestamps (New York
timezone by default) and contains the standard OHLCV fields produced by
Interactive Brokers: `open`, `high`, `low`, `close`, `volume`, `bar_count` and
`average_price`.  Pass a `Path` to `save_to` if you would like to persist it as
CSV or parquet.

### Running a backtest

```bash
python scripts/run_backtest.py AAPL --duration "10 D" --fast 10 --slow 30
```

The script prints a summary table with total return, annualised return, Sharpe
ratio and maximum drawdown.

### Live trading

```bash
python scripts/run_live_trading.py AAPL --quantity 10 --fast 10 --slow 30
```

The example strategy polls IBKR every minute and places market orders when the
fast moving average crosses the slow moving average.

## Extending the framework

Implement custom strategies by adhering to the `Strategy` protocol defined in
`ibkr_py_wzr/strategies/base.py`.  The same strategy can be reused in both the
backtester and the live trading service.
