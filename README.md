# ibkr_py_wzr

Utilities for downloading data, backtesting strategies and running live trading
sessions with Interactive Brokers.  The package is built on top of
[`ib_insync`](https://ib-insync.readthedocs.io/) and is organised into three
major parts:

1. **Data centre** – `IBKRDataCenter` wraps `ib_insync.IB` to fetch 1 minute US
   equity data for individual tickers or larger universes of stocks.
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
from datetime import datetime
from zoneinfo import ZoneInfo

from ibkr_py_wzr import IBKRDataCenter

data_center = IBKRDataCenter()
data_center.connect()

# Download the last five days of one minute bars for Apple and persist to disk.
df = data_center.download_intraday_bars(
    "AAPL",
    duration="5 D",
    save_to=Path("data/aapl_1m.parquet"),
)

# Alternatively, specify explicit start/end times and a different bar size.
df_hourly = data_center.download_intraday_bars(
    "AAPL",
    start_datetime=datetime(2024, 1, 1, tzinfo=ZoneInfo("America/New_York")),
    end_datetime=datetime(2024, 1, 31, tzinfo=ZoneInfo("America/New_York")),
    bar_size="1 hour",
)

# Pull the same window for a small universe and store it as a parquet dataset.
universe = data_center.download_equity_universe(
    ["AAPL", "MSFT", "NVDA", "AMZN"],
    start_datetime=datetime(2024, 1, 1, tzinfo=ZoneInfo("America/New_York")),
    end_datetime=datetime(2024, 1, 31, tzinfo=ZoneInfo("America/New_York")),
    bar_size="1 hour",
    save_to=Path("data/mega_caps_hourly.parquet"),
)

data_center.disconnect()
```

The returned `DataFrame` is indexed by timezone aware timestamps (New York
timezone by default) and contains the standard OHLCV fields produced by
Interactive Brokers: `open`, `high`, `low`, `close`, `volume`, `bar_count` and
`average_price`.  Universe downloads are indexed by both `timestamp` and
`symbol` so that they can be fed directly into the multi-asset backtester.
Pass a `Path` to `save_to` if you would like to persist the results as parquet
or pickle for efficient reuse.

The convenience CLI (`python -m ibkr_py_wzr.data_center`) also accepts a YAML
configuration file so that large universes can be refreshed without typing long
command lines.  Create a file such as:

```yaml
data_center:
  host: 127.0.0.1
  port: 7497
  client_id: 7
  timezone: America/New_York
  use_rth: true
  data_directory: data/nasdaq
  nasdaq_listing_url: https://ftp.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt
  nasdaq_listing_fallback_urls:
    - https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt
  # Optional HTTP tuning for the NASDAQ listings download
  nasdaq_listing_timeout: 45
  nasdaq_listing_retries: 5
  nasdaq_listing_retry_backoff: 10
update:
  bar_size: "1 min"
  what_to_show: TRADES
  start_date: 2011-01-01
  max_workers: 1
  throttle_seconds: 0.25
  progress: true
```

and run:

```bash
python -m ibkr_py_wzr.data_center --config configs/nasdaq.yaml
```

The `data_directory` in the configuration acts as the canonical cache.  When the
NASDAQ updater runs it first inspects the per-symbol parquet files stored in
that directory.  If the existing history already spans the configured start
date through the latest requested session the downloader performs an incremental
update by appending only the newly available bars.  If the cache is missing or
does not yet cover the requested period a full refresh is triggered and the
parquet file is replaced with a fresh download from Interactive Brokers.

Any CLI flag provided alongside `--config` overrides the value stored in the
file, making it straightforward to reuse defaults while experimenting with
different backfill windows or data directories.  Progress bars are enabled by
default when [`tqdm`](https://tqdm.github.io/) is installed; pass
`--no-progress` on the command line or set `progress: false` in the YAML file to
fall back to plain logging.  Increase `max_workers` to spread the download
across several IBKR connections (each worker automatically increments the
configured `client_id` so that the sessions remain unique) when your pacing
limits and hardware allow.

If your environment occasionally times out while retrieving the official
NASDAQ listings feed the YAML values above can be adjusted to increase the
request timeout, retry count or backoff delay.  The downloader now retries
failed requests automatically and falls back to the alternative HTTPS URL when
`ftp.nasdaqtrader.com` is unreachable before raising a descriptive error after
the final attempt.  This ensures transient network issues no longer terminate
the entire universe refresh without context.

> **Note**
> The historical bars returned by Interactive Brokers reflect the market data
> permissions available to your account.  If you subscribe to the live NASDAQ
> feed the dataset contains real-time bars.  Accounts without live access receive
> the delayed stream (typically 15 minutes) instead, even though the download
> workflow remains identical.

Saved files can be reloaded later without hitting the IBKR API:

```python
from ibkr_py_wzr import load_market_data

df = load_market_data(
    "data/aapl_1m.parquet",
    start="2024-01-10 09:30",
    end="2024-01-20 16:00",
)
```

### Running a backtest

```bash
python scripts/run_backtest.py AAPL --duration "10 D" --fast 10 --slow 30 \
    --account-type MARGIN
```

The script prints a summary table with total return, annualised return, Sharpe
ratio and maximum drawdown.  Supply `--account-type CASH` to prevent short
positions when modelling a cash or retirement style account.  If you already
have parquet/pickle data on disk you can run a backtest directly from it and
select a specific date range:

```bash
python scripts/run_backtest.py AAPL --data-path data/aapl_1m.parquet \
    --start "2024-01-10 09:30" --end "2024-01-20 16:00"
```

Multi-asset research is handled by passing several tickers (or a universe file)
and specifying how many of the highest/lowest alpha names to hold.  The bundled
moving average crossover strategy now produces an `alpha` column (fast minus
slow moving average) that drives the selection process:

```bash
python scripts/run_backtest.py --symbols AAPL MSFT NVDA AMZN --duration "30 D" \
    --fast 20 --slow 60 --top-n 2 --bottom-n 1 --bar-size "1 hour"
```

At each bar the backtester ranks the universe by the chosen `--alpha-column`,
goes long the `--top-n` symbols, shorts the `--bottom-n` names (if the account
type allows shorting) and scales weights so that the portfolio remains fully
invested.  Provide a parquet file generated by `download_equity_universe` to
reuse previously saved universes without re-downloading.

### Live trading

```bash
python scripts/run_live_trading.py AAPL --quantity 10 --fast 10 --slow 30 \
    --account-type MARGIN
```

The example strategy polls IBKR every minute and places market orders when the
fast moving average crosses the slow moving average.  Set
`--account-type CASH` to automatically suppress short signals for cash or IRA
accounts.  Commissions reported by Interactive Brokers are captured
automatically once fills are confirmed, so no manual fee configuration is
required.

### Commission fees from the IB API

`IBKRTradingService` records the `commission` and `currency` fields from the
`CommissionReport` objects supplied by Interactive Brokers once a trade is
filled.  IBKR reports fees as negative values (debits) and rebates as positive
values, already aggregated across exchange, regulatory and broker components.
No manual inputs are needed—each order log entry includes the raw numbers
returned by the API alongside the order identifier.

## Extending the framework

Implement custom strategies by adhering to the `Strategy` protocol defined in
`ibkr_py_wzr/strategies/base.py`.  The same strategy can be reused in both the
backtester and the live trading service.
