#!/usr/bin/env python3
"""Command line helper for running quick backtests."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ibkr_py_wzr import Backtester, IBKRDataCenter
from ibkr_py_wzr.strategies.moving_average import MovingAverageCrossStrategy

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="Ticker symbol e.g. AAPL")
    parser.add_argument(
        "--data-path",
        type=Path,
        help=(
            "Path to a parquet/pickle data set downloaded via IBKRDataCenter. "
            "When provided no download is performed."
        ),
    )
    parser.add_argument(
        "--duration",
        default="5 D",
        help="Historical duration to download (default: 5 D)",
    )
    parser.add_argument(
        "--fast",
        type=int,
        default=20,
        help="Fast moving average window (default: 20)",
    )
    parser.add_argument(
        "--slow",
        type=int,
        default=50,
        help="Slow moving average window (default: 50)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000,
        help="Initial capital for the backtest",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to store the downloaded data as parquet/pickle",
    )
    parser.add_argument(
        "--start",
        help="Optional start datetime for the backtest (e.g. '2024-01-10 09:30')",
    )
    parser.add_argument(
        "--end",
        help="Optional end datetime for the backtest (e.g. '2024-01-20 16:00')",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    start_input = _parse_timestamp(args.start)
    end_input = _parse_timestamp(args.end)

    data_source: Path | pd.DataFrame
    if args.data_path is not None:
        if not args.data_path.exists():
            LOGGER.error("Data path %s does not exist", args.data_path)
            return
        data_source = args.data_path
    else:
        data_center = IBKRDataCenter()
        data_center.connect()
        try:
            df = data_center.download_intraday_bars(
                args.symbol,
                duration=args.duration,
                start_datetime=_normalise_request_timestamp(
                    start_input, data_center.timezone
                ),
                end_datetime=_normalise_request_timestamp(
                    end_input, data_center.timezone
                ),
                save_to=args.save,
            )
        finally:
            data_center.disconnect()

        if df.empty:
            LOGGER.error("No data downloaded, aborting")
            return

        data_source = df

    strategy = MovingAverageCrossStrategy(
        name=f"MA({args.fast},{args.slow})",
        fast_window=args.fast,
        slow_window=args.slow,
    )

    backtester = Backtester(initial_cash=args.initial_cash)
    result = backtester.run(
        data_source,
        strategy,
        start=start_input,
        end=end_input,
    )

    LOGGER.info("Backtest statistics: %s", result.statistics)
    print("Strategy:", result.strategy_name)
    for key, value in result.statistics.items():
        print(f"{key:>20}: {value: .4f}")


def _parse_timestamp(raw: str | None) -> pd.Timestamp | None:
    if raw is None:
        return None
    return pd.Timestamp(raw)


def _normalise_request_timestamp(
    ts: pd.Timestamp | None, timezone: str
) -> datetime | None:
    if ts is None:
        return None

    localized = ts
    if localized.tzinfo is None:
        localized = localized.tz_localize(timezone)
    else:
        localized = localized.tz_convert(timezone)

    return localized.to_pydatetime()


if __name__ == "__main__":
    main()
