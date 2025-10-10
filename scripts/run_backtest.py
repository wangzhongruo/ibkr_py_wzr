#!/usr/bin/env python3
"""Command line helper for running quick backtests."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ibkr_py_wzr import Backtester, IBKRDataCenter
from ibkr_py_wzr.strategies.moving_average import MovingAverageCrossStrategy

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="Ticker symbol e.g. AAPL")
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
        help="Optional path to store the downloaded data as parquet",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    data_center = IBKRDataCenter()
    data_center.connect()
    try:
        df = data_center.download_intraday_bars(
            args.symbol,
            duration=args.duration,
            save_to=args.save,
        )
    finally:
        data_center.disconnect()

    if df.empty:
        LOGGER.error("No data downloaded, aborting")
        return

    strategy = MovingAverageCrossStrategy(
        name=f"MA({args.fast},{args.slow})",
        fast_window=args.fast,
        slow_window=args.slow,
    )

    backtester = Backtester(initial_cash=args.initial_cash)
    result = backtester.run(df, strategy)

    LOGGER.info("Backtest statistics: %s", result.statistics)
    print("Strategy:", result.strategy_name)
    for key, value in result.statistics.items():
        print(f"{key:>20}: {value: .4f}")


if __name__ == "__main__":
    main()
