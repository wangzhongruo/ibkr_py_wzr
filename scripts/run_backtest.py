#!/usr/bin/env python3
"""Command line helper for running quick backtests."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from ibkr_py_wzr import Backtester, IBKRDataCenter
from ibkr_py_wzr.strategies.moving_average import MovingAverageCrossStrategy

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", help="Ticker symbol e.g. AAPL")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Additional tickers to include in the universe download",
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        help="Path to a newline separated list of tickers",
    )
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
        "--bar-size",
        default="1 min",
        help="Bar size for historical data (default: 1 min)",
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
        "--account-type",
        default="MARGIN",
        help="IBKR account type to emulate (e.g. MARGIN, CASH)",
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
    parser.add_argument(
        "--top-n",
        type=int,
        help="Number of highest alpha symbols to hold long at each bar",
    )
    parser.add_argument(
        "--bottom-n",
        type=int,
        help="Number of lowest alpha symbols to short at each bar",
    )
    parser.add_argument(
        "--alpha-column",
        default="alpha",
        help="Alpha column produced by the strategy used for selection",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    start_input = _parse_timestamp(args.start)
    end_input = _parse_timestamp(args.end)

    symbols = _resolve_symbols(args)

    data_source: Path | pd.DataFrame
    if args.data_path is not None:
        if not args.data_path.exists():
            LOGGER.error("Data path %s does not exist", args.data_path)
            return
        data_source = args.data_path
    else:
        if not symbols:
            LOGGER.error(
                "No symbols supplied. Provide a symbol/universe or --data-path."
            )
            return
        data_center = IBKRDataCenter()
        data_center.connect()
        try:
            start_request = _normalise_request_timestamp(
                start_input, data_center.timezone
            )
            end_request = _normalise_request_timestamp(
                end_input, data_center.timezone
            )
            if len(symbols) == 1:
                df = data_center.download_intraday_bars(
                    symbols[0],
                    duration=args.duration,
                    start_datetime=start_request,
                    end_datetime=end_request,
                    bar_size=args.bar_size,
                    save_to=args.save,
                )
            else:
                df = data_center.download_equity_universe(
                    symbols,
                    duration=args.duration,
                    start_datetime=start_request,
                    end_datetime=end_request,
                    bar_size=args.bar_size,
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

    backtester = Backtester(
        initial_cash=args.initial_cash,
        account_type=args.account_type,
    )
    result = backtester.run(
        data_source,
        strategy,
        start=start_input,
        end=end_input,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        alpha_column=args.alpha_column,
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


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    symbols: List[str] = []
    if args.symbol:
        symbols.append(args.symbol)
    if args.symbols:
        symbols.extend(args.symbols)
    if args.universe_file:
        if not args.universe_file.exists():
            LOGGER.error("Universe file %s does not exist", args.universe_file)
        else:
            symbols.extend(_read_universe_file(args.universe_file))

    seen: set[str] = set()
    ordered: List[str] = []
    for ticker in symbols:
        normalized = ticker.strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _read_universe_file(path: Path) -> List[str]:
    tickers: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            tickers.append(cleaned)
    return tickers


if __name__ == "__main__":
    main()
