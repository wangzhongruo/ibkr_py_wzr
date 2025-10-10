#!/usr/bin/env python3
"""Launch a live trading session using a moving average crossover strategy."""
from __future__ import annotations

import argparse
import asyncio
import logging

from ibkr_py_wzr import IBKRDataCenter, IBKRTradingService
from ibkr_py_wzr.strategies.moving_average import MovingAverageCrossStrategy

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="Ticker symbol e.g. AAPL")
    parser.add_argument(
        "--quantity",
        type=int,
        default=100,
        help="Quantity to trade when the signal changes",
    )
    parser.add_argument(
        "--fast",
        type=int,
        default=20,
        help="Fast moving average window",
    )
    parser.add_argument(
        "--slow",
        type=int,
        default=50,
        help="Slow moving average window",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between polling IBKR for new data",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Number of minutes of history to provide to the strategy",
    )
    return parser.parse_args()


def build_strategy(args: argparse.Namespace) -> MovingAverageCrossStrategy:
    return MovingAverageCrossStrategy(
        name=f"MA({args.fast},{args.slow})",
        fast_window=args.fast,
        slow_window=args.slow,
    )


async def main_async() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    data_center = IBKRDataCenter()
    trading_service = IBKRTradingService(
        data_center=data_center,
        poll_interval=args.poll_interval,
        lookback_minutes=args.lookback,
    )

    strategy = build_strategy(args)

    trading_service.connect()
    LOGGER.info("Starting live trading for %s with %s", args.symbol, strategy.name)

    try:
        await trading_service.run_live_strategy(
            symbol=args.symbol,
            strategy=strategy,
            quantity=args.quantity,
        )
    except asyncio.CancelledError:  # pragma: no cover - runtime control flow
        LOGGER.info("Live trading cancelled")
    except KeyboardInterrupt:  # pragma: no cover - runtime control flow
        LOGGER.info("Received Ctrl+C, stopping")
    finally:
        trading_service.disconnect()


if __name__ == "__main__":
    asyncio.run(main_async())
