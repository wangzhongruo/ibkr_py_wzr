"""Backtesting utilities for Interactive Brokers sourced data."""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import datetime, tzinfo as dt_tzinfo
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd

from .strategies.base import Strategy

TRADING_MINUTES_PER_YEAR = 252 * 390

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container describing the outcome of a backtest run."""

    strategy_name: str
    account_type: str
    signals: pd.DataFrame
    equity_curve: pd.Series
    statistics: Dict[str, float]

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return {
            "strategy": self.strategy_name,
            "account_type": self.account_type,
            **self.statistics,
        }


class Backtester:
    """Simple vectorised backtester.

    Parameters
    ----------
    initial_cash:
        Starting capital of the strategy.
    commission_per_trade:
        Flat commission applied each time the strategy changes its position.
    slippage_bps:
        Round trip slippage cost in basis points applied on fills.
    account_type:
        Simulated IBKR account type (e.g. ``MARGIN`` or ``CASH``).  Cash
        accounts are prevented from taking short positions.
    """

    def __init__(
        self,
        initial_cash: float = 100_000,
        commission_per_trade: float = 0.0,
        slippage_bps: float = 0.0,
        account_type: str = "MARGIN",
    ) -> None:
        self.initial_cash = initial_cash
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.account_type = account_type.upper()
        self._supports_shorting = self.account_type not in {"CASH", "IRA"}

    def run(
        self,
        data: Union[pd.DataFrame, str, Path],
        strategy: Strategy,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> BacktestResult:
        """Run ``strategy`` over the supplied ``data``.

        ``data`` may be a :class:`pandas.DataFrame` or a path to a parquet/
        pickle file previously stored via :class:`IBKRDataCenter`.  ``start`` and
        ``end`` allow the run to focus on a particular slice of the data.
        """
        price_data = self._prepare_data(data, start=start, end=end)
        if price_data.empty:
            raise ValueError("Data frame cannot be empty")
        if "close" not in price_data.columns:
            raise ValueError("Data frame must contain a 'close' column")
        price_data = price_data.copy()
        price_data["returns"] = price_data["close"].pct_change().fillna(0.0)

        signals = strategy.generate_signals(price_data)
        if "signal" not in signals.columns:
            raise ValueError("Strategy must return a 'signal' column")
        merged = price_data.join(signals[["signal"]], how="left")
        merged["signal"] = merged["signal"].ffill().fillna(0)
        merged["signal"] = self._apply_account_constraints(merged["signal"])
        merged["position"] = merged["signal"].shift(1).fillna(0)
        merged["position"] = self._apply_account_constraints(merged["position"])

        per_trade_cost = self.commission_per_trade
        slippage = self.slippage_bps / 10_000

        trades = merged["position"].diff().abs().fillna(0)
        transaction_costs = trades * (per_trade_cost + slippage * merged["close"])
        merged["strategy_return"] = (
            merged["position"] * merged["returns"] - transaction_costs / self.initial_cash
        )

        equity_curve = (1 + merged["strategy_return"]).cumprod() * self.initial_cash

        statistics = self._create_statistics(merged, equity_curve)
        return BacktestResult(
            strategy_name=strategy.name,
            account_type=self.account_type,
            signals=signals,
            equity_curve=equity_curve,
            statistics=statistics,
        )

    def run_many(
        self,
        data: Union[pd.DataFrame, str, Path],
        strategies: Iterable[Strategy],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[BacktestResult]:
        return [
            self.run(data, strategy, start=start, end=end)
            for strategy in strategies
        ]

    def _apply_account_constraints(self, series: pd.Series) -> pd.Series:
        """Clip unsupported positions based on ``account_type``."""

        if self._supports_shorting:
            return series

        if (series < 0).any():
            LOGGER.info(
                "Clipping short positions for cash account backtest (account type %s)",
                self.account_type,
            )
        return series.clip(lower=0)

    def _create_statistics(
        self, merged: pd.DataFrame, equity_curve: pd.Series
    ) -> Dict[str, float]:
        strategy_returns = merged["strategy_return"].fillna(0.0)
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

        annualized_return = (1 + total_return) ** (
            TRADING_MINUTES_PER_YEAR / max(len(merged), 1)
        ) - 1

        if strategy_returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (
                strategy_returns.mean()
                / strategy_returns.std()
                * math.sqrt(TRADING_MINUTES_PER_YEAR)
            )

        drawdown = equity_curve / equity_curve.cummax() - 1
        max_drawdown = drawdown.min()

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
        }

    def _prepare_data(
        self,
        data: Union[pd.DataFrame, str, Path],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> pd.DataFrame:
        if isinstance(data, (str, Path)):
            return load_market_data(data, start=start, end=end)

        prepared = data.copy()
        if not isinstance(prepared.index, pd.DatetimeIndex):
            raise ValueError("Data frame index must be a DatetimeIndex")
        prepared.sort_index(inplace=True)

        tz = prepared.index.tz
        start_ts = _coerce_timestamp(start, tz)
        end_ts = _coerce_timestamp(end, tz)
        if start_ts is not None:
            prepared = prepared[prepared.index >= start_ts]
        if end_ts is not None:
            prepared = prepared[prepared.index <= end_ts]

        return prepared


def load_market_data(
    source: Union[str, Path],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load market data previously saved by :class:`IBKRDataCenter`.

    Parameters
    ----------
    source:
        Path to a parquet or pickle file generated by ``IBKRDataCenter``.
    start, end:
        Optional datetimes used to slice the returned data set.  Naive
        datetimes are assumed to be in the data's timezone.

    Returns
    -------
    pandas.DataFrame
        Time indexed OHLCV data filtered to the requested range.
    """

    path = Path(source)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    else:
        raise ValueError(
            "source must point to a .parquet, .pq, .pkl or .pickle file"
        )

    if isinstance(df.index, pd.MultiIndex):
        raise ValueError("Loaded data must have a one-dimensional index")

    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df.set_index("date", inplace=True)
        else:
            raise ValueError(
                "Loaded data must have a DatetimeIndex or a 'date' column"
            )

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df.sort_index(inplace=True)
    tz = df.index.tz
    start_ts = _coerce_timestamp(start, tz)
    end_ts = _coerce_timestamp(end, tz)
    if start_ts is not None:
        df = df[df.index >= start_ts]
    if end_ts is not None:
        df = df[df.index <= end_ts]

    return df


def _coerce_timestamp(
    value: Optional[datetime], tz: Optional[dt_tzinfo]
) -> Optional[pd.Timestamp]:
    if value is None:
        return None

    timestamp = pd.Timestamp(value)
    if tz is None:
        return timestamp

    if timestamp.tzinfo is None:
        return timestamp.tz_localize(tz)

    return timestamp.tz_convert(tz)
