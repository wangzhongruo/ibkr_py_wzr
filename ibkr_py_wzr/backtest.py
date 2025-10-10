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
        top_n: Optional[int] = None,
        bottom_n: Optional[int] = None,
        alpha_column: str = "alpha",
    ) -> BacktestResult:
        """Run ``strategy`` over the supplied ``data``.

        ``data`` may be a :class:`pandas.DataFrame` or a path to a parquet/
        pickle file previously stored via :class:`IBKRDataCenter`.  ``start`` and
        ``end`` allow the run to focus on a particular slice of the data.  When
        universe level data (multi-indexed by ``timestamp`` and ``symbol``) is
        supplied, ``top_n``/``bottom_n`` can be used to perform cross-sectional
        stock selection based on an ``alpha_column`` produced by the strategy.
        """
        price_data = self._prepare_data(data, start=start, end=end)
        if price_data.empty:
            raise ValueError("Data frame cannot be empty")
        if "close" not in price_data.columns:
            raise ValueError("Data frame must contain a 'close' column")
        price_data = price_data.copy()
        price_data["returns"] = self._compute_returns(price_data)

        if top_n is not None and top_n < 0:
            raise ValueError("top_n must be non-negative")
        if bottom_n is not None and bottom_n < 0:
            raise ValueError("bottom_n must be non-negative")

        signals = strategy.generate_signals(price_data)
        if "signal" not in signals.columns:
            raise ValueError("Strategy must return a 'signal' column")
        merged = self._merge_price_and_signals(
            price_data, signals, alpha_column=alpha_column
        )

        signal_series = self._forward_fill_signals(merged["signal"])
        if top_n is not None or bottom_n is not None:
            signal_series = self._apply_alpha_selection(
                signal_series,
                merged.get(alpha_column),
                top_n=top_n,
                bottom_n=bottom_n,
            )
        signal_series = self._apply_account_constraints(signal_series)

        position_series = self._shift_positions(signal_series)
        position_series = position_series.fillna(0)
        position_series = self._apply_account_constraints(position_series)

        per_trade_cost = self.commission_per_trade
        slippage = self.slippage_bps / 10_000

        trades = self._position_diff(position_series).abs().fillna(0)
        transaction_costs = trades * (per_trade_cost + slippage * merged["close"])

        portfolio_returns = self._compute_portfolio_returns(
            price_data["returns"],
            position_series,
            transaction_costs,
        )
        portfolio_returns = portfolio_returns.sort_index()
        equity_curve = (1 + portfolio_returns).cumprod() * self.initial_cash

        statistics = self._create_statistics(portfolio_returns, equity_curve)
        result_signals = signals.copy()
        result_signals["signal"] = signal_series
        result_signals["position"] = position_series
        return BacktestResult(
            strategy_name=strategy.name,
            account_type=self.account_type,
            signals=result_signals,
            equity_curve=equity_curve,
            statistics=statistics,
        )

    def run_many(
        self,
        data: Union[pd.DataFrame, str, Path],
        strategies: Iterable[Strategy],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        top_n: Optional[int] = None,
        bottom_n: Optional[int] = None,
        alpha_column: str = "alpha",
    ) -> List[BacktestResult]:
        return [
            self.run(
                data,
                strategy,
                start=start,
                end=end,
                top_n=top_n,
                bottom_n=bottom_n,
                alpha_column=alpha_column,
            )
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
        self, portfolio_returns: pd.Series, equity_curve: pd.Series
    ) -> Dict[str, float]:
        portfolio_returns = portfolio_returns.fillna(0.0)
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

        annualized_return = (1 + total_return) ** (
            TRADING_MINUTES_PER_YEAR / max(len(portfolio_returns), 1)
        ) - 1

        if portfolio_returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (
                portfolio_returns.mean()
                / portfolio_returns.std()
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
        prepared.sort_index(inplace=True)

        if isinstance(prepared.index, pd.MultiIndex):
            timestamp_level = self._timestamp_level(prepared.index)
            timestamps = pd.DatetimeIndex(
                prepared.index.get_level_values(timestamp_level)
            )
            tz = timestamps.tz
            start_ts = _coerce_timestamp(start, tz)
            end_ts = _coerce_timestamp(end, tz)
            mask = pd.Series(True, index=prepared.index)
            if start_ts is not None:
                mask &= pd.Series(
                    timestamps >= start_ts, index=prepared.index
                )
            if end_ts is not None:
                mask &= pd.Series(
                    timestamps <= end_ts, index=prepared.index
                )
            prepared = prepared[mask]
            return prepared

        if not isinstance(prepared.index, pd.DatetimeIndex):
            raise ValueError("Data frame index must be a DatetimeIndex")

        tz = prepared.index.tz
        start_ts = _coerce_timestamp(start, tz)
        end_ts = _coerce_timestamp(end, tz)
        if start_ts is not None:
            prepared = prepared[prepared.index >= start_ts]
        if end_ts is not None:
            prepared = prepared[prepared.index <= end_ts]

        return prepared

    @staticmethod
    def _timestamp_level(index: pd.MultiIndex) -> int:
        names = list(index.names)
        if "timestamp" in names:
            return names.index("timestamp")
        raise ValueError("MultiIndex must include a 'timestamp' level")

    @staticmethod
    def _symbol_level(index: pd.MultiIndex) -> int:
        names = list(index.names)
        if "symbol" in names:
            return names.index("symbol")
        raise ValueError("MultiIndex must include a 'symbol' level")

    def _compute_returns(self, price_data: pd.DataFrame) -> pd.Series:
        if isinstance(price_data.index, pd.MultiIndex):
            symbol_level = self._symbol_level(price_data.index)
            grouped = price_data.groupby(level=symbol_level)
            return grouped["close"].pct_change().fillna(0.0)
        return price_data["close"].pct_change().fillna(0.0)

    def _merge_price_and_signals(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        *,
        alpha_column: str,
    ) -> pd.DataFrame:
        columns_to_include: List[str] = ["signal"]
        if alpha_column in signals.columns:
            columns_to_include.append(alpha_column)
        extra_columns = [
            col
            for col in signals.columns
            if col not in price_data.columns and col not in columns_to_include
        ]
        columns_to_include.extend(extra_columns)
        columns_to_include = list(dict.fromkeys(columns_to_include))
        return price_data.join(signals[columns_to_include], how="left")

    def _forward_fill_signals(self, series: pd.Series) -> pd.Series:
        if isinstance(series.index, pd.MultiIndex):
            symbol_level = self._symbol_level(series.index)
            return (
                series.groupby(level=symbol_level).ffill().fillna(0).astype(float)
            )
        return series.ffill().fillna(0).astype(float)

    def _shift_positions(self, series: pd.Series) -> pd.Series:
        if isinstance(series.index, pd.MultiIndex):
            symbol_level = self._symbol_level(series.index)
            return series.groupby(level=symbol_level).shift(1)
        return series.shift(1)

    def _position_diff(self, series: pd.Series) -> pd.Series:
        if isinstance(series.index, pd.MultiIndex):
            symbol_level = self._symbol_level(series.index)
            return series.groupby(level=symbol_level).diff()
        return series.diff()

    def _apply_alpha_selection(
        self,
        signal_series: pd.Series,
        alpha_series: Optional[pd.Series],
        *,
        top_n: Optional[int],
        bottom_n: Optional[int],
    ) -> pd.Series:
        if alpha_series is None:
            raise ValueError(
                "Alpha selection requested but the strategy did not produce the"
                " required alpha column"
            )
        if not isinstance(signal_series.index, pd.MultiIndex):
            LOGGER.warning(
                "Alpha selection requested but data is single asset; ignoring"
            )
            return signal_series

        timestamp_level = self._timestamp_level(signal_series.index)

        def build_mask(group: pd.Series) -> pd.Series:
            mask = pd.Series(False, index=group.index)
            if top_n:
                top_index = group.nlargest(top_n).index
                mask.loc[top_index] = True
            if bottom_n:
                bottom_index = group.nsmallest(bottom_n).index
                mask.loc[bottom_index] = True
            return mask

        selection_mask = alpha_series.groupby(
            level=timestamp_level, group_keys=False
        ).apply(build_mask)
        return signal_series.where(selection_mask, 0)

    def _compute_portfolio_returns(
        self,
        returns: pd.Series,
        positions: pd.Series,
        transaction_costs: pd.Series,
    ) -> pd.Series:
        if isinstance(returns.index, pd.MultiIndex):
            timestamp_level = self._timestamp_level(returns.index)
            abs_positions = positions.abs()
            gross = abs_positions.groupby(level=timestamp_level).transform("sum")
            gross = gross.replace(0, 1)
            weights = positions / gross
            per_symbol_returns = (
                weights * returns - transaction_costs / self.initial_cash
            )
            return per_symbol_returns.groupby(level=timestamp_level).sum()

        strategy_returns = (
            positions * returns - transaction_costs / self.initial_cash
        )
        return strategy_returns


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
        names = list(df.index.names)
        if "timestamp" not in names or "symbol" not in names:
            raise ValueError(
                "Loaded data must be indexed by 'timestamp' and 'symbol'"
            )
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index(names, inplace=True)
    else:
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
    if isinstance(df.index, pd.MultiIndex):
        timestamp_level = df.index.names.index("timestamp")
        timestamps = pd.DatetimeIndex(df.index.get_level_values(timestamp_level))
        tz = timestamps.tz
        start_ts = _coerce_timestamp(start, tz)
        end_ts = _coerce_timestamp(end, tz)
        mask = pd.Series(True, index=df.index)
        if start_ts is not None:
            mask &= pd.Series(timestamps >= start_ts, index=df.index)
        if end_ts is not None:
            mask &= pd.Series(timestamps <= end_ts, index=df.index)
        df = df[mask]
    else:
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
