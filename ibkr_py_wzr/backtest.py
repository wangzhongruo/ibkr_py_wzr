"""Backtesting utilities for Interactive Brokers sourced data."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

from .strategies.base import Strategy

TRADING_MINUTES_PER_YEAR = 252 * 390


@dataclass
class BacktestResult:
    """Container describing the outcome of a backtest run."""

    strategy_name: str
    signals: pd.DataFrame
    equity_curve: pd.Series
    statistics: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        return {"strategy": self.strategy_name, **self.statistics}


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
    """

    def __init__(
        self,
        initial_cash: float = 100_000,
        commission_per_trade: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> None:
        self.initial_cash = initial_cash
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps

    def run(self, data: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        if data.empty:
            raise ValueError("Data frame cannot be empty")
        if "close" not in data.columns:
            raise ValueError("Data frame must contain a 'close' column")

        price_data = data.copy()
        price_data["returns"] = price_data["close"].pct_change().fillna(0.0)

        signals = strategy.generate_signals(price_data)
        if "signal" not in signals.columns:
            raise ValueError("Strategy must return a 'signal' column")
        merged = price_data.join(signals[["signal"]], how="left")
        merged["signal"] = merged["signal"].ffill().fillna(0)
        merged["position"] = merged["signal"].shift(1).fillna(0)

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
            signals=signals,
            equity_curve=equity_curve,
            statistics=statistics,
        )

    def run_many(
        self, data: pd.DataFrame, strategies: Iterable[Strategy]
    ) -> List[BacktestResult]:
        return [self.run(data, strategy) for strategy in strategies]

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
