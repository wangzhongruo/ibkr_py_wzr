"""Example strategy implementations."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import SignalOnlyStrategy


@dataclass
class MovingAverageCrossStrategy(SignalOnlyStrategy):
    """Simple moving average cross over strategy.

    Parameters
    ----------
    fast_window, slow_window:
        Window sizes (in bars) for the two moving averages.  The strategy goes
        long when the fast moving average crosses above the slow average and
        short when the fast moving average crosses below the slow average.
    """

    fast_window: int = 20
    slow_window: int = 50

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if "close" not in data.columns:
            raise ValueError("Data frame must contain a 'close' column")

        result = data.copy()
        result["fast_ma"] = result["close"].rolling(self.fast_window).mean()
        result["slow_ma"] = result["close"].rolling(self.slow_window).mean()
        result["signal"] = 0
        result.loc[result["fast_ma"] > result["slow_ma"], "signal"] = 1
        result.loc[result["fast_ma"] < result["slow_ma"], "signal"] = -1
        result["signal"] = result["signal"].ffill().fillna(0)
        return result
