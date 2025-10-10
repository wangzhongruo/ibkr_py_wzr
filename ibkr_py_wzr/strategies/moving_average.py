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
        if isinstance(result.index, pd.MultiIndex):
            grouped = result.groupby(level="symbol")
            result["fast_ma"] = grouped["close"].transform(
                lambda series: series.rolling(self.fast_window).mean()
            )
            result["slow_ma"] = grouped["close"].transform(
                lambda series: series.rolling(self.slow_window).mean()
            )
            result["alpha"] = result["fast_ma"] - result["slow_ma"]
            signal = pd.Series(0.0, index=result.index)
            signal[result["fast_ma"] > result["slow_ma"]] = 1.0
            signal[result["fast_ma"] < result["slow_ma"]] = -1.0
            signal = signal.groupby(level="symbol").ffill().fillna(0.0)
            result["signal"] = signal.astype(float)
        else:
            result["fast_ma"] = result["close"].rolling(self.fast_window).mean()
            result["slow_ma"] = result["close"].rolling(self.slow_window).mean()
            result["alpha"] = result["fast_ma"] - result["slow_ma"]
            result["signal"] = 0.0
            result.loc[result["fast_ma"] > result["slow_ma"], "signal"] = 1.0
            result.loc[result["fast_ma"] < result["slow_ma"], "signal"] = -1.0
            result["signal"] = result["signal"].ffill().fillna(0.0)
        return result
