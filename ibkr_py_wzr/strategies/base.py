"""Strategy abstractions used by both the backtester and live trading service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    """Protocol shared by all strategies."""

    name: str

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a data frame with at least a ``signal`` column.

        Positive values indicate a long position, negative values a short
        position and ``0`` means flat.  Strategies may also add auxiliary
        columns (for example indicators) that the backtester can include in its
        result payload.
        """

    def generate_live_signal(self, data: pd.DataFrame) -> int:
        """Return the current signal given the latest available data."""


@dataclass
class SignalOnlyStrategy:
    """Helper base class that implements ``generate_live_signal``."""

    name: str

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - abstract method
        raise NotImplementedError

    def generate_live_signal(self, data: pd.DataFrame) -> int:
        signals = self.generate_signals(data)
        return int(signals["signal"].iloc[-1])
