"""Built-in strategies."""

from .base import SignalOnlyStrategy, Strategy
from .moving_average import MovingAverageCrossStrategy

__all__ = [
    "SignalOnlyStrategy",
    "Strategy",
    "MovingAverageCrossStrategy",
]
