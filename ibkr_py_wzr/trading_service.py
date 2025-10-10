"""Live trading service built on top of ``ib_insync``."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ib_insync import IB, MarketOrder, Stock

from .data_center import IBKRDataCenter
from .strategies.base import Strategy

LOGGER = logging.getLogger(__name__)


@dataclass
class IBKRTradingService:
    """Utility for running a strategy in real time.

    Parameters
    ----------
    data_center:
        Instance of :class:`IBKRDataCenter` that provides the underlying
        ``IB`` connection and historical data helper.
    poll_interval:
        Number of seconds between polling for new data when in live mode.
    lookback_minutes:
        Minimum amount of history (in minutes) to feed to the strategy when
        generating live signals.
    """

    data_center: IBKRDataCenter
    poll_interval: int = 60
    lookback_minutes: int = 120
    _ib: IB = field(init=False, repr=False)
    _current_signal: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._ib = self.data_center.ib

    def connect(self) -> None:
        self.data_center.connect()

    def disconnect(self) -> None:
        self.data_center.disconnect()

    def place_market_order(self, symbol: str, quantity: int, action: str) -> None:
        """Submit a market order."""
        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        order = MarketOrder(action=action, totalQuantity=quantity)
        trade = self._ib.placeOrder(contract, order)
        LOGGER.info(
            "Placed %s order for %s x %s (order id %s)",
            action,
            quantity,
            symbol,
            trade.order.orderId,
        )

    async def run_live_strategy(
        self,
        symbol: str,
        strategy: Strategy,
        quantity: int = 100,
        what_to_show: str = "TRADES",
        use_rth: Optional[bool] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Continuously poll data and forward it to ``strategy``.

        The strategy receives a data frame containing at least ``lookback``
        minutes of history.  Its ``generate_live_signal`` method determines the
        desired position (long, flat or short).  Orders are submitted whenever
        the signal changes.
        """

        if use_rth is not None:
            original = self.data_center.use_rth
            self.data_center.use_rth = use_rth
        else:
            original = None

        try:
            while True:
                end_time = datetime.utcnow()
                duration = f"{max(self.lookback_minutes, 1)} m"
                df = self.data_center.download_intraday_bars(
                    symbol=symbol,
                    duration=duration,
                    end_datetime=end_time,
                    what_to_show=what_to_show,
                )
                if df.empty:
                    LOGGER.warning("No data returned for %s, retrying", symbol)
                else:
                    signal = strategy.generate_live_signal(df)
                    await self._maybe_rebalance(symbol, signal, quantity)

                if stop_event is not None and stop_event.is_set():
                    LOGGER.info("Stop event received, terminating live loop")
                    break
                await asyncio.sleep(self.poll_interval)
        finally:
            if original is not None:
                self.data_center.use_rth = original

    async def _maybe_rebalance(
        self, symbol: str, signal: int, quantity: int
    ) -> None:
        if signal == self._current_signal:
            return

        if signal > 0:
            action = "BUY"
        elif signal < 0:
            action = "SELL"
        else:
            # Go flat by reversing the existing position if needed.
            action = "SELL" if self._current_signal > 0 else "BUY"

        self.place_market_order(symbol, quantity, action)
        self._current_signal = signal
