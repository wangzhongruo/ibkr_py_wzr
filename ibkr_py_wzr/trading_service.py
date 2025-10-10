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
    account_type:
        IBKR account type the session is running against (``MARGIN`` by
        default).  Cash style accounts automatically suppress short signals.
    """

    data_center: IBKRDataCenter
    poll_interval: int = 60
    lookback_minutes: int = 120
    account_type: str = "MARGIN"
    _ib: IB = field(init=False, repr=False)
    _current_signal: int = field(default=0, init=False)
    _supports_shorting: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._ib = self.data_center.ib
        self.account_type = self.account_type.upper()
        self._supports_shorting = self.account_type not in {"CASH", "IRA"}

    def connect(self) -> None:
        self.data_center.connect()

    def disconnect(self) -> None:
        self.data_center.disconnect()

    async def place_market_order(self, symbol: str, quantity: int, action: str) -> None:
        """Submit a market order and log IBKR reported commissions."""

        contract = Stock(symbol, "SMART", "USD")
        await self._ib.qualifyContractsAsync(contract)

        order = MarketOrder(action=action, totalQuantity=quantity)
        trade = await self._ib.placeOrderAsync(contract, order)

        # Wait for the order to complete so the commission report becomes available.
        while not trade.isDone():
            await self._ib.waitOnUpdateAsync(timeout=1)

        commission_report = self._extract_commission_report(trade)
        if commission_report is not None:
            # IBKR reports fees as negative numbers (debits) and rebates as
            # positive amounts.  The value already includes exchange,
            # regulatory and broker components.
            LOGGER.info(
                "Placed %s order for %s x %s (order id %s, commission %s %s)",
                action,
                quantity,
                symbol,
                trade.order.orderId,
                commission_report.commission,
                commission_report.currency,
            )
        else:
            LOGGER.info(
                "Placed %s order for %s x %s (order id %s, commission pending)",
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
                    signal = self._apply_account_constraints(signal)
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

        await self.place_market_order(symbol, quantity, action)
        self._current_signal = signal

    @staticmethod
    def _extract_commission_report(trade) -> Optional[object]:
        """Return the most relevant commission report from ``trade`` if available."""

        if trade.commissionReport is not None:
            return trade.commissionReport

        for fill in getattr(trade, "fills", []):
            report = getattr(fill, "commissionReport", None)
            if report is not None:
                return report

        return None

    def _apply_account_constraints(self, signal: int) -> int:
        """Return ``signal`` adjusted for the configured ``account_type``."""

        if self._supports_shorting or signal >= 0:
            return int(signal)

        LOGGER.info(
            "Suppressing short signal %s for cash-style account type %s", signal, self.account_type
        )
        return 0
