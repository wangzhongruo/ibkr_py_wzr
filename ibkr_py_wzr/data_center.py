"""Utilities for downloading historical data from Interactive Brokers.

This module wraps ``ib_insync`` and provides a thin abstraction that makes it
simple to download US equity data with a configurable bar size.  The resulting
data is returned as a :class:`pandas.DataFrame` and can optionally be persisted
to disk.

Example
-------
>>> from ibkr_py_wzr.data_center import IBKRDataCenter
>>> data_center = IBKRDataCenter()
>>> data_center.connect()
>>> df = data_center.download_intraday_bars("AAPL", duration="5 D", bar_size="1 min")
>>> data_center.disconnect()

The data frame contains the standard OHLCV columns that are produced by
``reqHistoricalData``: ``open``, ``high``, ``low``, ``close``, ``volume``,
``bar_count`` and ``average_price``.  If a ``save_to`` path is supplied the data
frame is also written as a parquet or pickle file depending on the suffix.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Optional, Sequence

import pandas as pd

try:  # pragma: no cover - allows the code to be imported without ib_insync.
    from ib_insync import IB, BarDataList, Stock, util
except Exception as exc:  # pragma: no cover - keeps import error informative.
    raise ImportError(
        "ib_insync must be installed to use IBKRDataCenter."
    ) from exc

LOGGER = logging.getLogger(__name__)


@dataclass
class IBKRDataCenter:
    """Download historical bar data from Interactive Brokers.

    Parameters
    ----------
    host, port, client_id:
        Connection details for TWS or IB Gateway.
    use_rth:
        When ``True`` only Regular Trading Hours data is returned.
    timezone:
        Target timezone (as a tz database string) for the returned data frame.
    """

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    use_rth: bool = True
    timezone: str = "America/New_York"
    _ib: IB = field(default_factory=IB, init=False, repr=False)

    def connect(self) -> None:
        """Connect to IBKR if not connected already."""
        if not self._ib.isConnected():
            LOGGER.info(
                "Connecting to IBKR at %s:%s with client id %s",
                self.host,
                self.port,
                self.client_id,
            )
            self._ib.connect(self.host, self.port, clientId=self.client_id)

    def disconnect(self) -> None:
        """Close the IBKR connection."""
        if self._ib.isConnected():
            LOGGER.info("Disconnecting from IBKR")
            self._ib.disconnect()

    def download_intraday_bars(
        self,
        symbol: str,
        duration: str = "1 D",
        start_datetime: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        what_to_show: str = "TRADES",
        bar_size: str = "1 min",
        save_to: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Download historical bars for the provided symbol.

        Parameters
        ----------
        symbol:
            Ticker symbol of the US equity.
        duration:
            How far back to fetch data (Interactive Brokers duration string).
            Ignored when ``start_datetime`` is provided.
        start_datetime:
            Optional start time for the historical request.  When provided the
            ``duration`` argument is ignored and the request duration is
            inferred from ``start_datetime`` and ``end_datetime``.
        end_datetime:
            The end time for the historical request.  ``None`` means "now".
        what_to_show:
            Which data type should be returned (``TRADES`` / ``MIDPOINT`` ...).
        bar_size:
            Granularity of the returned bars (``1 min``, ``1 hour``, ``1 day``
            ...).  The value must be a valid Interactive Brokers bar size.
        save_to:
            Optional path for persisting the result.  Supported suffixes are
            ``.parquet``, ``.pq``, ``.pkl`` and ``.pickle``.

        Returns
        -------
        pandas.DataFrame
            Data frame indexed by timezone aware timestamps with OHLCV columns.
        """
        self.connect()

        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        request_end = end_datetime
        duration_str = duration

        if start_datetime is not None:
            if end_datetime is None:
                request_end = datetime.now(tz=start_datetime.tzinfo)
            if request_end <= start_datetime:
                raise ValueError(
                    "start_datetime must be before end_datetime"
                )
            duration_str = self._duration_from_range(start_datetime, request_end)

        bars: BarDataList = self._ib.reqHistoricalData(
            contract=contract,
            endDateTime=request_end,
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=self.use_rth,
            formatDate=1,
        )
        df = util.df(bars)
        if df.empty:
            LOGGER.warning("No historical data returned for symbol %s", symbol)
            return df

        df.set_index("date", inplace=True)
        df.index = df.index.tz_localize("UTC").tz_convert(self.timezone)
        df.index.name = "timestamp"
        df.rename(
            columns={
                "volume": "volume",
                "barCount": "bar_count",
                "average": "average_price",
            },
            inplace=True,
        )

        LOGGER.info(
            "Downloaded %s rows of data for %s", len(df.index), symbol
        )

        if save_to is not None:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            suffix = save_path.suffix.lower()
            if suffix in {".parquet", ".pq"}:
                df.to_parquet(save_path)
            elif suffix in {".pkl", ".pickle"}:
                df.to_pickle(save_path)
            else:  # pragma: no cover - defensive coding
                raise ValueError(
                    "save_to must have a .parquet, .pq, .pkl or .pickle extension"
                )
            LOGGER.info("Saved data to %s", save_path)

        return df

    def download_equity_universe(
        self,
        symbols: Sequence[str],
        duration: str = "1 D",
        start_datetime: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        what_to_show: str = "TRADES",
        bar_size: str = "1 min",
        save_to: Optional[Path] = None,
        throttle_seconds: float = 0.0,
    ) -> pd.DataFrame:
        """Download historical bars for multiple symbols and combine the results.

        Parameters
        ----------
        symbols:
            Iterable of ticker symbols that should be requested from Interactive
            Brokers.
        duration, start_datetime, end_datetime, what_to_show, bar_size:
            Identical to :meth:`download_intraday_bars`.
        save_to:
            Optional path used to persist the concatenated universe.  When
            provided the combined data frame is stored as parquet or pickle.
        throttle_seconds:
            Optional delay inserted between requests to avoid pacing violations
            for particularly large universes.

        Returns
        -------
        pandas.DataFrame
            Multi-indexed data frame keyed by ``timestamp`` and ``symbol``.
        """

        unique_symbols = list(dict.fromkeys(symbols))
        if not unique_symbols:
            raise ValueError("symbols must contain at least one entry")

        combined_frames: list[pd.DataFrame] = []
        self.connect()

        for idx, ticker in enumerate(unique_symbols, start=1):
            LOGGER.info(
                "Downloading data for %s (%s/%s)", ticker, idx, len(unique_symbols)
            )
            frame = self.download_intraday_bars(
                ticker,
                duration=duration,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                what_to_show=what_to_show,
                bar_size=bar_size,
            )
            if frame.empty:
                LOGGER.warning("Skipping %s - no data returned", ticker)
                continue

            frame = frame.copy()
            frame["symbol"] = ticker
            frame.set_index("symbol", append=True, inplace=True)
            frame = frame.reorder_levels(["timestamp", "symbol"])
            frame.sort_index(inplace=True)
            combined_frames.append(frame)

            if throttle_seconds > 0 and idx < len(unique_symbols):
                sleep(throttle_seconds)

        if not combined_frames:
            LOGGER.warning("No data downloaded for requested universe")
            combined = pd.DataFrame()
        else:
            combined = pd.concat(combined_frames).sort_index()

        if save_to is not None and not combined.empty:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            suffix = save_path.suffix.lower()
            if suffix in {".parquet", ".pq"}:
                combined.to_parquet(save_path)
            elif suffix in {".pkl", ".pickle"}:
                combined.to_pickle(save_path)
            else:  # pragma: no cover - defensive coding
                raise ValueError(
                    "save_to must have a .parquet, .pq, .pkl or .pickle extension"
                )
            LOGGER.info(
                "Saved universe data (%s symbols) to %s",
                len(combined.index.get_level_values("symbol").unique())
                if not combined.empty
                else 0,
                save_path,
            )

        return combined

    @staticmethod
    def _duration_from_range(start: datetime, end: datetime) -> str:
        """Return an IBKR duration string that spans ``start`` to ``end``."""

        delta = end - start
        total_seconds = delta.total_seconds()
        if total_seconds <= 0:
            raise ValueError("end must be after start")

        if total_seconds < 24 * 60 * 60:
            return f"{math.ceil(total_seconds)} S"

        days = total_seconds / (24 * 60 * 60)
        if days < 7:
            return f"{math.ceil(days)} D"

        weeks = days / 7
        if weeks < 52:
            return f"{math.ceil(weeks)} W"

        months = days / 30
        if months < 12:
            return f"{math.ceil(months)} M"

        years = days / 365
        return f"{math.ceil(years)} Y"

    @property
    def ib(self) -> IB:
        """Expose the underlying :class:`ib_insync.IB` instance."""
        return self._ib
