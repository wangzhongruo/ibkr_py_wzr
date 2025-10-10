"""Utilities for downloading historical data from Interactive Brokers.

This module wraps ``ib_insync`` and provides a thin abstraction that makes it
simple to download minute level US equity data.  The resulting data is returned
as a :class:`pandas.DataFrame` and can optionally be persisted to disk.

Example
-------
>>> from ibkr_py_wzr.data_center import IBKRDataCenter
>>> data_center = IBKRDataCenter()
>>> data_center.connect()
>>> df = data_center.download_intraday_bars("AAPL", duration="5 D")
>>> data_center.disconnect()

The data frame contains the standard OHLCV columns that are produced by
``reqHistoricalData``.  If a ``save_to`` path is supplied the data frame is also
written as a parquet or csv file depending on the suffix.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

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
        end_datetime: Optional[datetime] = None,
        what_to_show: str = "TRADES",
        save_to: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Download 1 minute bars for the provided symbol.

        Parameters
        ----------
        symbol:
            Ticker symbol of the US equity.
        duration:
            How far back to fetch data (Interactive Brokers duration string).
        end_datetime:
            The end time for the historical request.  ``None`` means "now".
        what_to_show:
            Which data type should be returned (``TRADES`` / ``MIDPOINT`` ...).
        save_to:
            Optional path for persisting the result.  Supported suffixes are
            ``.csv`` and ``.parquet``.

        Returns
        -------
        pandas.DataFrame
            Data frame indexed by timezone aware timestamps with OHLCV columns.
        """
        self.connect()

        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        bars: BarDataList = self._ib.reqHistoricalData(
            contract=contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting="1 min",
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
            if save_path.suffix.lower() == ".csv":
                df.to_csv(save_path)
            elif save_path.suffix.lower() in {".parquet", ".pq"}:
                df.to_parquet(save_path)
            else:  # pragma: no cover - defensive coding
                raise ValueError(
                    "save_to must have a .csv or .parquet extension"
                )
            LOGGER.info("Saved data to %s", save_path)

        return df

    @property
    def ib(self) -> IB:
        """Expose the underlying :class:`ib_insync.IB` instance."""
        return self._ib
