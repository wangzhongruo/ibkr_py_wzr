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
frame is also written as a parquet or pickle file depending on the suffix.  The
latency of the downloaded bars mirrors the market data subscriptions available
to the connected IBKR account: live feeds yield real-time bars, while accounts
without live permissions receive the delayed (typically 15 minute) stream.
"""
from __future__ import annotations

import io
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
import requests
from zoneinfo import ZoneInfo
import yaml

try:  # pragma: no cover - optional dependency for user experience.
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress bar is optional.
    tqdm = None

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
    nasdaq_listing_url, nasdaq_listing_fallback_urls,
    nasdaq_listing_timeout, nasdaq_listing_retries, nasdaq_listing_retry_backoff:
        Control the source URLs as well as the timeout and retry behaviour when
        downloading the NASDAQ listings file that seeds the universe updater.
    """

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    use_rth: bool = True
    timezone: str = "America/New_York"
    data_directory: Path = Path("data/nasdaq")
    nasdaq_listing_url: str = (
        "https://ftp.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt"
    )
    nasdaq_listing_fallback_urls: Sequence[str] = field(
        default_factory=lambda: (
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
        )
    )
    nasdaq_listing_timeout: float = 30.0
    nasdaq_listing_retries: int = 3
    nasdaq_listing_retry_backoff: float = 5.0
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
        primary_exchange: Optional[str] = None,
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
        primary_exchange:
            Optional primary exchange hint supplied to the contract to improve
            symbol disambiguation.
        save_to:
            Optional path for persisting the result.  Supported suffixes are
            ``.parquet``, ``.pq``, ``.pkl`` and ``.pickle``.

        Returns
        -------
        pandas.DataFrame
            Data frame indexed by timezone aware timestamps with OHLCV columns.
        """
        self.connect()

        if primary_exchange:
            contract = Stock(symbol, "SMART", "USD", primaryExchange=primary_exchange)
        else:
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
        if not bars:
            LOGGER.warning("No historical data returned for symbol %s", symbol)
            return pd.DataFrame()

        df = util.df(bars)
        if df is None or df.empty:
            LOGGER.warning("No historical data returned for symbol %s", symbol)
            return pd.DataFrame() if df is None else df

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
        primary_exchange: Optional[str] = None,
        save_to: Optional[Path] = None,
        throttle_seconds: float = 0.0,
        progress: bool = True,
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
        progress:
            When ``True`` and :mod:`tqdm` is installed a progress bar is emitted
            to STDERR while looping through ``symbols``.  Falls back to logging
            updates when the dependency is unavailable.

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

        progress_enabled = bool(progress and tqdm is not None)
        if progress and not progress_enabled:
            LOGGER.info(
                "Progress display requested but tqdm is not installed; falling back to logging"
            )

        iterator: Iterable[str]
        if progress_enabled:
            progress_bar = tqdm(unique_symbols, desc="Downloading symbols", unit="symbol")
            iterator = progress_bar  # type: ignore[assignment]
        else:
            progress_bar = None
            iterator = unique_symbols

        for idx, ticker in enumerate(iterator, start=1):
            if progress_bar is not None:
                progress_bar.set_postfix_str(ticker)
            else:
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
                primary_exchange=primary_exchange,
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

        if progress_bar is not None:
            progress_bar.close()

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

    def fetch_nasdaq_symbols(self) -> list[str]:
        """Return the complete list of NASDAQ-listed equities."""

        retries = max(1, int(self.nasdaq_listing_retries))
        timeout = float(self.nasdaq_listing_timeout)
        backoff = max(0.0, float(self.nasdaq_listing_retry_backoff))

        listing_urls: list[str] = []
        if self.nasdaq_listing_url:
            listing_urls.append(self.nasdaq_listing_url)
        listing_urls.extend(self.nasdaq_listing_fallback_urls or [])

        last_error: Optional[Exception] = None
        response: Optional[requests.Response] = None

        for url in listing_urls:
            LOGGER.info("Downloading NASDAQ listings from %s", url)
            response = None
            for attempt in range(1, retries + 1):
                try:
                    response = requests.get(url, timeout=timeout)
                    response.raise_for_status()
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    LOGGER.warning(
                        "Attempt %s/%s to download NASDAQ listings from %s failed: %s",
                        attempt,
                        retries,
                        url,
                        exc,
                    )
                    if attempt == retries:
                        break
                    if backoff:
                        sleep(backoff * attempt)
            if response is not None and response.status_code < 400:
                break

        if response is None or response.status_code >= 400:
            raise RuntimeError(
                "Unable to download NASDAQ listings after trying all configured sources"
            ) from last_error

        buffer = io.StringIO(response.text)
        listings = pd.read_csv(
            buffer,
            sep="|",
            dtype=str,
            engine="python",
        )
        listings = listings[listings["Symbol"].notna()]
        listings = listings[listings["Symbol"].str.strip() != ""]
        listings = listings[
            ~listings["Symbol"].str.contains("File Creation Time", case=False, na=False)
        ]
        if "Test Issue" in listings.columns:
            listings = listings[listings["Test Issue"] != "Y"]
        if "ETF" in listings.columns:
            listings = listings[listings["ETF"] != "Y"]
        if "Listing Exchange" in listings.columns:
            listings = listings[listings["Listing Exchange"].str.upper().isin({"Q"})]
        symbols = sorted(listings["Symbol"].str.upper().unique())
        LOGGER.info("Resolved %s NASDAQ symbols", len(symbols))
        return symbols

    def update_symbol_history(
        self,
        symbol: str,
        *,
        bar_size: str,
        what_to_show: str,
        start_datetime: datetime,
        end_datetime: Optional[datetime],
        data_directory: Path,
    ) -> None:
        """Ensure the local history for ``symbol`` is up to date."""

        symbol_path = data_directory / self._symbol_filename(symbol, bar_size)
        LOGGER.debug("Updating %s using %s", symbol, symbol_path)

        existing: Optional[pd.DataFrame]
        if symbol_path.exists():
            existing = pd.read_parquet(symbol_path)
            if not existing.empty:
                existing.sort_index(inplace=True)
                if existing.index.tz is None:
                    existing.index = existing.index.tz_localize(self.timezone)
        else:
            existing = None

        tz = ZoneInfo(self.timezone)
        bar_offset = self._bar_size_to_offset(bar_size)
        tolerance_start = pd.Timedelta(days=7)
        tolerance_end = max(bar_offset * 5, pd.Timedelta(days=2))

        target_start = self._ensure_timezone(start_datetime, tz)
        request_start = target_start

        request_end = (
            self._ensure_timezone(end_datetime, tz) if end_datetime is not None else None
        )
        target_end = request_end if request_end is not None else datetime.now(tz)

        if existing is not None and not existing.empty:
            earliest = existing.index.min()
            latest = existing.index.max()

            if earliest.tzinfo is None:
                earliest = earliest.tz_localize(tz)
            else:
                earliest = earliest.astimezone(tz)

            if latest.tzinfo is None:
                latest = latest.tz_localize(tz)
            else:
                latest = latest.astimezone(tz)

            missing_start = earliest > (target_start + tolerance_start)
            missing_end = latest < (target_end - tolerance_end)

            if missing_start or missing_end:
                LOGGER.info(
                    (
                        "Existing history for %s spans %s to %s but does not cover the "
                        "requested range %s to %s; performing a full refresh"
                    ),
                    symbol,
                    earliest,
                    latest,
                    target_start,
                    target_end,
                )
                existing = None
            else:
                if latest >= target_end - bar_offset:
                    LOGGER.info(
                        "%s already contains data through %s; skipping incremental update",
                        symbol,
                        latest,
                    )
                    return
                request_start = max(target_start, latest + bar_offset)

        if existing is None:
            request_start = target_start

        if request_end is not None and request_start >= request_end:
            LOGGER.info(
                "%s already up to date (no newer data before %s)",
                symbol,
                request_end,
            )
            return

        new_data = self.download_intraday_bars(
            symbol,
            start_datetime=request_start,
            end_datetime=request_end,
            what_to_show=what_to_show,
            bar_size=bar_size,
            primary_exchange="NASDAQ",
        )

        if new_data.empty:
            LOGGER.info("No new data returned for %s", symbol)
            return

        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_data

        combined.sort_index(inplace=True)

        symbol_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(symbol_path)
        LOGGER.info("Stored %s rows for %s", len(combined.index), symbol)

    def update_nasdaq_history(
        self,
        *,
        bar_size: str = "1 min",
        what_to_show: str = "TRADES",
        start_date: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        data_directory: Optional[Path] = None,
        throttle_seconds: float = 0.2,
        progress: bool = True,
        max_workers: int = 1,
    ) -> None:
        """Download and incrementally update NASDAQ equity data.

        Parameters
        ----------
        progress:
            When ``True`` and :mod:`tqdm` is installed a progress bar is rendered
            while syncing the exchange universe.  Falls back to logging when the
            dependency is unavailable.
        max_workers:
            Number of worker threads used for the download.  ``1`` preserves the
            historical single-threaded behaviour while larger values spawn
            additional background workers that each maintain their own IBKR
            connection.  Each worker uses an incremented ``client_id`` to avoid
            clashes with the primary session.
        """

        symbols = self.fetch_nasdaq_symbols()
        tz = ZoneInfo(self.timezone)
        start_dt = self._ensure_timezone(start_date, tz) if start_date else datetime(2011, 1, 1, tzinfo=tz)
        data_dir = Path(data_directory) if data_directory else self.data_directory
        data_dir.mkdir(parents=True, exist_ok=True)

        max_workers = max(1, int(max_workers))
        progress_enabled = bool(progress and tqdm is not None)
        if progress and not progress_enabled:
            LOGGER.info(
                "Progress display requested but tqdm is not installed; falling back to logging"
            )

        total_symbols = len(symbols)
        if total_symbols == 0:
            LOGGER.warning("No NASDAQ symbols resolved; nothing to update")
            return

        if max_workers == 1:
            self.connect()
            if progress_enabled:
                progress_bar = tqdm(symbols, desc="Updating NASDAQ history", unit="symbol")
                iterator = enumerate(progress_bar, start=1)
            else:
                progress_bar = None
                iterator = enumerate(symbols, start=1)

            try:
                for idx, symbol in iterator:
                    if progress_bar is not None:
                        progress_bar.set_postfix_str(symbol)
                    else:
                        LOGGER.info("(%s/%s) Updating %s", idx, total_symbols, symbol)
                    self.update_symbol_history(
                        symbol,
                        bar_size=bar_size,
                        what_to_show=what_to_show,
                        start_datetime=start_dt,
                        end_datetime=end_datetime,
                        data_directory=data_dir,
                    )
                    if throttle_seconds and idx < total_symbols:
                        sleep(throttle_seconds)
            finally:
                if progress_bar is not None:
                    progress_bar.close()
                self.disconnect()
            return

        # Parallel execution path -------------------------------------------------
        chunks = [symbols[i::max_workers] for i in range(max_workers)]
        chunks = [chunk for chunk in chunks if chunk]
        progress_bar = tqdm(
            total=total_symbols,
            desc="Updating NASDAQ history",
            unit="symbol",
            disable=not progress_enabled,
        ) if progress_enabled else None

        completed = 0
        completed_lock = threading.Lock()

        def worker(worker_index: int, chunk: Sequence[str]) -> None:
            nonlocal completed
            client_id = self.client_id + worker_index + 1
            worker_center = IBKRDataCenter(
                host=self.host,
                port=self.port,
                client_id=client_id,
                use_rth=self.use_rth,
                timezone=self.timezone,
                data_directory=data_dir,
                nasdaq_listing_url=self.nasdaq_listing_url,
                nasdaq_listing_fallback_urls=self.nasdaq_listing_fallback_urls,
                nasdaq_listing_timeout=self.nasdaq_listing_timeout,
                nasdaq_listing_retries=self.nasdaq_listing_retries,
                nasdaq_listing_retry_backoff=self.nasdaq_listing_retry_backoff,
            )
            try:
                worker_center.connect()
                for local_index, symbol in enumerate(chunk, start=1):
                    try:
                        worker_center.update_symbol_history(
                            symbol,
                            bar_size=bar_size,
                            what_to_show=what_to_show,
                            start_datetime=start_dt,
                            end_datetime=end_datetime,
                            data_directory=data_dir,
                        )
                    except Exception:  # pragma: no cover - robust worker handling
                        LOGGER.exception("Worker %s failed updating %s", worker_index + 1, symbol)
                    if progress_bar is not None:
                        progress_bar.update(1)
                    else:
                        with completed_lock:
                            completed += 1
                            LOGGER.info("(%s/%s) Updated %s", completed, total_symbols, symbol)
                    if throttle_seconds and local_index < len(chunk):
                        sleep(throttle_seconds)
            finally:
                worker_center.disconnect()

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(worker, idx, chunk) for idx, chunk in enumerate(chunks)
            ]
            for future in futures:
                try:
                    future.result()
                except Exception:  # pragma: no cover - defensive aggregate handling
                    LOGGER.exception("A background worker terminated unexpectedly")

        if progress_bar is not None:
            progress_bar.close()

    @staticmethod
    def _bar_size_to_offset(bar_size: str) -> pd.Timedelta:
        amount_str, unit = bar_size.split()
        amount = int(amount_str)
        unit = unit.lower()
        if unit.startswith("sec"):
            return pd.Timedelta(seconds=amount)
        if unit.startswith("min"):
            return pd.Timedelta(minutes=amount)
        if unit.startswith("hour"):
            return pd.Timedelta(hours=amount)
        if unit.startswith("day"):
            return pd.Timedelta(days=amount)
        if unit.startswith("week"):
            return pd.Timedelta(weeks=amount)
        raise ValueError(f"Unsupported bar size: {bar_size}")

    @staticmethod
    def _ensure_timezone(dt: datetime, tz: ZoneInfo) -> datetime:
        if isinstance(dt, pd.Timestamp):
            if dt.tzinfo is None:
                return dt.tz_localize(tz).to_pydatetime()
            return dt.tz_convert(tz).to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=tz)
        return dt.astimezone(tz)

    @staticmethod
    def _symbol_filename(symbol: str, bar_size: str) -> str:
        clean_symbol = "".join(ch if ch.isalnum() else "_" for ch in symbol.upper())
        suffix = bar_size.replace(" ", "").lower()
        return f"{clean_symbol}_{suffix}.parquet"

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


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Return the configuration stored in ``path`` or an empty dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("The YAML configuration must define a mapping at the top level")
    return data


def _maybe_parse_datetime(value: Any) -> Optional[datetime]:
    """Return ``value`` as a :class:`datetime` if possible."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return datetime.fromisoformat(str(value))


def main() -> None:
    """CLI helper for updating the NASDAQ data directory."""

    import argparse

    parser = argparse.ArgumentParser(description="Download NASDAQ historical data")
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML configuration file")
    parser.add_argument("--bar-size", default=None, help="Bar size for historical data (overrides config)")
    parser.add_argument(
        "--what-to-show",
        default=None,
        help="Data type to request from IBKR (overrides config)",
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=None,
        help="Directory used to store parquet files (overrides config)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional ISO start date override (overrides config)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional ISO end date override (overrides config)",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=None,
        help="Seconds to wait between requests (overrides config)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of worker threads used for downloads (overrides config)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars even when the dependency is installed",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config: dict[str, Any] = {}
    if args.config is not None:
        config = _load_yaml_config(args.config)

    data_center_config = dict(config.get("data_center", {}))
    update_config = dict(config.get("update", {}))

    if args.bar_size is not None:
        update_config["bar_size"] = args.bar_size
    if args.what_to_show is not None:
        update_config["what_to_show"] = args.what_to_show
    if args.data_directory is not None:
        update_config["data_directory"] = args.data_directory
        data_center_config.setdefault("data_directory", args.data_directory)
    if args.start_date is not None:
        update_config["start_date"] = args.start_date
    if args.end_date is not None:
        update_config["end_datetime"] = args.end_date
    if args.throttle is not None:
        update_config["throttle_seconds"] = args.throttle
    if args.max_workers is not None:
        update_config["max_workers"] = args.max_workers
    if args.no_progress:
        update_config["progress"] = False

    defaults = {
        "bar_size": "1 min",
        "what_to_show": "TRADES",
        "throttle_seconds": 0.2,
        "progress": True,
        "max_workers": 1,
    }
    for key, value in defaults.items():
        update_config.setdefault(key, value)

    if "start_date" not in update_config:
        update_config["start_date"] = datetime(2011, 1, 1)

    start_date = _maybe_parse_datetime(update_config.pop("start_date", None))
    end_datetime = _maybe_parse_datetime(
        update_config.pop("end_datetime", update_config.pop("end_date", None))
    )

    if "data_directory" in update_config and not isinstance(update_config["data_directory"], Path):
        update_config["data_directory"] = Path(update_config["data_directory"])
    if "data_directory" in data_center_config and not isinstance(
        data_center_config["data_directory"], Path
    ):
        data_center_config["data_directory"] = Path(data_center_config["data_directory"])
    if "nasdaq_listing_timeout" in data_center_config:
        data_center_config["nasdaq_listing_timeout"] = float(
            data_center_config["nasdaq_listing_timeout"]
        )
    if "nasdaq_listing_retry_backoff" in data_center_config:
        data_center_config["nasdaq_listing_retry_backoff"] = float(
            data_center_config["nasdaq_listing_retry_backoff"]
        )
    if "nasdaq_listing_retries" in data_center_config:
        data_center_config["nasdaq_listing_retries"] = max(
            1, int(data_center_config["nasdaq_listing_retries"])
        )
    if "nasdaq_listing_fallback_urls" in data_center_config:
        fallback_urls = data_center_config["nasdaq_listing_fallback_urls"]
        if isinstance(fallback_urls, (str, Path)):
            data_center_config["nasdaq_listing_fallback_urls"] = [
                str(fallback_urls)
            ]
        else:
            data_center_config["nasdaq_listing_fallback_urls"] = [
                str(url) for url in fallback_urls
            ]
    if "throttle_seconds" in update_config:
        update_config["throttle_seconds"] = float(update_config["throttle_seconds"])
    if "progress" in update_config:
        update_config["progress"] = bool(update_config["progress"])
    if "max_workers" in update_config:
        update_config["max_workers"] = max(1, int(update_config["max_workers"]))

    data_center = IBKRDataCenter(**data_center_config)
    data_center.update_nasdaq_history(
        start_date=start_date,
        end_datetime=end_datetime,
        **update_config,
    )


if __name__ == "__main__":
    main()
