import logging
import sys
from datetime import datetime
from typing import Optional, Union, Dict

import click
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

INTERVAL_SCALING: Dict[str, float] = {
    '1m': 365 * 24 * 60,
    '3m': 365 * 24 * 60 / 3,
    '5m': 365 * 24 * 12,
    '15m': 365 * 24 * 4,
    '30m': 365 * 24 * 2,
    '1h': 365 * 24,
    '2h': 365 * 12,
    '4h': 365 * 6,
    '1d': 365,
}


def create_session(retries: int = 3, backoff: float = 0.3) -> requests.Session:
    """
    Creates a requests.Session with retry logic.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        # Updated parameter name for urllib3>=1.26
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class VolatilityCalculator:
    """
    Calculates various volatility measures from OHLCV data.
    """

    def __init__(
        self,
        interval: str,
        session: Optional[requests.Session] = None
    ) -> None:
        if interval not in INTERVAL_SCALING:
            raise ValueError(f"Unsupported interval: {interval}")
        self.interval = interval
        self.annual_scaling = np.sqrt(INTERVAL_SCALING[interval])
        self.session = session or create_session()
        self.base_url = 'https://api.binance.com'
        logger.debug("Initialized VolatilityCalculator with interval=%s", interval)

    def datetime_to_milliseconds(self, date: Union[datetime, int, float]) -> int:
        if isinstance(date, datetime):
            return int(date.timestamp() * 1000)
        if isinstance(date, (int, float)):
            return int(date)
        raise TypeError("date must be datetime or timestamp int/float")

    def fetch_data(
        self,
        symbol: str,
        start_ts: Union[datetime, int, float],
        end_ts: Union[datetime, int, float],
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetches kline data from Binance API as a pandas DataFrame.
        """
        start_ms = self.datetime_to_milliseconds(start_ts)
        end_ms = self.datetime_to_milliseconds(end_ts)
        url = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': self.interval,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': limit
        }
        logger.info("Requesting data for %s: %s to %s", symbol, start_ts, end_ts)
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(data, columns=cols)
        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df['r'] = np.log(df['close'] / df['close'].shift(1))
        df['v2_park'] = (np.log(df['high'] / df['low']) ** 2) / (4 * np.log(2))
        df['v2_gk'] = (
            0.511 * np.log(df['high'] / df['low']) ** 2
            - 0.019 * np.log(df['close'] / df['open'])
             * np.log((df['high'] * df['low']) / df['open'] ** 2)
            - 0.383 * np.log(df['close'] / df['open']) ** 2
        )
        df['v2_rs'] = (
            np.log(df['high'] / df['open']) * np.log(df['high'] / df['close'])
            + np.log(df['low'] / df['open']) * np.log(df['low'] / df['close'])
        )
        df['bv_term'] = df['r'].abs() * df['r'].shift(1).abs()
        return df

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['date'] = df.index.date
        grouped = df.groupby('date')
        intervals = grouped.size()
        daily = pd.DataFrame({
            'RV': grouped['r'].apply(lambda x: (x**2).sum()),
            'BV': grouped['bv_term'].apply(lambda x: (np.pi/2 * x).sum()),
            'Parkinson': grouped['v2_park'].sum(),
            'GarmanKlass': grouped['v2_gk'].sum(),
            'RogersSatchell': grouped['v2_rs'].sum(),
        })
        for col in daily.columns:
            daily[col] /= intervals
            daily[f"{col}_ann"] = np.sqrt(daily[col]) * self.annual_scaling
        daily.index = pd.to_datetime(daily.index)
        return daily

    def intraday_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['hour'] = df.index.hour
        grouped = df.groupby('hour')
        counts = grouped['r'].count()
        intraday = pd.DataFrame({
            'RV': grouped['r'].apply(lambda x: (x**2).sum()),
            'Parkinson': grouped['v2_park'].sum(),
            'GarmanKlass': grouped['v2_gk'].sum(),
            'RogersSatchell': grouped['v2_rs'].sum(),
        })
        for col in intraday.columns:
            intraday[col] /= counts
            intraday[f"{col}_ann"] = np.sqrt(intraday[col]) * self.annual_scaling
        return intraday

    def rolling_volatility(
        self,
        df: pd.DataFrame,
        window: int = 24
    ) -> pd.Series:
        roll_sum = df['r'].pow(2).rolling(window).sum()
        scaling = np.sqrt(self.annual_scaling**2 / window)
        return (roll_sum**0.5) * scaling


@click.command()
@click.option(
    '--symbol', required=True, help='Ticker symbol, e.g., ETHUSDT'
)
@click.option(
    '--interval', default='1h', show_default=True,
    type=click.Choice(list(INTERVAL_SCALING.keys())), help='Data interval'
)
@click.option(
    '--window', default=168, show_default=True,
    type=int, help='Rolling window size in intervals'
)
@click.option(
    '--start', required=True,
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S"]),
    help='Start timestamp, e.g., 2024-06-01T00:00:00'
)
@click.option(
    '--end', required=True,
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S"]),
    help='End timestamp, e.g., 2024-07-01T00:00:00'
)
def main(
    symbol: str,
    interval: str,
    window: int,
    start: datetime,
    end: datetime
) -> None:
    """
    CLI entry point: fetches OHLCV data and computes volatility measures.
    """
    try:
        calc = VolatilityCalculator(interval)
        raw = calc.fetch_data(symbol, start, end)
        df = calc.process_data(raw)
        daily = calc.aggregate_daily(df)
        intraday = calc.intraday_pattern(df)
        rolling = calc.rolling_volatility(df, window)

        logger.info("Daily volatility summary:\n%s", daily.tail())
        logger.info("Intraday pattern summary:\n%s", intraday)
        logger.info("Rolling volatility (window=%s):\n%s", window, rolling.tail())
    except Exception as e:
        logger.exception("Failed to compute volatility: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
