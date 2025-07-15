import logging
import sys
import json
from datetime import datetime, timedelta, time
from typing import Union, Optional, Dict, Tuple

import click
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Type aliases for clarity
TimeSpan = Tuple[time, time]
TZDefs = Dict[str, TimeSpan]

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def create_session(retries: int = 3, backoff: float = 0.3) -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

class TimeZoneAnalyzer:
    # Default time windows in UTC
    DEFAULT_TZ_DEFS: TZDefs = {
        'asia':   (time(23, 0), time(7, 0)),   # 23:00–07:00 UTC
        'europe': (time(7, 0),  time(15, 0)),  # 07:00–15:00 UTC
        'us':     (time(15, 0), time(23, 0)),  # 15:00–23:00 UTC
    }

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        tz_defs: Optional[TZDefs] = None
    ) -> None:
        self.session = session or create_session()
        # Merge defaults with overrides
        self.tz_defs = {**self.DEFAULT_TZ_DEFS, **(tz_defs or {})}
        self.base_url = 'https://api.binance.com'
        self.interval = '1h'

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
        end_ts: Union[datetime, int, float]
    ) -> pd.DataFrame:
        start_ms = self.datetime_to_milliseconds(start_ts)
        end_ms = self.datetime_to_milliseconds(end_ts)
        url = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': self.interval,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 1000
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
        return pd.DataFrame(data, columns=cols)

    @staticmethod
    def in_span(t: time, span: TimeSpan) -> bool:
        start, end = span
        if start < end:
            return start <= t < end
        # wraparound across midnight
        return t >= start or t < end

    def which_session(self, dt: datetime) -> str:
        t = dt.time()
        for name, span in self.tz_defs.items():
            if self.in_span(t, span):
                return name
        return 'unknown'

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert epoch ms to datetime and set index
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        # Cast numeric columns
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        # Hourly log-return
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        # Timezone label
        df['timezone'] = df.index.to_series().apply(self.which_session)
        return df.dropna(subset=['log_return'])

    def hourly_avg_log_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average log-return for each hour of day (0–23 UTC)."""
        result = df.groupby(df.index.hour)['log_return'].mean().rename('avg_log_return').to_frame()
        return result

    def timezone_avg_daily_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average total daily return per timezone."""
        # Sum log returns per day and timezone
        daily = (
            df
            .groupby([df.index.date, 'timezone'])['log_return']
            .sum()
            .reset_index(name='daily_return')
        )
        # Average across days per timezone
        result = daily.groupby('timezone')['daily_return'].mean().rename('avg_daily_return').to_frame()
        return result

@click.command()
@click.option('--symbol', type=str, required=True, help='Trading pair symbol, e.g. BTCUSDT')
@click.option('--start-date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True,
              help='Start date (YYYY-MM-DD)')
@click.option('--end-date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True,
              help='End date (YYYY-MM-DD)')
@click.option('--tz-defs', type=str, default=None,
              help='JSON of timezone defs, e.g. {"asia":["23:00","07:00"],...}')
def main(start_date, end_date, symbol, tz_defs):
    # Parse timezone definitions if provided
    parsed_tz = None
    if tz_defs:
        raw = json.loads(tz_defs)
        parsed_tz = {
            name: (time.fromisoformat(s), time.fromisoformat(e))
            for name, (s, e) in raw.items()
        }
    analyzer = TimeZoneAnalyzer(tz_defs=parsed_tz)
    df = analyzer.fetch_data(symbol, start_date, end_date + timedelta(days=1))
    processed = analyzer.process_data(df)

    # Compute and display results
    hourly = analyzer.hourly_avg_log_return(processed)
    tz_daily = analyzer.timezone_avg_daily_return(processed)

    click.echo("\nAverage log-return per UTC hour:")
    click.echo(hourly.to_string())
    click.echo("\nAverage daily return per timezone:")
    click.echo(tz_daily.to_string())

if __name__ == '__main__':
    main()
