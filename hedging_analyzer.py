import logging
import sys
from datetime import datetime, timedelta
from typing import Union, Optional, Dict

import click
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pykalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Annualization scaling factors
SCALING_FACTORS = {
    '1h': np.sqrt(8760),
    '4h': np.sqrt(8760 / 4),
    '8h': np.sqrt(8760 / 8),
}

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

class VolatilityCalculator:
    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.interval = '1h'
        self.session = session or create_session()
        self.base_url = 'https://api.binance.com'

    def datetime_to_milliseconds(self, date: Union[datetime, int, float]) -> int:
        if isinstance(date, datetime):
            return int(date.timestamp() * 1000)
        if isinstance(date, (int, float)):
            return int(date)
        raise TypeError("date must be datetime or timestamp int/float")

    def fetch_data(self, symbol: str, start_ts: Union[datetime, int, float], end_ts: Union[datetime, int, float]) -> pd.DataFrame:
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
        df = pd.DataFrame(data, columns=cols)
        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df['r_1h'] = np.log(df['close'] / df['close'].shift(1))
        df['r_4h'] = df['r_1h'].rolling(window=4).sum()
        df['r_8h'] = df['r_1h'].rolling(window=8).sum()
        return df

    def calculate_realized_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        vol_1h = df['r_1h'].std(skipna=True) * SCALING_FACTORS['1h']
        vol_4h = df['r_4h'].std(skipna=True) * SCALING_FACTORS['4h']
        vol_8h = df['r_8h'].std(skipna=True) * SCALING_FACTORS['8h']
        return {
            'vol_1h': vol_1h,
            'vol_4h': vol_4h,
            'vol_8h': vol_8h
        }

    def calculate_mean_reversion_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        acorrs = {
            f'autocorr_{lag}h': df['r_1h'].autocorr(lag=lag) for lag in [1, 2, 3]
        }
        vr_4h = df['r_4h'].var() / (4 * df['r_1h'].var())
        vr_8h = df['r_8h'].var() / (8 * df['r_1h'].var())
        return {**acorrs, 'variance_ratio_4h': vr_4h, 'variance_ratio_8h': vr_8h}

def apply_kalman_filter(series: pd.Series) -> pd.Series:
    series = np.log(series)
    mu = series.mean()
    phi = 0.95
    q = 1e-4
    r = 1e-2

    kf = KalmanFilter(
        transition_matrices=[phi],
        transition_offsets=[(1 - phi) * mu],
        observation_matrices=[1],
        initial_state_mean=series.iloc[0],
        transition_covariance=q,
        observation_covariance=r
    )
    state_means, _ = kf.smooth(series.values)
    return pd.Series(state_means.flatten(), index=series.index)


def analyze_kalman_residuals(series: pd.Series) -> Dict[str, float]:
    log_price = np.log(series)
    filtered = apply_kalman_filter(series)
    residuals = log_price - filtered
    autocorr = residuals.autocorr(lag=1)
    adf_result = adfuller(residuals.dropna())
    return {
        'kalman_autocorr_1': autocorr,
        'kalman_adf_stat': adf_result[0],
        'kalman_adf_pvalue': adf_result[1]
    }



@click.command()
@click.option('--symbol', default='ETHUSDT', help='Trading pair symbol (default: ETHUSDT)')
def main(symbol: str) -> None:
    """
    CLI to calculate volatility and mean-reversion metrics for a trading pair.
    - Uses last 30 days for returns/volatility analysis
    - Uses last 7 days for Kalman filter residual diagnostics
    """
    vc = VolatilityCalculator()
    end_30d = datetime.now()
    start_30d = end_30d - timedelta(days=30)

    try:
        # Full 30-day data
        df_full = vc.fetch_data(symbol, start_30d, end_30d)
        df_full = vc.process_data(df_full)

        # Use last 7 days only for Kalman
        df_kf = df_full.loc[df_full.index >= (end_30d - timedelta(days=7))]

        vol = vc.calculate_realized_volatility(df_full)
        mr = vc.calculate_mean_reversion_metrics(df_full)
        kalman_stats = analyze_kalman_residuals(df_kf['close'])

        print(f"\nAnnualized Realized Volatility for {symbol} (30-day window):")
        for k, v in vol.items():
            print(f"{k}: {v:.2%}")

        print("\nMean Reversion Metrics (30-day window):")
        for k, v in mr.items():
            print(f"{k}: {v:.4f}")

        print("\nKalman Filter Residual Diagnostics (7-day window):")
        for k, v in kalman_stats.items():
            print(f"{k}: {v:.4f}")

        print("\nInterpretation:")
        if kalman_stats['kalman_autocorr_1'] < 0 and kalman_stats['kalman_adf_pvalue'] < 0.05:
            print("- Residuals are significantly mean-reverting → Likely mean-reverting price process.")
        elif kalman_stats['kalman_autocorr_1'] > 0 and kalman_stats['kalman_adf_pvalue'] > 0.1:
            print("- Residuals are persistent and non-stationary → Likely trending price process.")
        else:
            print("- Mixed signal; possibly weak mean reversion.")

    except Exception as e:
        logger.error("Error during processing: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
