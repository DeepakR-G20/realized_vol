import requests
import pandas as pd
import numpy as np
import datetime as dt
import click
import matplotlib.pyplot as plt

class VolatilityCalculator:
    def __init__(self, interval):
        self.base_url = 'https://api.binance.com'
        self.interval = interval
        self.annual_scaling = self._compute_annual_scaling(interval)

    def _compute_annual_scaling(self, interval):
        mapping = {
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
        freq = mapping.get(interval)
        if not freq:
            raise ValueError(f"Unsupported interval: {interval}")
        return np.sqrt(freq)

    def datetime_to_milliseconds(self, date):
        if isinstance(date, dt.datetime):
            return int(date.timestamp() * 1000)
        elif isinstance(date, (int, float)):
            return int(date)
        else:
            raise ValueError("Invalid datetime format.")

    def fetch_data(self, symbol, interval, startTime, endTime, limit=1000):
        url = (f"{self.base_url}/api/v3/klines"
               f"?symbol={symbol}&interval={interval}"
               f"&startTime={startTime}&endTime={endTime}&limit={limit}")
        response = requests.get(url)
        response.raise_for_status()
        df = pd.DataFrame(response.json(), columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        return df

    def process_data(self, df):
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        df['r'] = np.log(df['close'] / df['close'].shift(1))
        df['v2_park'] = (np.log(df['high'] / df['low']) ** 2) / (4 * np.log(2))
        df['v2_gk'] = (
            0.511 * np.log(df['high'] / df['low'])**2
            - 0.019 * np.log(df['close'] / df['open']) * np.log((df['high'] * df['low']) / df['open']**2)
            - 0.383 * np.log(df['close'] / df['open'])**2
        )
        df['v2_rs'] = (
            np.log(df['high'] / df['open']) * np.log(df['high'] / df['close'])
            + np.log(df['low'] / df['open']) * np.log(df['low'] / df['close'])
        )
        df['bv_term'] = np.abs(df['r']) * np.abs(df['r'].shift(1))
        return df

    def aggregate_daily(self, df):
        intervals_per_day = df.index.to_series().groupby(df.index.date).count()

        daily = pd.DataFrame({
            'RV': df['r'].pow(2).groupby(df.index.date).sum(),
            'BV': (np.pi/2 * df['bv_term']).groupby(df.index.date).sum(),
            'Parkinson': df['v2_park'].groupby(df.index.date).sum(),
            'GarmanKlass': df['v2_gk'].groupby(df.index.date).sum(),
            'RogersSatchell': df['v2_rs'].groupby(df.index.date).sum(),
        })

        # Normalize by count per day (to get per-interval variance)
        for col in ['RV','BV','Parkinson','GarmanKlass','RogersSatchell']:
            daily[col] = daily[col] / intervals_per_day

        daily['JumpVar'] = daily['RV'] - daily['BV']

        for col in ['RV','BV','Parkinson','GarmanKlass','RogersSatchell','JumpVar']:
            daily[f'{col}_ann'] = np.sqrt(daily[col]) * self.annual_scaling

        daily.index = pd.to_datetime(daily.index)
        return daily


    def intraday_pattern(self, df):
        df['hour'] = df.index.hour
        count_per_hour = df.groupby('hour')['r'].count()
        intraday = pd.DataFrame({
            'RV_hour': df.groupby('hour')['r'].apply(lambda x: np.sum(x**2)),
            'Park_hour': df.groupby('hour')['v2_park'].sum(),
            'GK_hour': df.groupby('hour')['v2_gk'].sum(),
            'RS_hour': df.groupby('hour')['v2_rs'].sum()
        })
        for col in ['RV_hour', 'Park_hour', 'GK_hour', 'RS_hour']:
            intraday[col] = intraday[col] / count_per_hour
            intraday[f'{col}_ann'] = np.sqrt(intraday[col]) * self.annual_scaling
        return intraday

    def rolling_volatility(self, df, window=24):
        roll = df['r'].pow(2).rolling(window).sum()
        sigma = np.sqrt(roll) * np.sqrt(self.annual_scaling**2 / window)
        return sigma

@click.command()
@click.option('--symbol', required=True, help='Ticker symbol, e.g., ETHUSDT')
@click.option('--interval', default='1h', help='Interval, e.g., 1m, 5m, 1h, 1d')
@click.option('--window', default=168, type=int, help='Rolling window size in hours')
@click.option('--start', required=True, help='Start datetime, e.g., 2024-06-01T00:00:00')
@click.option('--end', required=True, help='End datetime, e.g., 2024-07-01T00:00:00')
def main(symbol, interval, window,start, end):
    start_dt = dt.datetime.fromisoformat(start)
    end_dt = dt.datetime.fromisoformat(end)

    calc = VolatilityCalculator(interval)
    raw = calc.fetch_data(symbol, interval,
                          calc.datetime_to_milliseconds(start_dt),
                          calc.datetime_to_milliseconds(end_dt))
    df = calc.process_data(raw)
    daily = calc.aggregate_daily(df)
    intraday = calc.intraday_pattern(df)
    rolling = calc.rolling_volatility(df, window)

    print("\n--- Daily Vol ---")
    print(daily.tail())
    print("\n--- Intraday Pattern ---")
    print(intraday)
    print(f"\n--- Rolling Vol (window={window}) ---")
    print(rolling.tail())


if __name__ == '__main__':
    main()
