"""Data loading utilities for backtesting.

Handles loading historical data from CSV files and preparing it
for backtesting.
"""

import os
from datetime import datetime
from typing import Optional, Union
import pandas as pd

from ..utils.logger import get_logger


logger = get_logger(__name__)


class DataLoader:
    """Loads and prepares historical data for backtesting."""
    
    @staticmethod
    def load_csv(file_path: str, 
                 start_date: Optional[Union[str, datetime]] = None,
                 end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Load OHLCV data from CSV file.
        
        Expected CSV format:
        - Columns: timestamp/date, open, high, low, close, volume
        - timestamp can be Unix timestamp or datetime string
        
        Args:
            file_path: Path to CSV file
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            DataFrame with OHLCV data and datetime index
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        timestamp_cols = ['timestamp', 'date', 'datetime', 'time']
        
        # Find timestamp column
        timestamp_col = None
        for col in timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break
        
        if not timestamp_col:
            raise ValueError(f"No timestamp column found. Expected one of: {timestamp_cols}")
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        if df[timestamp_col].dtype == 'int64' or df[timestamp_col].dtype == 'float64':
            # Unix timestamp
            df['datetime'] = pd.to_datetime(df[timestamp_col], unit='ms')
        else:
            # String datetime
            df['datetime'] = pd.to_datetime(df[timestamp_col])
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Keep only OHLCV columns
        df = df[required_cols]
        
        # Filter by date range if specified
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]
        
        # Validate data
        DataLoader._validate_ohlcv(df)
        
        logger.info(
            f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}"
        )
        
        return df
    
    @staticmethod
    def load_binance_csv(file_path: str,
                        start_date: Optional[Union[str, datetime]] = None,
                        end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Load Binance format CSV data.
        
        Binance CSV format:
        - Columns: Open time, Open, High, Low, Close, Volume, Close time,
                  Quote asset volume, Number of trades, Taker buy base asset volume,
                  Taker buy quote asset volume, Ignore
        
        Args:
            file_path: Path to Binance CSV file
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Loading Binance format data from {file_path}")
        
        # Binance column names
        binance_columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ]
        
        # Load CSV with specific columns
        df = pd.read_csv(file_path, names=binance_columns)
        
        # Convert timestamp to datetime (milliseconds)
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Keep only OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter by date range
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]
        
        # Validate
        DataLoader._validate_ohlcv(df)
        
        return df
    
    @staticmethod
    def resample_data(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to different timeframe.
        
        Args:
            data: OHLCV DataFrame
            timeframe: Target timeframe (e.g., '1H', '4H', '1D')
            
        Returns:
            Resampled DataFrame
        """
        # Resample rules for OHLCV
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        resampled = data.resample(timeframe).agg(agg_rules)
        resampled.dropna(inplace=True)
        
        logger.info(
            f"Resampled {len(data)} rows to {len(resampled)} rows "
            f"at {timeframe} timeframe"
        )
        
        return resampled
    
    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame):
        """Validate OHLCV data integrity.
        
        Args:
            df: OHLCV DataFrame
        """
        # Check for NaN values
        if df.isnull().any().any():
            nan_counts = df.isnull().sum()
            logger.warning(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
            df.dropna(inplace=True)
        
        # Check OHLC relationships
        invalid_candles = df[
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ]
        
        if len(invalid_candles) > 0:
            logger.warning(f"Found {len(invalid_candles)} invalid candles")
            # Fix by adjusting high/low
            df.loc[invalid_candles.index, 'high'] = df.loc[invalid_candles.index, ['open', 'close', 'high']].max(axis=1)
            df.loc[invalid_candles.index, 'low'] = df.loc[invalid_candles.index, ['open', 'close', 'low']].min(axis=1)
        
        # Check for zero or negative prices
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            logger.warning("Found zero or negative prices")
            df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            logger.warning(f"Found {df.index.duplicated().sum()} duplicate timestamps")
            df = df[~df.index.duplicated(keep='first')]


def download_sample_data(symbol: str = 'BTCUSDT', 
                        timeframe: str = '1h',
                        start_date: str = '2023-01-01',
                        output_dir: str = 'data/') -> str:
    """Download sample data from Binance (requires ccxt).
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        timeframe: Timeframe (e.g., '1h', '1d')
        start_date: Start date for download
        output_dir: Directory to save data
        
    Returns:
        Path to saved CSV file
    """
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt required for downloading data. Install with: pip install ccxt")
        raise
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize exchange
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Convert symbol format
    ccxt_symbol = symbol.replace('USDT', '/USDT')
    
    # Fetch OHLCV data
    logger.info(f"Downloading {ccxt_symbol} {timeframe} data from Binance...")
    
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    all_candles = []
    
    while True:
        candles = exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=1000)
        if not candles:
            break
        
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        
        # Stop if we reach current time
        if since > exchange.milliseconds():
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # Save to CSV
    filename = f"{symbol}_{timeframe}_{start_date}.csv"
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    
    logger.info(f"Saved {len(df)} candles to {file_path}")
    return file_path