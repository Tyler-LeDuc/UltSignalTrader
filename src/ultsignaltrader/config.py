"""Configuration management for UltSignalTrader.

This module handles loading configuration from environment variables
and provides a centralized config object for the entire application.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv


@dataclass
class TradingConfig:
    """Trading configuration parameters."""
    mode: str
    symbol: str
    timeframe: str
    max_position_size: float
    stop_loss_percentage: float
    take_profit_percentage: float


@dataclass
class BinanceConfig:
    """Binance exchange configuration."""
    api_key: str
    api_secret: str
    testnet: bool = False


@dataclass
class BacktestConfig:
    """Backtesting configuration parameters."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission: float = 0.001  # 0.1% default commission


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    file_path: str
    max_bytes: int = 10_485_760  # 10MB
    backup_count: int = 5


class Config:
    """Main configuration class that loads and validates all settings."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration from environment variables.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in current directory.
        """
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Load trading config
        self.trading = TradingConfig(
            mode=os.getenv('TRADING_MODE', 'backtest').lower(),
            symbol=os.getenv('DEFAULT_SYMBOL', 'BTC/USDT'),
            timeframe=os.getenv('DEFAULT_TIMEFRAME', '1h'),
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', '0.1')),
            stop_loss_percentage=float(os.getenv('STOP_LOSS_PERCENTAGE', '2.0')),
            take_profit_percentage=float(os.getenv('TAKE_PROFIT_PERCENTAGE', '5.0'))
        )
        
        # Load Binance config
        self.binance = BinanceConfig(
            api_key=os.getenv('BINANCE_API_KEY', ''),
            api_secret=os.getenv('BINANCE_API_SECRET', ''),
            testnet=os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
        )
        
        # Load backtest config
        self.backtest = BacktestConfig(
            start_date=datetime.fromisoformat(os.getenv('BACKTEST_START_DATE', '2023-01-01')),
            end_date=datetime.fromisoformat(os.getenv('BACKTEST_END_DATE', '2023-12-31')),
            initial_capital=float(os.getenv('INITIAL_CAPITAL', '10000')),
            commission=float(os.getenv('BACKTEST_COMMISSION', '0.001'))
        )
        
        # Load logging config
        self.logging = LoggingConfig(
            level=os.getenv('LOG_LEVEL', 'INFO').upper(),
            file_path=os.getenv('LOG_FILE_PATH', 'logs/ultsignaltrader.log')
        )
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration values."""
        # Validate trading mode
        if self.trading.mode not in ['live', 'backtest']:
            raise ValueError(f"Invalid trading mode: {self.trading.mode}")
        
        # Validate live trading requirements
        if self.trading.mode == 'live':
            if not self.binance.api_key or not self.binance.api_secret:
                raise ValueError("API credentials required for live trading")
        
        # Validate numeric ranges
        if not 0 < self.trading.max_position_size <= 1:
            raise ValueError("max_position_size must be between 0 and 1")
        
        if self.trading.stop_loss_percentage < 0:
            raise ValueError("stop_loss_percentage must be positive")
        
        if self.trading.take_profit_percentage < 0:
            raise ValueError("take_profit_percentage must be positive")
        
        # Validate dates
        if self.backtest.start_date >= self.backtest.end_date:
            raise ValueError("Backtest start_date must be before end_date")
        
        # Validate logging level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.logging.level}")
    
    def is_live_trading(self) -> bool:
        """Check if running in live trading mode."""
        return self.trading.mode == 'live'
    
    def is_backtesting(self) -> bool:
        """Check if running in backtesting mode."""
        return self.trading.mode == 'backtest'


# Create a singleton config instance
config = Config()