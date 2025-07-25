#!/usr/bin/env python3
"""UltSignalTrader - Production-grade crypto trading bot.

Main entry point that supports both live trading and backtesting modes.
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
import pandas as pd

from src.ultsignaltrader.config import Config
from src.ultsignaltrader.utils.logger import Logger, get_logger
from src.ultsignaltrader.strategies.ema_crossover import EMACrossoverStrategy
from src.ultsignaltrader.live.binance_connector import BinanceConnector
from src.ultsignaltrader.backtests.engine import BacktestEngine
from src.ultsignaltrader.backtests.data_loader import DataLoader, download_sample_data
from src.ultsignaltrader.strategies.base import Signal


# Setup logging
Logger.setup_global_logging()
logger = get_logger(__name__)


class TradingBot:
    """Main trading bot controller."""
    
    def __init__(self, config: Config):
        """Initialize trading bot.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.strategy = None
        self.connector = None
        self.running = False
        
    def initialize_strategy(self, strategy_name: str = 'ema_crossover'):
        """Initialize trading strategy.
        
        Args:
            strategy_name: Name of strategy to use
        """
        logger.info(f"Initializing strategy: {strategy_name}")
        
        if strategy_name.lower() == 'ema_crossover':
            self.strategy = EMACrossoverStrategy(
                fast_period=12,
                slow_period=26,
                position_size_pct=self.config.trading.max_position_size
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        logger.info(f"Strategy initialized: {self.strategy.name}")
    
    def run_live_trading(self):
        """Run live trading mode."""
        logger.info("Starting live trading mode")
        
        # Initialize Binance connector
        self.connector = BinanceConnector(self.config.binance)
        
        # Test connection
        if not self.connector.test_connection():
            logger.error("Failed to connect to Binance")
            return
        
        # Get initial balance
        balance = self.connector.get_balance()
        logger.info(f"Account balance: {balance}")
        
        # Warmup strategy with historical data
        symbol = self.config.trading.symbol
        timeframe = self.config.trading.timeframe
        
        logger.info(f"Fetching historical data for strategy warmup...")
        historical_data = self.connector.get_ohlcv(symbol, timeframe, limit=100)
        self.strategy.initialize(historical_data)
        
        # Override strategy's get_portfolio_value
        self.strategy.get_portfolio_value = lambda: self.connector.get_portfolio_value()
        
        # Start trading loop
        self.running = True
        logger.info(f"Starting trading loop for {symbol} {timeframe}")
        
        while self.running:
            try:
                # Fetch latest data
                data = self.connector.get_ohlcv(symbol, timeframe, limit=100)
                
                # Get trading signal
                signal_info = self.strategy.on_data(data)
                
                if signal_info:
                    self._execute_live_trade(signal_info, symbol)
                
                # Sleep based on timeframe
                sleep_seconds = self._get_sleep_seconds(timeframe)
                logger.info(f"Sleeping for {sleep_seconds} seconds...")
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logger.info("Trading interrupted by user")
                self.running = False
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _execute_live_trade(self, signal_info: dict, symbol: str):
        """Execute live trade based on signal.
        
        Args:
            signal_info: Signal information from strategy
            symbol: Trading symbol
        """
        signal = signal_info['signal']
        amount = signal_info['amount']
        reason = signal_info['reason']
        
        logger.info(f"Signal: {signal.value} - {reason}")
        
        try:
            if signal == Signal.BUY:
                trade = self.connector.place_market_order(symbol, 'buy', amount)
                self.strategy.on_trade(trade)
                logger.info(f"Buy order executed: {trade}")
                
            elif signal == Signal.SELL:
                trade = self.connector.place_market_order(symbol, 'sell', amount)
                self.strategy.on_trade(trade)
                logger.info(f"Sell order executed: {trade}")
                
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
    
    def _get_sleep_seconds(self, timeframe: str) -> int:
        """Get sleep duration based on timeframe.
        
        Args:
            timeframe: Trading timeframe
            
        Returns:
            Sleep duration in seconds
        """
        timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return timeframe_seconds.get(timeframe, 3600)
    
    def run_backtest(self, data_file: str = None):
        """Run backtesting mode.
        
        Args:
            data_file: Path to CSV file with historical data
        """
        logger.info("Starting backtest mode")
        
        # Load or download data
        if data_file:
            logger.info(f"Loading data from {data_file}")
            data = DataLoader.load_csv(
                data_file,
                start_date=self.config.backtest.start_date,
                end_date=self.config.backtest.end_date
            )
        else:
            # Download sample data
            logger.info("No data file provided, downloading sample data...")
            symbol = self.config.trading.symbol.replace('/', '')
            data_file = download_sample_data(
                symbol=symbol,
                timeframe=self.config.trading.timeframe,
                start_date=self.config.backtest.start_date.strftime('%Y-%m-%d')
            )
            data = DataLoader.load_csv(data_file)
        
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=self.config.backtest.initial_capital,
            commission=self.config.backtest.commission
        )
        
        # Run backtest
        result = engine.run(
            strategy=self.strategy,
            data=data,
            symbol=self.config.trading.symbol
        )
        
        # Display results
        self._display_backtest_results(result)
        
        # Optionally plot equity curve
        self._plot_equity_curve(result)
    
    def _display_backtest_results(self, result):
        """Display backtest results.
        
        Args:
            result: BacktestResult object
        """
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        metrics = result.to_dict()
        
        # Format and display metrics
        print(f"\nPerformance Metrics:")
        print(f"  Initial Capital:     ${metrics['initial_capital']:,.2f}")
        print(f"  Final Capital:       ${metrics['final_capital']:,.2f}")
        print(f"  Total Return:        ${metrics['total_return']:,.2f}")
        print(f"  Total Return %:      {metrics['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        
        print(f"\nTrade Statistics:")
        print(f"  Total Trades:        {metrics['total_trades']}")
        print(f"  Winning Trades:      {metrics['winning_trades']}")
        print(f"  Losing Trades:       {metrics['losing_trades']}")
        print(f"  Win Rate:            {metrics['win_rate']*100:.1f}%")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"  Average Win:         ${metrics['avg_win']:.2f}")
        print(f"  Average Loss:        ${metrics['avg_loss']:.2f}")
        
        # Display strategy metrics
        strategy_metrics = self.strategy.get_strategy_metrics()
        print(f"\nStrategy Parameters:")
        for key, value in strategy_metrics.items():
            if key not in metrics:
                print(f"  {key}:              {value}")
        
        print("="*60 + "\n")
    
    def _plot_equity_curve(self, result):
        """Plot equity curve.
        
        Args:
            result: BacktestResult object
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Equity curve
            ax1.plot(result.equity_curve, label='Portfolio Value')
            ax1.axhline(y=result.initial_capital, color='r', linestyle='--', 
                       label='Initial Capital')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.set_title('Equity Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            rolling_max = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - rolling_max) / rolling_max * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, 
                           color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.set_title('Drawdown')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.info("Matplotlib not installed, skipping plot")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='UltSignalTrader - Crypto Trading Bot'
    )
    parser.add_argument(
        '--mode', 
        choices=['live', 'backtest'], 
        help='Trading mode (overrides config file)'
    )
    parser.add_argument(
        '--strategy',
        default='ema_crossover',
        help='Strategy to use'
    )
    parser.add_argument(
        '--data',
        help='CSV file for backtesting'
    )
    parser.add_argument(
        '--config',
        help='Path to .env config file'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Override mode if specified
        if args.mode:
            config.trading.mode = args.mode
        
        logger.info(f"Starting UltSignalTrader in {config.trading.mode} mode")
        
        # Initialize bot
        bot = TradingBot(config)
        bot.initialize_strategy(args.strategy)
        
        # Run appropriate mode
        if config.is_live_trading():
            bot.run_live_trading()
        else:
            bot.run_backtest(args.data)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()