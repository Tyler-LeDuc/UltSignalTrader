"""Backtesting engine for strategy evaluation.

Simulates trading on historical data to evaluate strategy performance.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..strategies.base import BaseStrategy, Signal, Trade, Position
from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    equity_curve: pd.Series
    trades: List[Trade]
    daily_returns: pd.Series
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy display."""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss
        }


class BacktestEngine:
    """Engine for running backtests on strategies."""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        """Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission: Trading commission as decimal (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
        
        logger.info(
            f"Initialized backtest engine with ${initial_capital:,.2f} capital "
            f"and {commission*100:.2f}% commission"
        )
    
    def reset(self):
        """Reset engine state."""
        self.cash = self.initial_capital
        self.positions = {}  # symbol -> Position
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
    
    def run(self, strategy: BaseStrategy, data: pd.DataFrame, 
            symbol: str = 'BTC/USDT', warmup_period: int = 50) -> BacktestResult:
        """Run backtest on historical data.
        
        Args:
            strategy: Strategy to test
            data: OHLCV data with datetime index
            symbol: Trading symbol
            warmup_period: Number of candles for strategy warmup
            
        Returns:
            Backtest results
        """
        logger.info(
            f"Starting backtest for {strategy.name} on {symbol} "
            f"from {data.index[0]} to {data.index[-1]}"
        )
        
        # Reset state
        self.reset()
        strategy.reset()
        
        # Validate data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Initialize strategy with warmup data
        if len(data) > warmup_period:
            warmup_data = data.iloc[:warmup_period]
            strategy.initialize(warmup_data)
            start_idx = warmup_period
        else:
            strategy.initialize(data)
            start_idx = 1
        
        # Override strategy's get_portfolio_value method
        strategy.get_portfolio_value = lambda: self._get_portfolio_value(data.iloc[-1])
        
        # Run through historical data
        for i in range(start_idx, len(data)):
            current_data = data.iloc[:i+1]
            current_candle = data.iloc[i]
            timestamp = data.index[i]
            
            # Update position current prices
            self._update_positions(current_candle)
            
            # Get signal from strategy
            signal_info = strategy.on_data(current_data)
            
            # Execute trade if signal generated
            if signal_info:
                self._execute_trade(
                    strategy=strategy,
                    signal_info=signal_info,
                    current_price=current_candle['close'],
                    symbol=symbol,
                    timestamp=timestamp
                )
            
            # Record equity
            portfolio_value = self._get_portfolio_value(current_candle)
            self.equity_curve.append(portfolio_value)
            self.timestamps.append(timestamp)
        
        # Calculate results
        result = self._calculate_results(data)
        
        logger.info(
            f"Backtest complete: Return {result.total_return_pct:.2f}%, "
            f"Sharpe {result.sharpe_ratio:.2f}, "
            f"Max DD {result.max_drawdown_pct:.2f}%"
        )
        
        return result
    
    def _execute_trade(self, strategy: BaseStrategy, signal_info: Dict,
                      current_price: float, symbol: str, timestamp: datetime):
        """Execute a trade based on signal.
        
        Args:
            strategy: Strategy instance
            signal_info: Signal information from strategy
            current_price: Current asset price
            symbol: Trading symbol
            timestamp: Current timestamp
        """
        signal = signal_info['signal']
        amount = signal_info['amount']
        
        if signal == Signal.BUY:
            # Check if we have enough cash
            cost = amount * current_price
            commission_cost = cost * self.commission
            total_cost = cost + commission_cost
            
            if total_cost > self.cash:
                # Adjust amount to available cash
                available_cash = self.cash * 0.99  # Leave 1% buffer
                amount = (available_cash / current_price) / (1 + self.commission)
                cost = amount * current_price
                commission_cost = cost * self.commission
                total_cost = cost + commission_cost
                
                logger.warning(
                    f"Insufficient cash. Adjusted buy amount to {amount:.4f}"
                )
            
            if amount > 0:
                # Execute buy
                self.cash -= total_cost
                
                # Create or update position
                if symbol in self.positions:
                    position = self.positions[symbol]
                    # Update average price
                    total_amount = position.amount + amount
                    avg_price = (
                        (position.amount * position.avg_entry_price + 
                         amount * current_price) / total_amount
                    )
                    position.amount = total_amount
                    position.avg_entry_price = avg_price
                    position.timestamp = timestamp
                else:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        amount=amount,
                        avg_entry_price=current_price,
                        current_price=current_price,
                        timestamp=timestamp
                    )
                
                # Record trade
                trade = Trade(
                    symbol=symbol,
                    side='buy',
                    amount=amount,
                    price=current_price,
                    timestamp=timestamp,
                    commission=commission_cost
                )
                self.trades.append(trade)
                strategy.on_trade(trade)
                
                logger.info(
                    f"BUY {amount:.4f} {symbol} @ ${current_price:.2f} "
                    f"(commission: ${commission_cost:.2f})"
                )
        
        elif signal == Signal.SELL:
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Limit sell amount to position size
                amount = min(amount, position.amount)
                
                if amount > 0:
                    # Calculate proceeds
                    proceeds = amount * current_price
                    commission_cost = proceeds * self.commission
                    net_proceeds = proceeds - commission_cost
                    
                    # Update cash and position
                    self.cash += net_proceeds
                    position.amount -= amount
                    
                    # Remove position if fully sold
                    if position.amount <= 0:
                        del self.positions[symbol]
                    
                    # Record trade
                    trade = Trade(
                        symbol=symbol,
                        side='sell',
                        amount=amount,
                        price=current_price,
                        timestamp=timestamp,
                        commission=commission_cost
                    )
                    self.trades.append(trade)
                    strategy.on_trade(trade)
                    
                    logger.info(
                        f"SELL {amount:.4f} {symbol} @ ${current_price:.2f} "
                        f"(commission: ${commission_cost:.2f})"
                    )
    
    def _update_positions(self, candle: pd.Series):
        """Update position current prices.
        
        Args:
            candle: Current candle data
        """
        for position in self.positions.values():
            position.current_price = candle['close']
    
    def _get_portfolio_value(self, candle: pd.Series) -> float:
        """Calculate total portfolio value.
        
        Args:
            candle: Current candle data
            
        Returns:
            Total portfolio value
        """
        total_value = self.cash
        
        for position in self.positions.values():
            position_value = position.amount * candle['close']
            total_value += position_value
        
        return total_value
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate backtest performance metrics.
        
        Args:
            data: Full OHLCV data
            
        Returns:
            Backtest results
        """
        equity_series = pd.Series(self.equity_curve, index=self.timestamps)
        
        # Basic metrics
        final_capital = equity_series.iloc[-1] if len(equity_series) > 0 else self.initial_capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Daily returns
        daily_equity = equity_series.resample('D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        
        # Sharpe ratio (annualized)
        if len(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = max_drawdown * 100
        
        # Trade statistics
        wins = []
        losses = []
        
        # Calculate P&L for each round trip
        buy_trades = [t for t in self.trades if t.side == 'buy']
        sell_trades = [t for t in self.trades if t.side == 'sell']
        
        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                buy = buy_trades[i]
                pnl = (sell.price - buy.price) * sell.amount - buy.commission - sell.commission
                if pnl > 0:
                    wins.append(pnl)
                else:
                    losses.append(abs(pnl))
        
        total_trades = len(buy_trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        
        # Calculate trade metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else 0.0
        
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=equity_series,
            trades=self.trades,
            daily_returns=daily_returns
        )