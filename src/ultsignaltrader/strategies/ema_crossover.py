"""EMA Crossover Strategy implementation.

A simple trend-following strategy that generates signals based on
exponential moving average crossovers.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .base import BaseStrategy, Signal, Position
from ..utils.logger import get_logger


logger = get_logger(__name__)


class EMACrossoverStrategy(BaseStrategy):
    """Exponential Moving Average Crossover Strategy.
    
    Generates BUY signals when fast EMA crosses above slow EMA,
    and SELL signals when fast EMA crosses below slow EMA.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 position_size_pct: float = 0.1):
        """Initialize EMA Crossover strategy.
        
        Args:
            fast_period: Period for fast EMA (default: 12)
            slow_period: Period for slow EMA (default: 26)
            position_size_pct: Position size as percentage of portfolio (default: 0.1)
        """
        super().__init__(
            name="EMA_Crossover",
            params={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'position_size_pct': position_size_pct
            }
        )
        
        # Validate parameters
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size_pct = position_size_pct
        
        # Track previous EMAs for crossover detection
        self._prev_fast_ema = None
        self._prev_slow_ema = None
        self._last_signal = Signal.HOLD
        self._signal_reason = ""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA indicators.
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with EMA indicators added
        """
        df = data.copy()
        
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate EMA difference and percentage
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_diff_pct'] = (df['ema_diff'] / df['ema_slow']) * 100
        
        return df
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on EMA crossover.
        
        Args:
            data: OHLCV data with indicators
            
        Returns:
            Trading signal
        """
        # Get current and previous EMA values
        current_fast = data['ema_fast'].iloc[-1]
        current_slow = data['ema_slow'].iloc[-1]
        
        # Need at least 2 data points for crossover
        if len(data) < 2:
            self._signal_reason = "Insufficient data for crossover detection"
            return Signal.HOLD
        
        prev_fast = data['ema_fast'].iloc[-2]
        prev_slow = data['ema_slow'].iloc[-2]
        
        # Check for crossover
        signal = Signal.HOLD
        
        # Bullish crossover: fast EMA crosses above slow EMA
        if prev_fast <= prev_slow and current_fast > current_slow:
            signal = Signal.BUY
            self._signal_reason = (
                f"Bullish crossover: Fast EMA ({current_fast:.2f}) "
                f"crossed above Slow EMA ({current_slow:.2f})"
            )
            logger.info(self._signal_reason)
        
        # Bearish crossover: fast EMA crosses below slow EMA
        elif prev_fast >= prev_slow and current_fast < current_slow:
            signal = Signal.SELL
            self._signal_reason = (
                f"Bearish crossover: Fast EMA ({current_fast:.2f}) "
                f"crossed below Slow EMA ({current_slow:.2f})"
            )
            logger.info(self._signal_reason)
        
        else:
            # Additional filters to reduce false signals
            ema_diff_pct = data['ema_diff_pct'].iloc[-1]
            
            if abs(ema_diff_pct) < 0.1:  # EMAs too close
                self._signal_reason = f"EMAs too close (diff: {ema_diff_pct:.2f}%)"
            elif current_fast > current_slow:
                self._signal_reason = f"Uptrend continues (Fast EMA above Slow EMA)"
            else:
                self._signal_reason = f"Downtrend continues (Fast EMA below Slow EMA)"
        
        self._last_signal = signal
        return signal
    
    def calculate_position_size(self, 
                              signal: Signal,
                              current_price: float,
                              portfolio_value: float,
                              current_position: Optional[Position] = None) -> float:
        """Calculate position size based on Kelly Criterion or fixed percentage.
        
        Args:
            signal: Trading signal
            current_price: Current asset price
            portfolio_value: Total portfolio value
            current_position: Current position if any
            
        Returns:
            Position size in base asset units
        """
        # For HOLD signal, no position change
        if signal == Signal.HOLD:
            return 0.0
        
        # Calculate position value
        position_value = portfolio_value * self.position_size_pct
        
        # For BUY signal
        if signal == Signal.BUY:
            # Don't buy if already have position
            if current_position and current_position.amount > 0:
                logger.info("Already in position, skipping BUY signal")
                return 0.0
            
            # Calculate amount to buy
            amount = position_value / current_price
            logger.info(f"BUY position size: {amount:.4f} units at ${current_price:.2f}")
            return amount
        
        # For SELL signal
        elif signal == Signal.SELL:
            # Only sell if we have a position
            if not current_position or current_position.amount <= 0:
                logger.info("No position to sell")
                return 0.0
            
            # Sell entire position
            amount = current_position.amount
            logger.info(f"SELL position size: {amount:.4f} units at ${current_price:.2f}")
            return amount
        
        return 0.0
    
    def get_signal_reason(self) -> str:
        """Get explanation for the last signal."""
        return self._signal_reason
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy-specific metrics.
        
        Returns:
            Dictionary of metrics
        """
        trades = self.get_trades()
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        # Group trades by buy/sell pairs
        wins = []
        losses = []
        
        # Simple P&L calculation (pairs of buy/sell)
        buy_trades = [t for t in trades if t.side == 'buy']
        sell_trades = [t for t in trades if t.side == 'sell']
        
        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                buy = buy_trades[i]
                pnl = (sell.price - buy.price) * sell.amount
                if pnl > 0:
                    wins.append(pnl)
                else:
                    losses.append(abs(pnl))
        
        total_trades = len(buy_trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0.0,
            'avg_win': np.mean(wins) if wins else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'profit_factor': sum(wins) / sum(losses) if losses and sum(losses) > 0 else 0.0,
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'position_size_pct': self.position_size_pct
        }
        
        return metrics