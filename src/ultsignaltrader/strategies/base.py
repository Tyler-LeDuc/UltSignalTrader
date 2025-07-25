"""Base strategy interface for UltSignalTrader.

All trading strategies must inherit from BaseStrategy and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    timestamp: datetime
    order_id: Optional[str] = None
    commission: Optional[float] = None
    
    @property
    def value(self) -> float:
        """Calculate trade value."""
        return self.amount * self.price


@dataclass
class Position:
    """Represents current position in an asset."""
    symbol: str
    amount: float
    avg_entry_price: float
    current_price: float
    timestamp: datetime
    
    @property
    def value(self) -> float:
        """Current position value."""
        return self.amount * self.current_price
    
    @property
    def pnl(self) -> float:
        """Profit/Loss in base currency."""
        return (self.current_price - self.avg_entry_price) * self.amount
    
    @property
    def pnl_percent(self) -> float:
        """Profit/Loss percentage."""
        if self.avg_entry_price == 0:
            return 0
        return ((self.current_price - self.avg_entry_price) / self.avg_entry_price) * 100


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """Initialize strategy.
        
        Args:
            name: Strategy name
            params: Strategy-specific parameters
        """
        self.name = name
        self.params = params or {}
        self._position: Optional[Position] = None
        self._trades: List[Trade] = []
        self._is_initialized = False
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators needed for the strategy.
        
        Args:
            data: OHLCV data with columns: open, high, low, close, volume
            
        Returns:
            DataFrame with original data and calculated indicators
        """
        pass
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on current market data.
        
        Args:
            data: OHLCV data with indicators
            
        Returns:
            Trading signal (BUY, SELL, or HOLD)
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, 
                              signal: Signal,
                              current_price: float,
                              portfolio_value: float,
                              current_position: Optional[Position] = None) -> float:
        """Calculate position size for the trade.
        
        Args:
            signal: Trading signal
            current_price: Current asset price
            portfolio_value: Total portfolio value
            current_position: Current position if any
            
        Returns:
            Position size (amount to buy/sell)
        """
        pass
    
    def initialize(self, initial_data: pd.DataFrame):
        """Initialize strategy with historical data.
        
        Args:
            initial_data: Initial OHLCV data for warmup
        """
        # Calculate initial indicators
        self.calculate_indicators(initial_data)
        self._is_initialized = True
    
    def on_data(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Process new market data and generate trading decisions.
        
        Args:
            data: Latest OHLCV data
            
        Returns:
            Trading decision dict or None if no action needed
        """
        if not self._is_initialized:
            raise RuntimeError("Strategy not initialized. Call initialize() first.")
        
        # Calculate indicators
        data_with_indicators = self.calculate_indicators(data)
        
        # Generate signal
        signal = self.generate_signal(data_with_indicators)
        
        if signal == Signal.HOLD:
            return None
        
        # Get current price (last close)
        current_price = float(data['close'].iloc[-1])
        
        # Calculate position size
        # Note: portfolio_value should be provided by the trading engine
        portfolio_value = self.get_portfolio_value()
        position_size = self.calculate_position_size(
            signal, current_price, portfolio_value, self._position
        )
        
        if position_size <= 0:
            return None
        
        return {
            'signal': signal,
            'price': current_price,
            'amount': position_size,
            'timestamp': data.index[-1],
            'reason': self.get_signal_reason()
        }
    
    def on_trade(self, trade: Trade):
        """Called when a trade is executed.
        
        Args:
            trade: Executed trade details
        """
        self._trades.append(trade)
        
        # Update position
        if trade.side == 'buy':
            if self._position:
                # Add to existing position
                total_amount = self._position.amount + trade.amount
                avg_price = (
                    (self._position.amount * self._position.avg_entry_price + 
                     trade.amount * trade.price) / total_amount
                )
                self._position.amount = total_amount
                self._position.avg_entry_price = avg_price
            else:
                # New position
                self._position = Position(
                    symbol=trade.symbol,
                    amount=trade.amount,
                    avg_entry_price=trade.price,
                    current_price=trade.price,
                    timestamp=trade.timestamp
                )
        else:  # sell
            if self._position:
                self._position.amount -= trade.amount
                if self._position.amount <= 0:
                    self._position = None
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value.
        
        This should be overridden by the trading engine.
        """
        return 10000.0  # Default value
    
    def get_signal_reason(self) -> str:
        """Get explanation for the last signal.
        
        Override this to provide strategy-specific reasoning.
        """
        return f"{self.name} signal generated"
    
    def get_position(self) -> Optional[Position]:
        """Get current position."""
        return self._position
    
    def get_trades(self) -> List[Trade]:
        """Get list of all trades."""
        return self._trades.copy()
    
    def reset(self):
        """Reset strategy state."""
        self._position = None
        self._trades = []
        self._is_initialized = False