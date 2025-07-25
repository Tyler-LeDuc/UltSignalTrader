"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ultsignaltrader.strategies.base import BaseStrategy, Signal, Trade, Position
from src.ultsignaltrader.strategies.ema_crossover import EMACrossoverStrategy


class TestBaseStrategy:
    """Test BaseStrategy abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStrategy("test")
    
    def test_position_tracking(self):
        """Test position tracking functionality."""
        # Create a concrete implementation for testing
        strategy = EMACrossoverStrategy()
        
        # Initially no position
        assert strategy.get_position() is None
        
        # Execute a buy trade
        buy_trade = Trade(
            symbol="BTC/USDT",
            side="buy",
            amount=1.0,
            price=50000.0,
            timestamp=datetime.now()
        )
        strategy.on_trade(buy_trade)
        
        # Check position
        position = strategy.get_position()
        assert position is not None
        assert position.amount == 1.0
        assert position.avg_entry_price == 50000.0
        
        # Execute a partial sell
        sell_trade = Trade(
            symbol="BTC/USDT",
            side="sell",
            amount=0.5,
            price=55000.0,
            timestamp=datetime.now()
        )
        strategy.on_trade(sell_trade)
        
        # Check updated position
        position = strategy.get_position()
        assert position.amount == 0.5
        
        # Sell remaining
        sell_trade2 = Trade(
            symbol="BTC/USDT",
            side="sell",
            amount=0.5,
            price=55000.0,
            timestamp=datetime.now()
        )
        strategy.on_trade(sell_trade2)
        
        # Position should be closed
        assert strategy.get_position() is None


class TestEMACrossoverStrategy:
    """Test EMA Crossover Strategy."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        
        # Create trending data
        trend = np.linspace(50000, 55000, 100)
        noise = np.random.normal(0, 100, 100)
        prices = trend + noise
        
        data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.001,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(100, 200, 100)
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def downtrend_data(self):
        """Create downtrending OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        
        # Create downtrending data
        trend = np.linspace(55000, 50000, 100)
        noise = np.random.normal(0, 100, 100)
        prices = trend + noise
        
        data = pd.DataFrame({
            'open': prices * 1.001,
            'high': prices * 1.002,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.uniform(100, 200, 100)
        }, index=dates)
        
        return data
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = EMACrossoverStrategy(fast_period=10, slow_period=20)
        
        assert strategy.name == "EMA_Crossover"
        assert strategy.fast_period == 10
        assert strategy.slow_period == 20
        assert strategy.position_size_pct == 0.1
    
    def test_invalid_parameters(self):
        """Test invalid parameter validation."""
        with pytest.raises(ValueError):
            # Fast period >= slow period
            EMACrossoverStrategy(fast_period=20, slow_period=10)
    
    def test_calculate_indicators(self, sample_data):
        """Test indicator calculation."""
        strategy = EMACrossoverStrategy(fast_period=12, slow_period=26)
        
        # Calculate indicators
        data_with_indicators = strategy.calculate_indicators(sample_data)
        
        # Check that indicators are added
        assert 'ema_fast' in data_with_indicators.columns
        assert 'ema_slow' in data_with_indicators.columns
        assert 'ema_diff' in data_with_indicators.columns
        assert 'ema_diff_pct' in data_with_indicators.columns
        
        # Check that EMAs are calculated correctly
        assert not data_with_indicators['ema_fast'].isna().all()
        assert not data_with_indicators['ema_slow'].isna().all()
        
        # Fast EMA should be more responsive
        price_change = sample_data['close'].diff().sum()
        ema_fast_change = data_with_indicators['ema_fast'].diff().sum()
        ema_slow_change = data_with_indicators['ema_slow'].diff().sum()
        
        assert abs(ema_fast_change) > abs(ema_slow_change)
    
    def test_generate_buy_signal(self, sample_data):
        """Test buy signal generation on uptrend."""
        strategy = EMACrossoverStrategy(fast_period=5, slow_period=10)
        strategy.initialize(sample_data[:30])
        
        # Process data in chunks to find crossover
        for i in range(30, len(sample_data)):
            data_chunk = sample_data[:i+1]
            data_with_ind = strategy.calculate_indicators(data_chunk)
            signal = strategy.generate_signal(data_with_ind)
            
            if signal == Signal.BUY:
                # Verify it's a valid crossover
                assert data_with_ind['ema_fast'].iloc[-1] > data_with_ind['ema_slow'].iloc[-1]
                break
    
    def test_generate_sell_signal(self, downtrend_data):
        """Test sell signal generation on downtrend."""
        strategy = EMACrossoverStrategy(fast_period=5, slow_period=10)
        strategy.initialize(downtrend_data[:30])
        
        # Process data to find sell crossover
        for i in range(30, len(downtrend_data)):
            data_chunk = downtrend_data[:i+1]
            data_with_ind = strategy.calculate_indicators(data_chunk)
            signal = strategy.generate_signal(data_with_ind)
            
            if signal == Signal.SELL:
                # Verify it's a valid crossover
                assert data_with_ind['ema_fast'].iloc[-1] < data_with_ind['ema_slow'].iloc[-1]
                break
    
    def test_position_sizing(self):
        """Test position size calculation."""
        strategy = EMACrossoverStrategy(position_size_pct=0.1)
        
        portfolio_value = 10000.0
        current_price = 50000.0
        
        # Test buy position sizing
        size = strategy.calculate_position_size(
            Signal.BUY, current_price, portfolio_value, None
        )
        expected_size = (portfolio_value * 0.1) / current_price
        assert abs(size - expected_size) < 0.0001
        
        # Test that we don't buy if already in position
        position = Position(
            symbol="BTC/USDT",
            amount=0.2,
            avg_entry_price=50000.0,
            current_price=50000.0,
            timestamp=datetime.now()
        )
        size = strategy.calculate_position_size(
            Signal.BUY, current_price, portfolio_value, position
        )
        assert size == 0.0
        
        # Test sell sizing (should sell entire position)
        size = strategy.calculate_position_size(
            Signal.SELL, current_price, portfolio_value, position
        )
        assert size == position.amount
    
    def test_on_data_integration(self, sample_data):
        """Test the full on_data workflow."""
        strategy = EMACrossoverStrategy()
        
        # Must initialize first
        with pytest.raises(RuntimeError):
            strategy.on_data(sample_data)
        
        # Initialize and process
        strategy.initialize(sample_data[:30])
        
        # Override get_portfolio_value
        strategy.get_portfolio_value = lambda: 10000.0
        
        # Process remaining data
        result = strategy.on_data(sample_data[:50])
        
        # Result should be None or a dict with signal info
        if result is not None:
            assert 'signal' in result
            assert 'price' in result
            assert 'amount' in result
            assert 'timestamp' in result
            assert 'reason' in result