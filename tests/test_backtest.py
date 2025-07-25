"""Tests for backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ultsignaltrader.backtests.engine import BacktestEngine, BacktestResult
from src.ultsignaltrader.strategies.ema_crossover import EMACrossoverStrategy


class TestBacktestEngine:
    """Test the backtesting engine."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data with clear trend changes."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='1H')
        
        # Create data with uptrend, then downtrend
        prices = []
        
        # Uptrend for first 100 periods
        uptrend = np.linspace(50000, 55000, 100)
        prices.extend(uptrend + np.random.normal(0, 50, 100))
        
        # Downtrend for next 100 periods
        downtrend = np.linspace(55000, 50000, 100)
        prices.extend(downtrend + np.random.normal(0, 50, 100))
        
        data = pd.DataFrame({
            'open': np.array(prices) * 0.999,
            'high': np.array(prices) * 1.001,
            'low': np.array(prices) * 0.998,
            'close': prices,
            'volume': np.random.uniform(100, 200, 200)
        }, index=dates)
        
        return data
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = BacktestEngine(initial_capital=10000, commission=0.001)
        
        assert engine.initial_capital == 10000
        assert engine.commission == 0.001
        assert engine.cash == 10000
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0
    
    def test_reset(self):
        """Test engine reset functionality."""
        engine = BacktestEngine()
        
        # Modify state
        engine.cash = 5000
        engine.positions['BTC/USDT'] = 'dummy_position'
        engine.trades.append('dummy_trade')
        
        # Reset
        engine.reset()
        
        assert engine.cash == engine.initial_capital
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0
    
    def test_basic_backtest(self, sample_data):
        """Test basic backtest execution."""
        engine = BacktestEngine(initial_capital=10000, commission=0.001)
        strategy = EMACrossoverStrategy(fast_period=5, slow_period=10)
        
        # Run backtest
        result = engine.run(strategy, sample_data, warmup_period=20)
        
        # Check result structure
        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 10000
        assert result.final_capital > 0
        assert len(result.equity_curve) == len(sample_data) - 20
        assert result.total_trades >= 0
    
    def test_trade_execution(self, sample_data):
        """Test trade execution mechanics."""
        engine = BacktestEngine(initial_capital=10000, commission=0.001)
        strategy = EMACrossoverStrategy(
            fast_period=5, 
            slow_period=10,
            position_size_pct=0.5  # Use 50% of capital
        )
        
        result = engine.run(strategy, sample_data)
        
        # Should have executed some trades
        assert len(engine.trades) > 0
        
        # Check trade properties
        for trade in engine.trades:
            assert trade.symbol == 'BTC/USDT'
            assert trade.side in ['buy', 'sell']
            assert trade.amount > 0
            assert trade.price > 0
            assert trade.commission > 0
    
    def test_commission_calculation(self):
        """Test that commissions are properly calculated."""
        engine = BacktestEngine(initial_capital=10000, commission=0.01)  # 1%
        
        # Mock a buy trade
        from src.ultsignaltrader.strategies.base import Signal
        signal_info = {
            'signal': Signal.BUY,
            'amount': 0.1,
            'price': 50000,
            'timestamp': datetime.now(),
            'reason': 'test'
        }
        
        initial_cash = engine.cash
        engine._execute_trade(
            strategy=EMACrossoverStrategy(),
            signal_info=signal_info,
            current_price=50000,
            symbol='BTC/USDT',
            timestamp=datetime.now()
        )
        
        # Check commission was deducted
        expected_cost = 0.1 * 50000  # 5000
        expected_commission = expected_cost * 0.01  # 50
        expected_cash = initial_cash - expected_cost - expected_commission
        
        assert abs(engine.cash - expected_cash) < 0.01
    
    def test_insufficient_funds(self):
        """Test behavior when insufficient funds for trade."""
        engine = BacktestEngine(initial_capital=1000, commission=0.001)
        
        # Try to buy more than we can afford
        from src.ultsignaltrader.strategies.base import Signal
        signal_info = {
            'signal': Signal.BUY,
            'amount': 1.0,  # 1 BTC at 50k = 50k (we only have 1k)
            'price': 50000,
            'timestamp': datetime.now(),
            'reason': 'test'
        }
        
        engine._execute_trade(
            strategy=EMACrossoverStrategy(),
            signal_info=signal_info,
            current_price=50000,
            symbol='BTC/USDT',
            timestamp=datetime.now()
        )
        
        # Should have adjusted the amount
        assert len(engine.trades) == 1
        assert engine.trades[0].amount < 1.0
        assert engine.cash < 100  # Most cash should be used
    
    def test_performance_metrics(self, sample_data):
        """Test performance metric calculations."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = EMACrossoverStrategy(fast_period=5, slow_period=10)
        
        result = engine.run(strategy, sample_data)
        
        # Check metrics are calculated
        assert result.sharpe_ratio is not None
        assert result.max_drawdown <= 0
        assert result.max_drawdown_pct <= 0
        assert 0 <= result.win_rate <= 1
        assert result.profit_factor >= 0
        
        # Check equity curve properties
        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] > 0
        
        # Check daily returns
        assert len(result.daily_returns) > 0
    
    def test_position_tracking(self, sample_data):
        """Test that positions are properly tracked."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = EMACrossoverStrategy()
        
        # Run backtest
        result = engine.run(strategy, sample_data[:50])  # Use subset
        
        # After backtest, check final position state
        final_portfolio_value = result.final_capital
        
        # Portfolio value should include cash + position values
        assert final_portfolio_value > 0
        
        # If we have open positions, they should be valued correctly
        if engine.positions:
            position_value = sum(
                pos.amount * sample_data.iloc[-1]['close'] 
                for pos in engine.positions.values()
            )
            assert final_portfolio_value == engine.cash + position_value