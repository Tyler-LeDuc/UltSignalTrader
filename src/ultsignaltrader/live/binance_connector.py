"""Binance exchange connector for live trading.

Handles order execution, balance queries, and market data fetching
using the ccxt library.
"""

import ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

from ..strategies.base import Trade
from ..utils.logger import get_logger
from ..config import BinanceConfig


logger = get_logger(__name__)


class BinanceConnector:
    """Binance exchange connector using ccxt."""
    
    def __init__(self, config: BinanceConfig):
        """Initialize Binance connector.
        
        Args:
            config: Binance configuration object
        """
        self.config = config
        
        # Initialize ccxt exchange
        exchange_params = {
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # spot trading
                'adjustForTimeDifference': True
            }
        }
        
        if config.testnet:
            # Use Binance testnet
            exchange_params['urls'] = {
                'api': {
                    'public': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api',
                }
            }
        
        self.exchange = ccxt.binance(exchange_params)
        
        # Cache for market data
        self._markets = None
        self._last_market_update = None
        
        logger.info(f"Initialized Binance connector (testnet: {config.testnet})")
    
    def test_connection(self) -> bool:
        """Test connection to Binance.
        
        Returns:
            True if connection successful
        """
        try:
            self.exchange.fetch_time()
            logger.info("Successfully connected to Binance")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            return False
    
    def get_markets(self, force_refresh: bool = False) -> Dict:
        """Get available markets.
        
        Args:
            force_refresh: Force refresh of market data
            
        Returns:
            Dictionary of markets
        """
        # Cache markets for 1 hour
        if (not force_refresh and self._markets and self._last_market_update and
            datetime.now() - self._last_market_update < timedelta(hours=1)):
            return self._markets
        
        try:
            self._markets = self.exchange.load_markets()
            self._last_market_update = datetime.now()
            logger.info(f"Loaded {len(self._markets)} markets")
            return self._markets
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance.
        
        Returns:
            Dictionary of asset balances
        """
        try:
            balance = self.exchange.fetch_balance()
            
            # Filter out zero balances
            non_zero_balance = {
                asset: {
                    'free': balance[asset]['free'],
                    'used': balance[asset]['used'],
                    'total': balance[asset]['total']
                }
                for asset in balance['total']
                if balance['total'][asset] > 0
            }
            
            logger.info(f"Fetched balance for {len(non_zero_balance)} assets")
            return non_zero_balance
            
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Ticker data
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', 
                  limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise
    
    def place_market_order(self, symbol: str, side: str, amount: float) -> Trade:
        """Place a market order.
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            amount: Amount to trade (in base currency)
            
        Returns:
            Executed trade details
        """
        try:
            # Validate inputs
            if side not in ['buy', 'sell']:
                raise ValueError(f"Invalid side: {side}")
            
            # Get market info for precision
            markets = self.get_markets()
            if symbol not in markets:
                raise ValueError(f"Unknown symbol: {symbol}")
            
            market = markets[symbol]
            
            # Round amount to market precision
            amount = self.exchange.amount_to_precision(symbol, amount)
            
            logger.info(f"Placing {side} market order: {amount} {symbol}")
            
            # Place order
            order = self.exchange.create_market_order(symbol, side, amount)
            
            # Wait for order to be filled
            order_id = order['id']
            filled_order = self._wait_for_order(symbol, order_id)
            
            # Create Trade object
            trade = Trade(
                symbol=symbol,
                side=side,
                amount=filled_order['filled'],
                price=filled_order['average'],
                timestamp=datetime.fromtimestamp(filled_order['timestamp'] / 1000),
                order_id=order_id,
                commission=self._calculate_commission(filled_order)
            )
            
            logger.info(
                f"Order executed: {side} {trade.amount} {symbol} "
                f"@ {trade.price} (ID: {order_id})"
            )
            
            return trade
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    def place_limit_order(self, symbol: str, side: str, amount: float, 
                         price: float) -> str:
        """Place a limit order.
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            amount: Amount to trade
            price: Limit price
            
        Returns:
            Order ID
        """
        try:
            # Round to market precision
            amount = self.exchange.amount_to_precision(symbol, amount)
            price = self.exchange.price_to_precision(symbol, price)
            
            logger.info(
                f"Placing {side} limit order: {amount} {symbol} @ {price}"
            )
            
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            
            logger.info(f"Limit order placed: {order['id']}")
            return order['id']
            
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol
            
        Returns:
            True if cancelled successfully
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get list of open orders.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of open orders
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            logger.info(f"Found {len(orders)} open orders")
            return orders
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            raise
    
    def get_order_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get order history.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of orders to fetch
            
        Returns:
            List of historical orders
        """
        try:
            orders = self.exchange.fetch_closed_orders(symbol, limit=limit)
            return orders
        except Exception as e:
            logger.error(f"Failed to fetch order history: {e}")
            raise
    
    def _wait_for_order(self, symbol: str, order_id: str, 
                       timeout: int = 30) -> Dict:
        """Wait for order to be filled.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID
            timeout: Timeout in seconds
            
        Returns:
            Filled order details
        """
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                order = self.exchange.fetch_order(order_id, symbol)
                if order['status'] in ['closed', 'filled']:
                    return order
                elif order['status'] in ['canceled', 'rejected', 'expired']:
                    raise Exception(f"Order {order_id} {order['status']}")
                
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error checking order status: {e}")
                raise
        
        raise TimeoutError(f"Order {order_id} not filled within {timeout}s")
    
    def _calculate_commission(self, order: Dict) -> float:
        """Calculate commission from order.
        
        Args:
            order: Order details
            
        Returns:
            Commission amount
        """
        if 'fee' in order and order['fee']:
            return order['fee']['cost']
        
        # Default Binance fee is 0.1%
        return order['filled'] * order['average'] * 0.001
    
    def get_portfolio_value(self, quote_currency: str = 'USDT') -> float:
        """Calculate total portfolio value in quote currency.
        
        Args:
            quote_currency: Currency to calculate value in
            
        Returns:
            Total portfolio value
        """
        try:
            balance = self.get_balance()
            total_value = 0.0
            
            for asset, amounts in balance.items():
                if asset == quote_currency:
                    total_value += amounts['total']
                else:
                    # Convert to quote currency
                    symbol = f"{asset}/{quote_currency}"
                    try:
                        ticker = self.get_ticker(symbol)
                        if ticker and 'last' in ticker:
                            value = amounts['total'] * ticker['last']
                            total_value += value
                    except:
                        # Skip if can't get price
                        logger.warning(f"Could not get price for {symbol}")
            
            logger.info(f"Portfolio value: {total_value:.2f} {quote_currency}")
            return total_value
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio value: {e}")
            raise