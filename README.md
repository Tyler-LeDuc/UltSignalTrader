# UltSignalTrader

A production-grade cryptocurrency trading bot framework that supports both live trading and backtesting from the same codebase.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/ultsignaltrader.git
cd ultsignaltrader
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your settings

# 3. Run backtest (no API keys needed)
python main.py --mode backtest

# 4. Run live trading (requires API setup)
python main.py --mode live
```

## Features

- **Modular Architecture**: Easily pluggable strategies with clean separation of concerns
- **Live Trading**: Integration with Binance via `ccxt` library
- **Backtesting**: Historical simulation with comprehensive performance metrics
- **Strategy Framework**: Abstract base class for creating custom strategies
- **Risk Management**: Built-in position sizing and risk controls
- **Logging**: Comprehensive logging to console and rotating log files
- **Configuration**: Environment-based configuration with `.env` files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ultsignaltrader.git
cd ultsignaltrader
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

## Setting Up Binance API

### For Testing (Recommended to start)

1. Go to [Binance Testnet](https://testnet.binance.vision/)
2. Create a testnet account
3. Generate API keys for testing
4. Set `BINANCE_TESTNET=true` in your `.env` file

### For Live Trading

⚠️ **WARNING**: Live trading involves real money. Test thoroughly on testnet first!

1. Create a [Binance account](https://www.binance.com/)
2. Complete identity verification (KYC)
3. Go to [API Management](https://www.binance.com/en/my/settings/api-management)
4. Create a new API key with the following permissions:
   - **Enable Reading** ✓
   - **Enable Spot Trading** ✓
   - **Restrict IP access** (recommended for security)
5. Copy your API Key and Secret Key to `.env`
6. Set `BINANCE_TESTNET=false` in your `.env` file

### Security Best Practices

- Never commit your `.env` file to version control
- Use IP whitelist restrictions on your API keys
- Start with small amounts when live trading
- Enable 2FA on your Binance account
- Consider using a dedicated sub-account for trading

## Usage

### Backtesting Mode

Run a backtest with sample data:
```bash
python main.py --mode backtest
```

Run a backtest with your own data:
```bash
python main.py --mode backtest --data path/to/your/data.csv
```

### Live Trading Mode

**Warning**: Live trading involves real money. Test thoroughly with small amounts first.

```bash
python main.py --mode live
```

### Command Line Options

- `--mode`: Trading mode (`live` or `backtest`)
- `--strategy`: Strategy to use (default: `ema_crossover`)
- `--data`: CSV file path for backtesting
- `--config`: Custom .env file path

## Project Structure

```
ultsignaltrader/
├── src/
│   └── ultsignaltrader/
│       ├── strategies/      # Trading strategies
│       │   ├── base.py     # Base strategy interface
│       │   └── ema_crossover.py
│       ├── live/           # Live trading components
│       │   └── binance_connector.py
│       ├── backtests/      # Backtesting engine
│       │   ├── engine.py
│       │   └── data_loader.py
│       ├── utils/          # Utilities
│       │   └── logger.py
│       └── config.py       # Configuration management
├── data/                   # Historical data storage
├── logs/                   # Log files
├── notebooks/              # Jupyter notebooks for prototyping
├── tests/                  # Unit tests
├── main.py                 # Entry point
├── requirements.txt
├── setup.py
└── .env.example
```

## Creating Custom Strategies

To create a custom strategy, inherit from `BaseStrategy`:

```python
from src.ultsignaltrader.strategies.base import BaseStrategy, Signal

class MyStrategy(BaseStrategy):
    def calculate_indicators(self, data):
        # Add your indicators
        pass
    
    def generate_signal(self, data):
        # Return BUY, SELL, or HOLD
        pass
    
    def calculate_position_size(self, signal, current_price, 
                               portfolio_value, current_position):
        # Calculate how much to trade
        pass
```

## Configuration

Key configuration options in `.env`:

- `BINANCE_API_KEY`: Your Binance API key
- `BINANCE_API_SECRET`: Your Binance API secret
- `TRADING_MODE`: `live` or `backtest`
- `DEFAULT_SYMBOL`: Trading pair (e.g., `BTC/USDT`)
- `DEFAULT_TIMEFRAME`: Candle timeframe (e.g., `1h`)
- `MAX_POSITION_SIZE`: Maximum position as percentage of portfolio

## Safety Features

- Position size limits
- Stop loss and take profit configuration
- Comprehensive error handling
- Detailed logging for audit trail
- Testnet support for Binance

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/ tests/
```

Type checking:
```bash
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading carries substantial risk of loss. The authors assume no responsibility for trading losses incurred using this software.
