FOREX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/JPY": "EURJPY=X",
    "EUR/GBP": "EURGBP=X",
}

STOCKS = {
    "S&P 500 (SPY)": "SPY",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Google (GOOGL)": "GOOGL",
}

COMMODITIES = {
    "Oro (GLD)": "GLD",
    "Petrolio (USO)": "USO",
}

CRYPTO = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
}

COMMON_NAMES = {
    "oro": "GLD",
    "gold": "GLD",
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "petrolio": "USO",
    "oil": "USO",
    "sp500": "SPY",
    "s&p": "SPY",
}

LOOKBACK_DAYS = 730
SEQUENCE_LENGTH = 60
EPOCHS = 10
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
DEFAULT_TRAIN_SYMBOL = "EURUSD=X"
