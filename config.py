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
    "Coca-Cola (KO)": "KO",
    "McDonald's (MCD)": "MCD",
    "Disney (DIS)": "DIS",
    "PepsiCo (PEP)": "PEP",
    "Nike (NKE)": "NKE",
    "Starbucks (SBUX)": "SBUX",
    "Visa (V)": "V",
    "Netflix (NFLX)": "NFLX",
    "Walmart (WMT)": "WMT",
    "Boeing (BA)": "BA",
}

COMMODITIES = {
    "Oro (GLD)": "GLD",
    "Argento (SLV)": "SLV",
    "Petrolio (USO)": "USO",
    "Gas Naturale (UNG)": "UNG",
    "Rame (CPER)": "CPER",
    "Grano (WEAT)": "WEAT",
}

CRYPTO = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "XRP (XRP-USD)": "XRP-USD",
}

COMMON_NAMES = {
    "oro": "GLD",
    "gold": "GLD",
    "argento": "SLV",
    "silver": "SLV",
    "petrolio": "USO",
    "oil": "USO",
    "gas": "UNG",
    "rame": "CPER",
    "grano": "WEAT",
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "solana": "SOL-USD",
    "sol": "SOL-USD",
    "xrp": "XRP-USD",
    "sp500": "SPY",
    "s&p": "SPY",
    "coca": "KO",
    "coca-cola": "KO",
    "mcdonald": "MCD",
    "mcdonald's": "MCD",
    "disney": "DIS",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "nike": "NKE",
    "starbucks": "SBUX",
    "visa": "V",
    "netflix": "NFLX",
    "walmart": "WMT",
    "boeing": "BA",
}

LOOKBACK_DAYS = 730
SEQUENCE_LENGTH = 60
EPOCHS = 10
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
DEFAULT_TRAIN_SYMBOL = "SPY"
