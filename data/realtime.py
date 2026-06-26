import data.binance_provider as binance

CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"}

def get_price(symbol):
    """Real-time price: Binance per crypto, None per altri (caller usa yfinance)."""
    if symbol in CRYPTO_SYMBOLS:
        p = binance.get_price(symbol)
        if p is not None:
            return p
    return None

def get_bars(symbol, interval="1m", limit=30):
    """Real-time candele: Binance per crypto, None per altri."""
    if symbol in CRYPTO_SYMBOLS:
        df = binance.get_bars(symbol, interval=interval, limit=limit)
        if df is not None:
            return df
    return None
