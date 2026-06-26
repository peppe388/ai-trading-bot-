import data.binance_provider as binance
import data.alphavantage_provider as alphav

CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"}


def _is_forex(symbol):
    return symbol.endswith("=X")


def get_price(symbol):
    if symbol in CRYPTO_SYMBOLS:
        p = binance.get_price(symbol)
        if p is not None:
            return p
    if _is_forex(symbol):
        p = alphav.get_price(symbol)
        if p is not None:
            return p
    return None


def get_bars(symbol, interval="1m", limit=30):
    av_interval = interval
    if symbol in CRYPTO_SYMBOLS:
        df = binance.get_bars(symbol, interval=interval, limit=limit)
        if df is not None:
            return df
    if _is_forex(symbol):
        df = alphav.get_bars(symbol, interval=av_interval, limit=limit)
        if df is not None:
            return df
    return None
