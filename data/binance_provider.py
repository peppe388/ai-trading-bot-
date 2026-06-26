import requests
import pandas as pd

BINANCE_BASE = "https://api.binance.com"

SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT",
}

def to_binance(symbol):
    return SYMBOL_MAP.get(symbol)

def get_price(symbol):
    b_sym = to_binance(symbol)
    if not b_sym:
        return None
    try:
        r = requests.get(f"{BINANCE_BASE}/api/v3/ticker/price",
                         params={"symbol": b_sym}, timeout=10)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None

def get_bars(symbol, interval="1m", limit=30):
    b_sym = to_binance(symbol)
    if not b_sym:
        return None
    try:
        r = requests.get(f"{BINANCE_BASE}/api/v3/klines",
                         params={"symbol": b_sym, "interval": interval, "limit": limit},
                         timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = []
        times = []
        for k in data:
            rows.append({
                "Open": float(k[1]), "High": float(k[2]),
                "Low": float(k[3]), "Close": float(k[4]),
                "Volume": float(k[5]),
            })
            times.append(pd.Timestamp(int(k[0]), unit='ms', tz='UTC'))
        df = pd.DataFrame(rows, index=times)
        return df
    except Exception:
        return None
