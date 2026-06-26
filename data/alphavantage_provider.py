import os, logging
import requests
import pandas as pd

API_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
BASE = "https://www.alphavantage.co/query"
_log = logging.getLogger(__name__)


def _parse_yahoo_forex(symbol):
    """EURUSD=X → (EUR, USD). Restituisce None se non è forex."""
    if not symbol.endswith("=X"):
        return None
    pair = symbol.replace("=X", "")
    if len(pair) != 6:
        return None
    return pair[:3], pair[3:]


def get_price(symbol):
    """Prezzo real-time forex via Alpha Vantage."""
    pair = _parse_yahoo_forex(symbol)
    if not pair:
        return None
    if not API_KEY:
        return None
    from_c, to_c = pair
    try:
        r = requests.get(BASE, params={
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_c,
            "to_currency": to_c,
            "apikey": API_KEY,
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
        rate = data.get("Realtime Currency Exchange Rate", {})
        price = rate.get("5. Exchange Rate")
        if price:
            return float(price)
        return None
    except Exception as e:
        _log.warning("Alpha Vantage prezzo fallito per %s: %s", symbol, e)
        return None


def get_bars(symbol, interval="1min", limit=30):
    """Candele intraday forex via Alpha Vantage."""
    pair = _parse_yahoo_forex(symbol)
    if not pair:
        return None
    if not API_KEY:
        return None
    from_c, to_c = pair
    av_interval = interval.replace("m", "min").replace("h", "min")
    if av_interval not in ("1min", "5min", "15min", "30min", "60min"):
        av_interval = "1min"
    try:
        r = requests.get(BASE, params={
            "function": "FX_INTRADAY",
            "from_symbol": from_c,
            "to_symbol": to_c,
            "interval": av_interval,
            "outputsize": "compact",
            "datatype": "json",
            "apikey": API_KEY,
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
        series = data.get("Time Series FX ({}".format(av_interval) + ")", {})
        if not series:
            return None
        rows = []
        times = []
        for ts in sorted(series.keys(), reverse=True)[:limit]:
            ohlc = series[ts]
            rows.append({
                "Open": float(ohlc["1. open"]),
                "High": float(ohlc["2. high"]),
                "Low": float(ohlc["3. low"]),
                "Close": float(ohlc["4. close"]),
            })
            times.append(pd.Timestamp(ts))
        df = pd.DataFrame(rows, index=times)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.iloc[::-1]
        return df
    except Exception as e:
        _log.warning("Alpha Vantage candele fallite per %s: %s", symbol, e)
        return None
