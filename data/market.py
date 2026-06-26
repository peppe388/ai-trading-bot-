import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
import config
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import data.realtime as realtime

_fetch_executor = ThreadPoolExecutor(max_workers=4)
_cache = {}
_cache_lock = threading.Lock()
CACHE_DURATION = 300


def fetch_data(symbol, days=730):
    key = f"{symbol}_{days}"
    now = datetime.now()
    with _cache_lock:
        if key in _cache:
            entry = _cache[key]
            if (now - entry["ts"]).total_seconds() < CACHE_DURATION:
                return entry["data"].copy()

    end = now
    start = end - timedelta(days=days)
    try:
        future = _fetch_executor.submit(yf.download, symbol, start=start, end=end, progress=False)
        df = future.result(timeout=30)
    except TimeoutError:
        logging.warning(f"yfinance timeout per {symbol}, riprovo...")
        try:
            future = _fetch_executor.submit(yf.download, symbol, start=start, end=end)
            df = future.result(timeout=30)
        except TimeoutError:
            raise TimeoutError(f"yfinance non risponde per {symbol}")
    except TypeError:
        df = yf.download(symbol, start=start, end=end)
    if df.empty:
        raise ValueError(f"Nessun dato trovato per {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    outlier_mask = df['Close'].pct_change().abs() > 0.15
    if outlier_mask.any():
        df = df[~outlier_mask].copy()

    with _cache_lock:
        _cache[key] = {"data": df.copy(), "ts": now}
    return df


def get_current_price(symbol):
    rt = realtime.get_price(symbol)
    if rt is not None:
        return rt
    try:
        future = _fetch_executor.submit(yf.download, symbol, period="5d", progress=False)
        data = future.result(timeout=20)
    except TimeoutError:
        return None
    except TypeError:
        data = yf.download(symbol, period="5d")
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    return float(data["Close"].iloc[-1])


def get_news(symbol, max_items=5):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        items = []
        for article in news[:max_items]:
            c = article.get("content", {})
            provider = c.get("provider", {})
            items.append({
                "title": c.get("title", ""),
                "link": c.get("canonicalUrl", {}).get("url", "") if isinstance(c.get("canonicalUrl"), dict) else "",
                "publisher": provider.get("displayName", "") if isinstance(provider, dict) else "",
            })
        return items
    except Exception:
        return []


def get_news_text(symbol):
    news = get_news(symbol, 3)
    if not news:
        return ""
    lines = ["Notizie recenti:"]
    for n in news:
        lines.append(f"- {n['title']} ({n['publisher']})")
    return "\n".join(lines)


def clear_cache():
    with _cache_lock:
        _cache.clear()


def get_asset_name(symbol):
    for name_list in [config.FOREX_PAIRS, config.STOCKS, config.COMMODITIES, config.CRYPTO]:
        for name, sym in name_list.items():
            if sym == symbol:
                return name
    return symbol


def resolve_symbol(text):
    text = text.strip().lower()

    for cn, sym in config.COMMON_NAMES.items():
        if cn in text:
            for name_list in [config.FOREX_PAIRS, config.STOCKS, config.COMMODITIES, config.CRYPTO]:
                for name, s in name_list.items():
                    if s == sym:
                        return name, sym
            return cn.upper(), sym

    for name_list in [config.FOREX_PAIRS, config.STOCKS, config.COMMODITIES, config.CRYPTO]:
        for name, sym in name_list.items():
            nl = name.lower()
            sl = sym.lower()
            if text in nl or text in sl or nl in text or sl in text:
                return name, sym

    maybe = text.upper().replace(" ", "")
    if "/" in maybe:
        parts = maybe.split("/")
        if len(parts) == 2 and len(parts[0]) <= 4 and len(parts[1]) <= 4:
            symbol = f"{parts[0]}{parts[1]}=X"
            return maybe, symbol

    return text.upper(), text.upper()
