import time, threading
from datetime import datetime
import config


class PriceStreamer:
    def __init__(self):
        self.prices = {}
        self.running = False
        self._thread = None
        self._listeners = []

    def start(self, interval=30):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, args=(interval,), daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def on_price(self, callback):
        self._listeners.append(callback)

    def _loop(self, interval):
        import pandas as pd
        import yfinance as yf

        all_syms = (
            list(config.FOREX_PAIRS.values())
            + list(config.STOCKS.values())
            + list(config.COMMODITIES.values())
            + list(config.CRYPTO.values())
        )
        while self.running:
            for sym in all_syms:
                if not self.running:
                    return
                try:
                    data = yf.download(sym, period="5d", progress=False)
                    if data.empty:
                        continue
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [c[0] for c in data.columns]
                    price = float(data["Close"].iloc[-1])
                    prev = float(data["Close"].iloc[-2]) if len(data) >= 2 else price
                    change = (price / prev - 1) * 100
                    self.prices[sym] = {
                        "price": round(price, 4),
                        "change": round(change, 2),
                        "time": datetime.now(),
                    }
                    for cb in self._listeners:
                        try:
                            cb(sym, self.prices[sym])
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(max(1, interval // max(len(all_syms), 1)))
