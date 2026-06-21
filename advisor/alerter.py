import time, threading
from datetime import datetime
import config


class Alerter:
    def __init__(self):
        self.previous = {}
        self.alerts = []
        self.running = False
        self._thread = None

    def start(self, interval=300):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, args=(interval,), daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self, interval):
        while self.running:
            try:
                self._check()
            except Exception:
                pass
            for _ in range(interval):
                if not self.running:
                    return
                time.sleep(1)

    def _check(self):
        from data.market import fetch_data
        from data.indicators import add_indicators, get_latest_indicators

        items = (
            list(config.FOREX_PAIRS.items())
            + list(config.STOCKS.items())
            + list(config.COMMODITIES.items())
            + list(config.CRYPTO.items())
        )
        for label, sym in items:
            try:
                df = fetch_data(sym, 90)
                df = add_indicators(df)
                ind = get_latest_indicators(df)

                # Build signal from simple rules
                signal = "HOLD"
                conf = 15
                if ind["rsi"] < 35:
                    signal = "BUY"
                    conf = 60
                elif ind["rsi"] > 65:
                    signal = "SELL"
                    conf = 60

                prev_sig = self.previous.get(sym)
                if prev_sig and prev_sig["signal"] != signal:
                    self.alerts.insert(0, {
                        "time": datetime.now(), "symbol": sym,
                        "label": label, "from": prev_sig["signal"],
                        "to": signal, "price": ind["price"],
                    })
                self.previous[sym] = {"signal": signal, "price": ind["price"]}
            except Exception:
                pass
