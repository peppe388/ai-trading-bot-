import os, json
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTFOLIO_FILE = os.path.join(BASE, "portfolio.json")


class Portfolio:
    def __init__(self):
        self.positions = {}
        self.trades = []
        self.initial_capital = 10000
        self.load()

    def load(self):
        if os.path.exists(PORTFOLIO_FILE):
            try:
                with open(PORTFOLIO_FILE) as f:
                    d = json.load(f)
                    self.positions = d.get("positions", {})
                    self.trades = d.get("trades", [])
                    self.initial_capital = d.get("initial_capital", 10000)
            except Exception:
                pass

    def save(self):
        os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump({
                "positions": self.positions,
                "trades": self.trades,
                "initial_capital": self.initial_capital,
            }, f, indent=2)

    def buy(self, symbol, price, qty):
        pos = self.positions.setdefault(symbol, {"qty": 0, "avg_price": 0, "total_invested": 0})
        pos["qty"] += qty
        pos["total_invested"] += price * qty
        pos["avg_price"] = pos["total_invested"] / pos["qty"]
        self.trades.append({
            "date": datetime.now().isoformat(), "symbol": symbol,
            "type": "BUY", "price": round(price, 4), "qty": qty,
        })
        self.save()

    def sell(self, symbol, price, qty=None):
        if symbol not in self.positions:
            return None
        pos = self.positions[symbol]
        qty = min(qty, pos["qty"]) if qty else pos["qty"]
        pnl = (price - pos["avg_price"]) * qty
        pos["qty"] -= qty
        pos["total_invested"] -= pos["avg_price"] * qty
        if pos["qty"] <= 0:
            del self.positions[symbol]
        self.trades.append({
            "date": datetime.now().isoformat(), "symbol": symbol,
            "type": "SELL", "price": round(price, 4), "qty": qty,
            "pnl": round(pnl, 2),
        })
        self.save()
        return round(pnl, 2)

    def pnl(self, symbol, current_price):
        pos = self.positions.get(symbol)
        if not pos or pos["qty"] <= 0:
            return 0
        return round((current_price - pos["avg_price"]) * pos["qty"], 2)

    def total_value(self, prices):
        return sum(pos["qty"] * prices.get(sym, 0) for sym, pos in self.positions.items())

    def summary(self, prices):
        invested = sum(p.get("total_invested", 0) for p in self.positions.values())
        cur = self.total_value(prices)
        closed_pnl = sum(t.get("pnl", 0) for t in self.trades if t["type"] == "SELL")
        total = cur + closed_pnl
        ret = round((total - self.initial_capital) / max(self.initial_capital, 1) * 100, 2)
        return {
            "invested": round(invested, 2), "current_value": round(cur, 2),
            "closed_pnl": round(closed_pnl, 2), "total_return": ret,
            "num_positions": len(self.positions),
            "num_trades": len([t for t in self.trades if t["type"] == "SELL"]),
        }

    def reset(self, capital=10000):
        self.positions = {}
        self.trades = []
        self.initial_capital = capital
        self.save()

    def auto_trade(self, symbol, signal, confidence, price):
        if signal == "BUY" and confidence >= 50:
            if symbol not in self.positions:
                qty = round(100 / price, 4)
                self.buy(symbol, price, max(qty, 0.001))
                return f"Comprato {symbol} x {qty} @ ${price}"
        elif signal == "SELL" and confidence >= 50:
            if symbol in self.positions:
                pnl = self.sell(symbol, price)
                if pnl is not None:
                    return f"Venduto {symbol} @ ${price} P&L: ${pnl}"
        return None
