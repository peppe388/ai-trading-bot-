import pandas as pd
import numpy as np
from data.market import fetch_data
from data.indicators import add_indicators, get_latest_indicators
from advisor.analyser import _ta_ensemble, _volume_score, _multi_tf_score


def run_backtest(symbol, years=2):
    days = int(years * 365)
    df = fetch_data(symbol, days)
    df = add_indicators(df)
    df = df.dropna().copy()

    capital = 10000
    position = 0
    trades = []
    equity = []

    for i in range(60, len(df)):
        window = df.iloc[:i + 1]
        latest = window.iloc[-1]
        ind = get_latest_indicators(window)
        s_ta = _ta_ensemble(ind)
        s_vol = _volume_score(window)
        s_tf = _multi_tf_score(window)
        score = s_ta * 0.35 + s_vol * 0.15 + s_tf * 0.15
        norm = score / 0.65 if 0.65 > 0 else 0

        signal = "HOLD"
        if norm > 0.2:
            signal = "BUY"
        elif norm < -0.2:
            signal = "SELL"

        price = float(latest["Close"])
        if signal == "BUY" and position == 0:
            position = capital / price
            capital = 0
            trades.append({"date": latest.name, "type": "BUY", "price": price, "qty": position})
        elif signal == "SELL" and position > 0:
            proceeds = position * price
            buy_cost = sum(t["qty"] * t["price"] for t in trades if t["type"] == "BUY")
            pnl = proceeds - buy_cost
            trades.append({"date": latest.name, "type": "SELL", "price": price, "qty": position, "pnl": round(pnl, 2)})
            capital = proceeds
            position = 0

        equity.append({"date": latest.name, "value": capital + position * price})

    if position > 0:
        capital = position * float(df["Close"].iloc[-1])
        position = 0

    eq_df = pd.DataFrame(equity).set_index("date")
    total_return = (capital / 10000 - 1) * 100
    buys = [t for t in trades if t["type"] == "BUY"]
    sells = [t for t in trades if t["type"] == "SELL"]
    winning = sum(1 for t in sells if t.get("pnl", 0) > 0)
    win_rate = round(winning / max(len(sells), 1) * 100, 1)

    eq_df["peak"] = eq_df["value"].cummax()
    eq_df["dd"] = (eq_df["value"] - eq_df["peak"]) / eq_df["peak"] * 100
    max_dd = round(eq_df["dd"].min(), 2)
    eq_df["ret"] = eq_df["value"].pct_change()
    sharpe = round(np.sqrt(252) * eq_df["ret"].mean() / max(eq_df["ret"].std(), 1e-10), 2)

    return {
        "symbol": symbol, "total_return": round(total_return, 2),
        "win_rate": win_rate, "max_drawdown": max_dd,
        "sharpe_ratio": sharpe, "num_trades": len(sells),
        "final_capital": round(capital, 2), "equity_curve": eq_df,
    }


def format_backtest(result):
    lines = [
        f"Simbolo: {result['symbol']}",
        f"Rendimento: {result['total_return']:+.2f}%",
        f"Vittorie: {result['win_rate']}% ({result['num_trades']} operazioni)",
        f"Max Drawdown: {result['max_drawdown']:.2f}%",
        f"Sharpe Ratio: {result['sharpe_ratio']}",
        f"Capitale finale: ${result['final_capital']:.2f}",
    ]
    return "\n".join(lines)
