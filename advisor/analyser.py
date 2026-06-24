import pandas as pd
import numpy as np
from datetime import datetime
from data.indicators import get_latest_indicators

_prediction_history = []


def _rsi_score(rsi):
    if rsi < 30:
        return 1.0
    if rsi < 40:
        return 0.5
    if rsi > 70:
        return -1.0
    if rsi > 60:
        return -0.5
    return 0.0


def _macd_score(macd_status):
    return 1.0 if macd_status == "bullish" else -1.0


def _stoch_score(stoch):
    if stoch < 20:
        return 1.0
    if stoch < 30:
        return 0.5
    if stoch > 80:
        return -1.0
    if stoch > 70:
        return -0.5
    return 0.0


def _ta_ensemble(indicators):
    s_rsi = _rsi_score(indicators["rsi"])
    s_macd = _macd_score(indicators["macd_status"])
    s_stoch = _stoch_score(indicators["stoch"])
    s_trend = 1.0 if indicators["trend"] == "rialzista" else -1.0
    s_obv = 1.0 if indicators["obv_trend"] == "positivo" else -1.0

    score = (
        s_rsi * 0.15
        + s_macd * 0.25
        + s_stoch * 0.15
        + s_trend * 0.30
        + s_obv * 0.15
    )
    return score


def _volume_score(df):
    vols = df["Volume"]
    if vols.max() == 0 or vols.isna().all():
        return 0.0
    last_20 = vols.tail(20)
    if len(last_20) < 5:
        return 0.0
    avg_vol = last_20.mean()
    cur_vol = vols.iloc[-1]
    if avg_vol == 0:
        return 0.0
    ratio = cur_vol / avg_vol
    if ratio > 1.5 and df["Close"].iloc[-1] > df["Close"].iloc[-2]:
        return 1.0
    if ratio > 1.5 and df["Close"].iloc[-1] < df["Close"].iloc[-2]:
        return -1.0
    if ratio > 1.2:
        return 0.5 if df["Close"].iloc[-1] > df["Close"].iloc[-2] else -0.5
    return 0.0


def _multi_tf_score(df):
    if len(df) < 60:
        return 0.0
    weekly = df.resample("W").agg({
        "Close": "last", "High": "max", "Low": "min", "Volume": "sum"
    }).dropna()

    daily_trend = 1.0 if df["Close"].iloc[-1] > df["Close"].iloc[-21] else -1.0
    weekly_sma = weekly["Close"].rolling(8).mean()
    weekly_trend = 1.0 if len(weekly_sma) >= 1 and weekly["Close"].iloc[-1] > weekly_sma.iloc[-1] else -1.0

    return (daily_trend * 0.6 + weekly_trend * 0.4)


def _lstm_score(pred_pct):
    return 1.0 if pred_pct > 0 else -1.0


def _lstm_weight(pred_pct):
    return min(abs(pred_pct) / 3.0, 1.0) * 0.35


def record_prediction(symbol, predicted_change, later_change, correct):
    _prediction_history.append({
        "symbol": symbol,
        "time": datetime.now(),
        "predicted": predicted_change,
        "actual": later_change,
        "correct": correct,
    })
    if len(_prediction_history) > 200:
        _prediction_history.pop(0)


def get_accuracy(symbol=None, n=30):
    hist = _prediction_history
    if symbol:
        hist = [p for p in hist if p["symbol"] == symbol]
    if len(hist) > n:
        hist = hist[-n:]
    if not hist:
        return None
    correct = sum(1 for p in hist if p["correct"])
    return round(correct / len(hist) * 100, 1)


def _expert_score(score):
    if score > 0.25: return 1    # BUY vote
    if score < -0.25: return -1  # SELL vote
    return 0                     # NEUTRAL

def analyse(df, lstm_prediction_pct):
    indicators = get_latest_indicators(df)

    has_lstm = lstm_prediction_pct != 0
    s_lstm = _lstm_score(lstm_prediction_pct) if has_lstm else 0
    w_lstm = _lstm_weight(lstm_prediction_pct) if has_lstm else 0
    s_ta = _ta_ensemble(indicators)
    s_vol = _volume_score(df)
    s_tf = _multi_tf_score(df)

    # Expert votes: each component votes BUY(+1) SELL(-1) NEUTRAL(0)
    votes = []
    if has_lstm and w_lstm > 0:
        votes.append(_expert_score(s_lstm))
    votes.append(_expert_score(s_ta))
    votes.append(_expert_score(s_vol))
    votes.append(_expert_score(s_tf))

    n_votes = len(votes)
    buys = sum(1 for v in votes if v == 1)
    sells = sum(1 for v in votes if v == -1)
    neutrals = n_votes - buys - sells

    # Confluence rules
    if buys >= 3 and sells == 0:
        signal = "BUY"
        pct = buys / n_votes
    elif sells >= 3 and buys == 0:
        signal = "SELL"
        pct = sells / n_votes
    elif buys >= 2 and sells == 0:
        signal = "BUY"
        pct = 0.5
    elif sells >= 2 and buys == 0:
        signal = "SELL"
        pct = 0.5
    else:
        signal = "HOLD"
        pct = 0.3

    # Confidence
    if signal in ("BUY", "SELL"):
        base = int(pct * 60 + 20)
        adx = indicators.get("adx", 0)
        if adx < 20:
            base = max(20, base - 15)
        confidence = min(base, 85)
    else:
        confidence = 20

    price = indicators["price"]
    support = indicators.get("support", 0)
    resistance = indicators.get("resistance", 0)
    if support and resistance and support < price < resistance:
        stop_loss = round(support, 2)
        target = round(resistance, 2)
    else:
        stop_loss = round(price - 1.5 * indicators["atr"], 2) if indicators["atr"] else 0
        target = round(price + 1.5 * indicators["atr"], 2) if indicators["atr"] else 0

    accuracy = get_accuracy()

    # Normalized scores for display (same scale as before)
    n_lstm = round(float(s_lstm * w_lstm), 3) if has_lstm else 0
    n_ta = round(float(s_ta * 0.35), 3)
    n_vol = round(float(s_vol * 0.15), 3)
    n_tf = round(float(s_tf * 0.15), 3)
    denom = w_lstm + 0.35 + 0.15 + 0.15
    total = round(float((n_lstm + n_ta + n_vol + n_tf) / denom if denom > 0 else 0), 3)

    return {
        "signal": signal,
        "confidence": confidence,
        "lstm_prediction_pct": lstm_prediction_pct,
        "indicators": indicators,
        "ensemble": {
            "lstm": n_lstm,
            "technical": n_ta,
            "volume": n_vol,
            "multi_tf": n_tf,
            "total": total,
            "consensus": buys,
            "votes": {"buy": buys, "sell": sells, "neutral": neutrals},
        },
        "stop_loss": stop_loss,
        "target": target,
        "backtest_accuracy": accuracy,
    }
