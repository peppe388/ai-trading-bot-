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


def analyse(df, lstm_prediction_pct):
    indicators = get_latest_indicators(df)

    s_lstm = _lstm_score(lstm_prediction_pct)
    w_lstm = _lstm_weight(lstm_prediction_pct)

    s_ta = _ta_ensemble(indicators)
    w_ta = 0.35

    s_vol = _volume_score(df)
    w_vol = 0.15

    s_tf = _multi_tf_score(df)
    w_tf = 0.15

    total = s_lstm * w_lstm + s_ta * w_ta + s_vol * w_vol + s_tf * w_tf
    max_possible = w_lstm + w_ta + w_vol + w_tf
    normalized = total / max_possible if max_possible > 0 else 0

    consensus = sum(1 for s in [s_lstm > 0, s_ta > 0, s_vol > 0, s_tf > 0])
    discord = sum(1 for s in [s_lstm < 0, s_ta < 0, s_vol < 0, s_tf < 0])

    if abs(normalized) < 0.15 or (consensus >= 1 and discord >= 1 and abs(normalized) < 0.3):
        signal = "HOLD"
        confidence = max(15, int(abs(normalized) * 100))
    elif normalized > 0.2:
        signal = "BUY"
        confidence = min(95, int(abs(normalized) * 100))
    elif normalized < -0.2:
        signal = "SELL"
        confidence = min(95, int(abs(normalized) * 100))
    else:
        signal = "HOLD"
        confidence = max(15, int(abs(normalized) * 100))

    stop_loss = round(indicators["price"] - 1.5 * indicators["atr"], 4) if indicators["atr"] else 0
    target = round(indicators["price"] + 1.5 * indicators["atr"], 4) if indicators["atr"] else 0

    accuracy = get_accuracy()

    return {
        "signal": signal,
        "confidence": min(confidence, 90),
        "lstm_prediction_pct": lstm_prediction_pct,
        "indicators": indicators,
        "ensemble": {
            "lstm": round(float(s_lstm * w_lstm), 3),
            "technical": round(float(s_ta * w_ta), 3),
            "volume": round(float(s_vol * w_vol), 3),
            "multi_tf": round(float(s_tf * w_tf), 3),
            "total": round(float(normalized), 3),
            "consensus": consensus,
        },
        "stop_loss": stop_loss,
        "target": target,
        "backtest_accuracy": accuracy,
    }
