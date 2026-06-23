import pandas as pd
import numpy as np

try:
    import ta
except ImportError:
    raise ImportError(
        "Libreria 'ta' mancante. Installala con: pip install ta"
    )


def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()

    df["sma_20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()

    bb = ta.volatility.BollingerBands(close, window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    return df


def get_latest_indicators(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi_val = last["rsi"]
    rsi_signal = "ipercomprato" if rsi_val > 70 else "ipervenduto" if rsi_val < 30 else "neutrale"

    macd_bullish = last["macd"] > last["macd_signal"]
    macd_status = "bullish" if macd_bullish else "bearish"

    stoch_val = last["stoch_k"]
    stoch_signal = "ipercomprato" if stoch_val > 80 else "ipervenduto" if stoch_val < 20 else "neutrale"

    obv_val = last["obv"]
    obv_prev = prev["obv"]
    obv_trend = "positivo" if obv_val > obv_prev else "negativo"

    sma_bullish = last["sma_20"] > last["sma_50"]
    trend = "rialzista" if sma_bullish else "ribassista"

    bb_pos = (last["Close"] - last["bb_low"]) / (last["bb_high"] - last["bb_low"])
    bb_signal = "vicino banda sup" if bb_pos > 0.95 else "vicino banda inf" if bb_pos < 0.05 else "centrato"

    price_change = ((last["Close"] - prev["Close"]) / prev["Close"]) * 100

    adx_val = last["adx"]
    adx_signal = "trend" if adx_val > 25 else "range" if adx_val < 20 else "debole"

    window_high = df["High"].rolling(20).max()
    window_low = df["Low"].rolling(20).min()
    resistance = round(float(window_high.iloc[-2]), 2) if len(df) >= 3 else round(float(last["High"] * 1.02), 2)
    support = round(float(window_low.iloc[-2]), 2) if len(df) >= 3 else round(float(last["Low"] * 0.98), 2)

    return {
        "price": round(float(last["Close"]), 4),
        "rsi": round(float(rsi_val), 1),
        "rsi_signal": rsi_signal,
        "macd_status": macd_status,
        "stoch": round(float(stoch_val), 1),
        "stoch_signal": stoch_signal,
        "obv_trend": obv_trend,
        "trend": trend,
        "sma_20": round(float(last["sma_20"]), 4),
        "sma_50": round(float(last["sma_50"]), 4),
        "atr": round(float(last["atr"]), 4),
        "bb_signal": bb_signal,
        "price_change_pct": round(price_change, 2),
        "adx": round(float(adx_val), 1),
        "adx_signal": adx_signal,
        "resistance": resistance,
        "support": support,
    }
