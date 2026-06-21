import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from data.market import fetch_data
from data.indicators import add_indicators
from models.lstm import build_lstm
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_DIR = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(SAVED_DIR, exist_ok=True)

FEATURE_COLS = ["Close", "Volume", "rsi", "macd", "macd_signal", "sma_20", "sma_50"]
_models = {}


def _model_path(symbol):
    return os.path.join(SAVED_DIR, f"lstm_{symbol.replace('=', '_').replace('/', '_')}.keras")


def _prepare_data(symbol):
    df = fetch_data(symbol, config.LOOKBACK_DAYS)
    df = add_indicators(df)
    df = df.dropna().copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLS].values)
    X, y = [], []
    for i in range(config.SEQUENCE_LENGTH, len(scaled)):
        X.append(scaled[i - config.SEQUENCE_LENGTH : i])
        future = (df["Close"].iloc[i] - df["Close"].iloc[i - 1]) / df["Close"].iloc[i - 1]
        y.append(future)
    return np.array(X), np.array(y), scaler, df


def train_model(symbol, force_retrain=False):
    model_path = _model_path(symbol)
    if not force_retrain and os.path.exists(model_path):
        model = load_model(model_path)
        _, _, scaler, df = _prepare_data(symbol)
        _models[symbol] = (model, scaler)
        return model, scaler, df

    X, y, scaler, df = _prepare_data(symbol)
    split = int(len(X) * config.TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm((X.shape[1], X.shape[2]))
    model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=0,
    )
    model.save(model_path)
    _models[symbol] = (model, scaler)
    return model, scaler, df


def preload_model():
    sym = config.DEFAULT_TRAIN_SYMBOL
    try:
        train_model(sym, force_retrain=False)
    except Exception:
        try:
            train_model(sym, force_retrain=True)
        except Exception as e:
            pass


def _get_model(symbol):
    if symbol in _models:
        return _models[symbol]
    return train_model(symbol)


def predict(model, scaler, df):
    last_seq = df[FEATURE_COLS].iloc[-config.SEQUENCE_LENGTH:].values
    last_seq_scaled = scaler.transform(last_seq)
    last_seq_scaled = last_seq_scaled.reshape(1, config.SEQUENCE_LENGTH, len(FEATURE_COLS))
    pred = model.predict(last_seq_scaled, verbose=0)[0][0]
    return round(float(pred) * 100, 2)
