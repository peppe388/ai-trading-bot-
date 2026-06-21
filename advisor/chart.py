import os
import tempfile
import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")


_dark_style = mpf.make_mpf_style(
    base_mpf_style="charles",
    marketcolors=mpf.make_marketcolors(
        up="#00ff88",
        down="#ff4444",
        edge={"up": "#00ff88", "down": "#ff4444"},
        wick={"up": "#00ff88", "down": "#ff4444"},
        volume={"up": "#00ff8844", "down": "#ff444444"},
    ),
    facecolor="#0d1117",
    figcolor="#0d1117",
    gridcolor="#21262d",
    gridstyle=":",
    y_on_right=True,
)


def generate_chart(df, symbol_name, signal=None):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index)

    last_90 = df.tail(90).copy()
    apds = []
    mavs = (20, 50) if len(last_90) >= 50 else None

    if signal and signal in ["BUY", "SELL"]:
        color = "#00ff88" if signal == "BUY" else "#ff4444"
        label = f"{signal} HERE"
        apds.append(mpf.make_addplot(
            [float(last_90["Close"].iloc[-1])] * len(last_90),
            color=color,
            linestyle="--",
            linewidths=1,
            label=label,
        ))

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        mpf.plot(
            last_90,
            type="candle",
            style=_dark_style,
            title=f"{symbol_name} - Ultimi 90 giorni",
            ylabel="Prezzo",
            mav=mavs,
            volume=True,
            addplot=apds,
            savefig=tmp.name,
            figsize=(10, 6),
            tight_layout=True,
        )
    except Exception:
        mpf.plot(
            last_90,
            type="line",
            style=_dark_style,
            title=symbol_name,
            savefig=tmp.name,
            figsize=(10, 6),
        )
    return tmp.name
