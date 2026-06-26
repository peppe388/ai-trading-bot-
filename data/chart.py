import os, tempfile
import pandas as pd
import yfinance as yf
from data.market import fetch_data

CHART_ENABLED = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    CHART_ENABLED = True
except:
    pass

DARK_BG = '#0F141C'
DARK_GRID = '#1E232A'
GREEN = '#219D58'
RED = '#D5414D'
ORANGE = '#ffa726'
BLUE = '#42a5f5'

def _style():
    mc = mpf.make_marketcolors(
        up=GREEN, down=RED,
        edge='inherit', wick='inherit', volume=DARK_GRID
    )
    return mpf.make_mpf_style(
        marketcolors=mc,
        figcolor=DARK_BG, facecolor=DARK_BG,
        gridcolor=DARK_GRID, gridstyle=':',
        rc={'font.size': 9, 'axes.labelcolor': 'white',
            'axes.edgecolor': '#555', 'xtick.color': 'white',
            'ytick.color': 'white'}
    )

def create_chart(symbol, label, days=90):
    if not CHART_ENABLED:
        return None
    df = fetch_data(symbol, days)
    if df.empty or len(df) < 20:
        return None
    plot_df = df.tail(60).copy()
    apds = []
    sma20 = plot_df['Close'].rolling(20, min_periods=1).mean()
    sma50 = plot_df['Close'].rolling(50, min_periods=1).mean()
    if sma20.notna().sum() > 0:
        apds.append(mpf.make_addplot(sma20, color=ORANGE, width=0.8))
    if sma50.notna().sum() > 0:
        apds.append(mpf.make_addplot(sma50, color=BLUE, width=0.8))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    try:
        mpf.plot(
            plot_df, type='candle', style=_style(),
            addplot=apds, volume=True,
            title=label, ylabel='Prezzo ($)',
            savefig=dict(fname=tmp.name, dpi=120, facecolor=DARK_BG),
            figsize=(10, 6), tight_layout=True,
        )
        return tmp.name
    except Exception:
        try: os.unlink(tmp.name)
        except: pass
        return None

def create_comparison(sym1, sym2, label1, label2, days=90):
    if not CHART_ENABLED:
        return None
    df1, df2 = fetch_data(sym1, days), fetch_data(sym2, days)
    if df1.empty or df2.empty:
        return None
    norm1 = (df1['Close'] / df1['Close'].iloc[0] - 1) * 100
    norm2 = (df2['Close'] / df2['Close'].iloc[0] - 1) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.plot(norm1.index, norm1.values, color=GREEN, linewidth=1.5, label=label1)
    ax.plot(norm2.index, norm2.values, color=RED, linewidth=1.5, label=label2)
    ax.axhline(y=0, color='#555', linewidth=0.5)
    ax.legend(facecolor=DARK_BG, labelcolor='white', fontsize=9)
    ax.set_ylabel('Rendimento %', color='white', fontsize=9)
    ax.set_title(f'{label1} vs {label2}', color='white', fontsize=11)
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(True, alpha=0.2)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.tight_layout()
    plt.savefig(tmp.name, facecolor=fig.get_facecolor(), dpi=120)
    plt.close()
    return tmp.name

def create_live_chart(symbol, label):
    """Intraday chart (1m/15m) with daily fallback. Always fresh."""
    if not CHART_ENABLED:
        return None
    plot_df = None
    timeframe = ""
    for period, interval in [("1d", "1m"), ("5d", "15m")]:
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df.empty or len(df) < 5:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            plot_df = df.tail(30).copy()
            timeframe = interval
            break
        except Exception:
            continue
    if plot_df is None:
        try:
            df = yf.download(symbol, period="7d", interval="1d", progress=False)
            if df.empty or len(df) < 3:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            plot_df = df.tail(10).copy()
            timeframe = "daily"
        except Exception:
            return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    try:
        mc = mpf.make_marketcolors(up=GREEN, down=RED, edge='inherit', wick='inherit', volume=DARK_GRID)
        style = mpf.make_mpf_style(
            marketcolors=mc, figcolor=DARK_BG, facecolor=DARK_BG,
            gridcolor=DARK_GRID, gridstyle=':',
            rc={'font.size': 8, 'axes.labelcolor': 'white',
                'axes.edgecolor': '#555', 'xtick.color': 'white',
                'ytick.color': 'white'}
        )
        mpf.plot(plot_df, type='candle', style=style, volume=False,
                 title=f'{label} ({timeframe})', ylabel='$',
                 savefig=dict(fname=tmp.name, dpi=120, facecolor=DARK_BG),
                 figsize=(9, 4), tight_layout=True)
        return tmp.name
    except Exception:
        try: os.unlink(tmp.name)
        except: pass
        return None
