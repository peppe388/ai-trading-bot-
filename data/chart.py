import os, tempfile
import pandas as pd
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

DARK_BG = '#131722'
DARK_GRID = '#2a2e39'
GREEN = '#26a69a'
RED = '#ef5350'
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
    except:
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
