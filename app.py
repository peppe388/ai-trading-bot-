import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gradio as gr
import config
from data.market import fetch_data, resolve_symbol, get_asset_name
from data.indicators import add_indicators
from models.trainer import train_model, predict, preload_model
from advisor.analyser import analyse, record_prediction
from advisor.reporter import get_advice
from advisor.chart import generate_chart

DARK_CSS = """
:root {
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --green: #00ff88;
    --red: #ff4444;
    --yellow: #ffaa00;
}
.gradio-container { background: var(--bg) !important; max-width: 900px; margin: auto; }
footer { display: none !important; }
.gr-box, .panel, .card, .tab-nav, .form { background: var(--card) !important; border-color: var(--border) !important; }
label, .label-text { color: var(--text) !important; }
input, textarea { background: #0d1117 !important; color: var(--text) !important; border-color: var(--border) !important; }
.markdown, .prose, p, h1, h2, h3, h4 { color: var(--text) !important; }
h1 { font-size: 1.5em !important; margin-bottom: 0.3em !important; }
h3 { color: var(--text-muted) !important; font-size: 0.9em !important; }
.message { background: var(--card) !important; border: 1px solid var(--border) !important; }
"""





def chat_response(message, history):
    msg = message.strip()
    if not msg:
        yield "Scrivi un asset da analizzare (es. EUR/USD, oro, TSLA, BTC)."
        return

    asset_name, symbol = resolve_symbol(msg)
    asset_label = get_asset_name(symbol)

    yield {"role": "assistant", "content": [{"type": "text", "text": f"Analizzo {asset_label} ({symbol})..."}]}

    try:
        df = fetch_data(symbol, config.LOOKBACK_DAYS)
    except Exception as e:
        yield {"role": "assistant", "content": [{"type": "text", "text": f"Errore: {str(e)}"}]}
        return

    df = add_indicators(df)

    yield {"role": "assistant", "content": [{"type": "text", "text": "Calcolo indicatori e predizione AI..."}]}

    try:
        model, scaler, _ = train_model(symbol)
        lstm_pred = predict(model, scaler, df)
    except Exception as e:
        yield {"role": "assistant", "content": [{"type": "text", "text": f"Errore modello: {str(e)}"}]}
        return

    analysis = analyse(df, lstm_pred)

    later_change = float(df["Close"].tail(2).diff().iloc[-1] / df["Close"].iloc[-2] * 100) if len(df) >= 2 else 0
    record_prediction(symbol, lstm_pred, later_change, (lstm_pred > 0) == (later_change > 0))

    yield {"role": "assistant", "content": [{"type": "text", "text": "Genero report e grafico..."}]}

    try:
        reply = get_advice(asset_label, symbol, analysis)
    except Exception:
        i = analysis["indicators"]
        reply = (
            f"{asset_label} - ${i['price']} | {i['trend'].upper()}\n"
            f"RSI {i['rsi']} | MACD {i['macd_status']} | Stoc {i['stoch']}\n"
            f"{analysis['signal']} (conf: {analysis['confidence']}%)\n"
            f"DISCLAIMER: progetto educativo"
        )

    chart_path = None
    try:
        chart_path = generate_chart(df, asset_label, analysis["signal"])
    except Exception:
        pass

    content = [{"type": "text", "text": reply}]
    if chart_path:
        content.append({"type": "file", "file": {"path": chart_path}})

    yield {"role": "assistant", "content": content}


with gr.Blocks(title="AI Trading Advisor Pro") as demo:
    gr.HTML("""
    <div style="text-align:center; padding: 10px 0;">
        <h1 style="color:#00ff88; margin:0;">AI Trading Advisor Pro</h1>
        <p style="color:#8b949e; margin:0;">Analisi Forex · Azioni · Crypto · Commodities</p>
    </div>
    """)

    chat = gr.ChatInterface(
        fn=chat_response,
        textbox=gr.Textbox(placeholder='Scrivi un asset: EUR/USD, oro, TSLA, bitcoin...'),
    )

    gr.HTML("""
    <div style="display:flex; flex-wrap:wrap; gap:6px; justify-content:center; padding:10px;">
        <span onclick="document.querySelector('textarea').value='EUR/USD'; document.querySelector('textarea').dispatchEvent(new Event('input'))" style="cursor:pointer; background:#21262d; color:#e6edf3; padding:4px 12px; border-radius:12px; font-size:0.85em; border:1px solid #30363d;">EUR/USD</span>
        <span onclick="document.querySelector('textarea').value='GBP/USD'; document.querySelector('textarea').dispatchEvent(new Event('input'))" style="cursor:pointer; background:#21262d; color:#e6edf3; padding:4px 12px; border-radius:12px; font-size:0.85em; border:1px solid #30363d;">GBP/USD</span>
        <span onclick="document.querySelector('textarea').value='oro'; document.querySelector('textarea').dispatchEvent(new Event('input'))" style="cursor:pointer; background:#21262d; color:#e6edf3; padding:4px 12px; border-radius:12px; font-size:0.85em; border:1px solid #30363d;">Oro</span>
        <span onclick="document.querySelector('textarea').value='bitcoin'; document.querySelector('textarea').dispatchEvent(new Event('input'))" style="cursor:pointer; background:#21262d; color:#e6edf3; padding:4px 12px; border-radius:12px; font-size:0.85em; border:1px solid #30363d;">Bitcoin</span>
        <span onclick="document.querySelector('textarea').value='TSLA'; document.querySelector('textarea').dispatchEvent(new Event('input'))" style="cursor:pointer; background:#21262d; color:#e6edf3; padding:4px 12px; border-radius:12px; font-size:0.85em; border:1px solid #30363d;">TSLA</span>
    </div>
    <div style="text-align:center; color:#8b949e; font-size:0.8em; padding-bottom:5px;">
        Forex: EUR/USD · GBP/USD · USD/JPY · USD/CHF · AUD/USD · USD/CAD · NZD/USD · EUR/JPY
    </div>
    """)


if __name__ == "__main__":
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 7860
    while sock.connect_ex(("127.0.0.1", port)) == 0:
        port += 1
    sock.close()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        css=DARK_CSS,
        quiet=True,
    )
