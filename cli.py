import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.market import fetch_data, resolve_symbol, get_asset_name, get_news_text
from data.indicators import add_indicators
from models.trainer import train_model, predict
from advisor.analyser import analyse, record_prediction
from advisor.reporter import get_advice, chat
from advisor.chart import generate_chart

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; N = "\033[0m"

_asset_names = set()
for d in [config.FOREX_PAIRS, config.STOCKS, config.COMMODITIES, config.CRYPTO]:
    _asset_names.update(v.lower() for v in d.values())
    _asset_names.update(k.lower().split(" (")[0] for k in d)
_asset_names.update(config.COMMON_NAMES.keys())

_chat_history = []


def is_asset_query(msg):
    m = msg.lower().strip()
    for kw in ["analizza","analisi","investi","comprare","vendere","conviene"]:
        if m.startswith(kw): return True
    for name in sorted(_asset_names, key=len, reverse=True):
        if name in m: return True
    for symbol in list(config.FOREX_PAIRS.values())+list(config.STOCKS.values())+list(config.COMMODITIES.values())+list(config.CRYPTO.values()):
        if symbol.lower() in m: return True
    for pair in config.FOREX_PAIRS:
        if pair.lower() in m: return True
    return False


def print_header():
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{C}{B}")
    print("  +======================================+")
    print("  |    AI TRADING ADVISOR PRO            |")
    print("  |  Analisi + Chat finanziaria          |")
    print("  +======================================+")
    print(f"{N}")
    print("  EUR/USD, oro, TSLA, BTC  -> analisi INVESTI/NON INVESTIRE")
    print("  Domande generiche        -> chat AI libera")
    print("  'exit' per uscire\n")


def handle_asset(msg):
    asset_name, symbol = resolve_symbol(msg)
    label = get_asset_name(symbol)
    print(f"  Analizzo {label} ({symbol})...")
    t0 = time.time()
    try: df = fetch_data(symbol, config.LOOKBACK_DAYS)
    except Exception as e: print(f"  {R}Errore dati: {e}{N}"); return
    df = add_indicators(df)
    try:
        model, scaler, _ = train_model(symbol)
        lstm_pred = predict(model, scaler, df)
    except Exception as e: print(f"  {R}Errore AI: {e}{N}"); return
    analysis = analyse(df, lstm_pred)
    later = 0
    if len(df) >= 2:
        later = float(df["Close"].tail(2).diff().iloc[-1] / df["Close"].iloc[-2] * 100)
    record_prediction(symbol, lstm_pred, later, (lstm_pred > 0) == (later > 0))
    i = analysis["indicators"]; e = analysis["ensemble"]; acc = analysis["backtest_accuracy"]
    t1 = time.time()
    print(f"\n  {B}{label}{N} - ${i['price']} ({i['price_change_pct']}%)  |  {t1-t0:.1f}s")
    print(f"  {'-'*40}")
    sig = analysis["signal"]
    inv = "INVESTI" if sig == "BUY" and analysis["confidence"] >= 50 else "NON INVESTIRE"
    print(f"  RSI: {i['rsi']} ({i['rsi_signal']})    | MACD: {i['macd_status']}")
    print(f"  Stoc: {i['stoch']}    | OBV: {i['obv_trend']}")
    print(f"  AI pred: {analysis['lstm_prediction_pct']}%     | Ensemble: {e['consensus']}/4")
    print(f"  AI acc: {i['trend']}")
    print(f"  {'-'*40}")
    print(f"  {inv} (conf: {analysis['confidence']}%)", end="")
    if acc: print(f"  |  Backtest: {acc}%", end="")
    print(f"\n  Stop: ${analysis['stop_loss']}  |  Target: ${analysis['target']}\n")

    print(f"  {C}Notizie recenti:{N}")
    news = get_news_text(symbol)
    print(f"  {news or 'Nessuna notizia disponibile'}")
    print()

    try:
        reply = get_advice(asset_name, symbol, analysis, news)
        print(f"  {reply}\n")
    except Exception:
        pass
    try:
        cp = generate_chart(df, label, analysis["signal"])
        print(f"  Grafico: {cp}")
    except Exception:
        pass
    print(f"  {C}{'-'*40}{N}\n")


def handle_chat(msg):
    global _chat_history
    _chat_history.append({"role": "user", "msg": msg})
    print("  ...")
    t0 = time.time()
    reply = chat(msg, _chat_history)
    if reply:
        print(f"\n  {reply}")
        _chat_history.append({"role": "assistant", "msg": reply})
    else:
        print(f"\n  {Y}Ollama non risponde.{N}")
    print(f"  ({time.time()-t0:.1f}s)\n")


def main():
    print_header()
    while True:
        try: msg = input(f"{C}> {N}").strip()
        except (EOFError, KeyboardInterrupt): print(f"\n{Y}Arrivederci!{N}"); break
        if not msg: continue
        if msg.lower() in ("exit","quit","esci","q"): print(f"{Y}Arrivederci!{N}"); break
        if is_asset_query(msg): handle_asset(msg)
        else: handle_chat(msg)

if __name__ == "__main__":
    main()
