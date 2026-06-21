import os, sys, json, logging, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
if not TOKEN:
    token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.txt")
    if os.path.exists(token_file):
        with open(token_file) as f:
            TOKEN = f.read().strip()
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN non impostata")

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
AUTHORIZED_USERS = set()

def _load_auth():
    global AUTHORIZED_USERS
    if os.path.exists(CFG_FILE):
        try:
            with open(CFG_FILE) as f:
                d = json.load(f)
                AUTHORIZED_USERS = set(d.get("authorized", []))
        except:
            AUTHORIZED_USERS = set()

def _esc(text):
    return str(text).replace("_", "\\_").replace("*", "\\*").replace("`", "\\`")

def _save_auth():
    os.makedirs(os.path.dirname(CFG_FILE), exist_ok=True)
    with open(CFG_FILE, "w") as f:
        json.dump({"authorized": list(AUTHORIZED_USERS)}, f, indent=2)

_load_auth()

from data.market import fetch_data, resolve_symbol, get_news_text, get_asset_name
from data.indicators import add_indicators, get_latest_indicators
from advisor.analyser import analyse
from advisor.portfolio import Portfolio
from config import FOREX_PAIRS, STOCKS, COMMODITIES, CRYPTO

ALL_ASSETS = list(FOREX_PAIRS.items()) + list(STOCKS.items()) + list(COMMODITIES.items()) + list(CRYPTO.items())

SPARK_CHARS = "▁▂▃▄▅▆▇█"

def _sparkline(data, width=10):
    if not data or len(data) < 2:
        return ""
    step = max(1, len(data) // width)
    sampled = data[::step][:width]
    mn, mx = min(sampled), max(sampled)
    rng = mx - mn if mx != mn else 1
    return "".join(SPARK_CHARS[min(int((v - mn) / rng * 7), 7)] for v in sampled)

def _gemini_advice(asset_name, symbol, analysis, news_text=""):
    from google import genai
    client = genai.Client(api_key=GEMINI_KEY)
    i = analysis["indicators"]
    signal = analysis["signal"]
    conf = analysis["confidence"]
    verdict = "INVESTI" if signal == "BUY" and conf >= 50 else "NON INVESTIRE"
    prompt = (
        f"Sei un analista finanziario. Rispondi in italiano.\n"
        f"Analisi {asset_name}: segnale {verdict} "
        f"(segnale={signal}, conf={conf}%, "
        f"prezzo=${i['price']}, RSI={i['rsi']}, MACD={i['macd_status']}, "
        f"trend={i['trend']}, ensemble={analysis['ensemble']['consensus']}/4). "
        f"{news_text} "
        f"Conferma o meno '{verdict}' e spiega perche'. "
        f"Termina con: DISCLAIMER: progetto educativo."
    )
    try:
        resp = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return resp.text
    except Exception as e:
        return f"{verdict} (conf: {conf}%) | {asset_name} a ${i['price']}, trend {i['trend']}. DISCLAIMER: progetto educativo"

def _gemini_chat(message, history=None):
    from google import genai
    client = genai.Client(api_key=GEMINI_KEY)
    prompt = "Sei un assistente AI utile, esperto in finanza ed economia. Rispondi in modo diretto e conciso, senza presentarti ogni volta. Parla italiano.\n\n"
    if history:
        ctx = "\n".join(f"{h['role']}: {h['msg']}" for h in history[-6:])
        prompt += ctx + "\n\n"
    user_msg = f"user: {message}"
    prompt += user_msg
    try:
        resp = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return resp.text
    except Exception as e:
        return f"Errore AI: {str(e)[:100]}"

if GEMINI_KEY:
    USE_GEMINI = True
    get_advice = _gemini_advice
    chat_ai = _gemini_chat
else:
    USE_GEMINI = False
    from advisor.reporter import get_advice, chat as chat_ai

def _get_analysis(symbol, label):
    df = fetch_data(symbol, 90)
    df = add_indicators(df)
    prices = df["Close"].values[-30:].tolist()
    spk = _sparkline(prices, 8)
    ind = get_latest_indicators(df)
    lstm_pred = 0
    if not USE_GEMINI:
        try:
            from models.trainer import train_model, predict
            lstm_model = train_model(symbol)
            lstm_pred = predict(lstm_model, symbol)
            lstm_pred = round(float(lstm_pred), 2)
        except:
            pass
    analysis = analyse(df, lstm_pred)
    return analysis, ind, spk

_chat_history = {}

def _get_history(user_id):
    if user_id not in _chat_history:
        _chat_history[user_id] = []
    return _chat_history[user_id]

def authorized(func):
    async def wrapper(update, context):
        uid = update.effective_user.id
        if uid not in AUTHORIZED_USERS:
            await update.message.reply_text("Bot privato. Non sei autorizzato.")
            return
        return await func(update, context)
    return wrapper

def format_asset_list():
    lines = ["📋 *Asset disponibili:*\n"]
    for cat, items in [("💶 Forex", FOREX_PAIRS), ("📈 Stocks", STOCKS), ("🏅 Commodities", COMMODITIES), ("₿ Crypto", CRYPTO)]:
        names = "\n".join(f"  • `{n}`" for n in items)
        lines.append(f"*{cat}*\n{names}\n")
    return "\n".join(lines)

def format_portfolio():
    pf = Portfolio()
    prices = {}
    for _, sym in ALL_ASSETS:
        try:
            from data.market import get_current_price
            p = get_current_price(sym)
            if p:
                prices[sym] = p
        except:
            pass
    s = pf.summary(prices)
    lines = [f"📊 *Portafoglio Virtuale*"]
    lines.append(f"💰 Capitale iniziale: ${pf.initial_capital:,.2f}")
    lines.append(f"📈 Valore investito: ${s['invested']:,.2f}")
    lines.append(f"💵 Valore corrente: ${s['current_value']:,.2f}")
    lines.append(f"📉 Closed P&L: ${s['closed_pnl']:,.2f}")
    lines.append(f"📊 *Rendimento totale: {s['total_return']:+.2f}%*")
    lines.append(f"📦 Posizioni aperte: {s['num_positions']}")
    lines.append(f"🔄 Trade chiusi: {s['num_trades']}")
    if pf.positions:
        lines.append("\n*Posizioni aperte:*")
        for sym, pos in pf.positions.items():
            cur = prices.get(sym, 0)
            pnl = pf.pnl(sym, cur)
            icon = "🟢" if pnl >= 0 else "🔴"
            name = get_asset_name(sym)
            lines.append(f"  {icon} {name}: {pos['qty']}x @ ${pos['avg_price']:.2f} → ${cur:.2f} ({pnl:+.2f})")
    return "\n".join(lines)

async def start(update, context):
    uid = update.effective_user.id
    name = update.effective_user.first_name or "Utente"
    if uid not in AUTHORIZED_USERS:
        AUTHORIZED_USERS.add(uid)
        _save_auth()
        await update.message.reply_text(
            f"✅ Benvenuto {name}! Sei stato autorizzato come unico utente del bot.\n\n"
            f"Comandi:\n"
            f"`/lista` - Tutti gli asset\n"
            f"`/analizza <nome>` - Analisi completa\n"
            f"`/portafoglio` - P&L virtuale\n"
            f"`/chat <msg>` - Parla con l'AI\n\n"
            f"Invia qualsiasi messaggio per chattare con l'AI."
        )
    else:
        await update.message.reply_text(
            f"Ciao {name}! Sono il tuo trading advisor.\n"
            f"`/lista` `/analizza <nome>` `/portafoglio` `/chat <msg>`"
        )

@authorized
async def lista(update, context):
    await update.message.reply_text(format_asset_list(), parse_mode="Markdown")

@authorized
async def portafoglio(update, context):
    msg = format_portfolio()
    await update.message.reply_text(msg, parse_mode="Markdown")

@authorized
async def analizza(update, context):
    if not context.args:
        await update.message.reply_text("Usa: `/analizza Tesla` o `/analizza oro`", parse_mode="Markdown")
        return
    text = " ".join(context.args)
    await update.message.reply_text(f"🔍 Cerco {_esc(text)}...")
    try:
        label, symbol = resolve_symbol(text)
    except:
        await update.message.reply_text(f"❌ Asset '{text}' non trovato.")
        return
    status_msg = await update.message.reply_text(f"⏳ Analisi {label} in corso...")
    try:
        analysis, ind, spk = _get_analysis(symbol, label)
        sig = analysis["signal"]
        conf = analysis["confidence"]
        sl = analysis["stop_loss"]
        tg = analysis["target"]
        acc = analysis["backtest_accuracy"]
        e = analysis["ensemble"]
        emoji_sig = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}.get(sig, sig)
        lines = [
            f"📊 *{_esc(label)}* — ${ind['price']:.2f}",
            f"📉 Trend: `{_esc(spk)}` | RSI {ind['rsi']} | MACD {_esc(ind['macd_status'])}",
            f"📊 Stoccastico: {ind.get('stoch', 'N/A')} | OBV: {_esc(ind.get('obv_status', 'N/A'))}",
            f"",
            f"📈 Segnale: *{_esc(emoji_sig)}* (conf: {conf}%)",
            f"🎯 Target: ${tg:.2f} | 🛑 Stop: ${sl:.2f}",
            f"",
            f"🧩 Ensemble: LSTM {e['lstm']:.2f} | TA {e['technical']:.2f} | Vol {e['volume']:.2f} | TF {e['multi_tf']:.2f}",
            f"📊 Accuratezza backtest: {acc}%" if acc else "",
        ]
        await status_msg.edit_text("\n".join(filter(None, lines)), parse_mode="Markdown")
        news = get_news_text(symbol)
        final_msg = await update.message.reply_text(f"⏳ Consulto AI per verdetto finale...")
        advice = get_advice(label, symbol, analysis, news)
        await final_msg.edit_text(f"🧠 Verdetto AI:\n{advice}")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore analisi {label}: {str(e)[:200]}")

@authorized
async def chat_cmd(update, context):
    if not context.args:
        await update.message.reply_text("Usa: `/chat cosa ne pensi del mercato?`", parse_mode="Markdown")
        return
    msg = " ".join(context.args)
    uid = update.effective_user.id
    history = _get_history(uid)
    await update.message.reply_text("💬 AI sta pensando...")
    reply = chat_ai(msg, history)
    if reply:
        history.append({"role": "user", "msg": msg})
        history.append({"role": "assistant", "msg": reply})
        if len(history) > 12:
            _chat_history[uid] = history[-12:]
    await update.message.reply_text(reply[:4000])

@authorized
async def handle_message(update, context):
    text = update.message.text.strip()
    uid = update.effective_user.id
    history = _get_history(uid)
    await update.message.reply_text("💬 AI sta pensando...")
    reply = chat_ai(text, history)
    if reply:
        history.append({"role": "user", "msg": text})
        history.append({"role": "assistant", "msg": reply})
        if len(history) > 12:
            _chat_history[uid] = history[-12:]
    await update.message.reply_text(reply[:4000])

def _health_server():
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        def log_message(self, *a):
            pass
    port = int(os.environ.get("PORT", 8080))
    s = HTTPServer(("0.0.0.0", port), H)
    s.serve_forever()

def start_bot():
    t = threading.Thread(target=_health_server, daemon=True)
    t.start()
    from telegram.ext import Application, CommandHandler, MessageHandler, filters
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("lista", lista))
    app.add_handler(CommandHandler("portafoglio", portafoglio))
    app.add_handler(CommandHandler("analizza", analizza))
    app.add_handler(CommandHandler("chat", chat_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    mode = "Gemini" if USE_GEMINI else "Ollama"
    print(f" Telegram Bot avviato su @oracle_fx_bot (AI: {mode})")
    app.run_polling(drop_pending_updates=True)

def run():
    start_bot()

if __name__ == "__main__":
    run()
