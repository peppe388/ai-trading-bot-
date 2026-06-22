import os, sys, json, logging, threading, sqlite3
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

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auth.db")
AUTHORIZED_USERS = set()

def _load_auth():
    global AUTHORIZED_USERS
    admin = os.environ.get("ADMIN_ID", "")
    if admin and admin.isdigit():
        AUTHORIZED_USERS.add(int(admin))
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("CREATE TABLE IF NOT EXISTS auth (user_id INTEGER PRIMARY KEY)")
        for row in conn.execute("SELECT user_id FROM auth"):
            AUTHORIZED_USERS.add(row[0])
        conn.close()
    except:
        pass

def _add_user(user_id):
    AUTHORIZED_USERS.add(user_id)
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT OR IGNORE INTO auth (user_id) VALUES (?)", (user_id,))
        conn.commit()
        conn.close()
    except:
        pass

def _del_user(user_id):
    AUTHORIZED_USERS.discard(user_id)
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("DELETE FROM auth WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
    except:
        pass

def _esc(text):
    return str(text).replace("_", "\\_").replace("*", "\\*").replace("`", "\\`")

_load_auth()

from data.market import fetch_data, resolve_symbol, get_news_text, get_asset_name
from data.indicators import add_indicators, get_latest_indicators
from advisor.analyser import analyse
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

def _groq_advice(asset_name, symbol, analysis, news_text=""):
    from groq import Groq
    client = Groq(api_key=GROQ_KEY)
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
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=512,
        )
        return resp.choices[0].message.content
    except Exception:
        return f"{verdict} (conf: {conf}%) | {asset_name} a ${i['price']}, trend {i['trend']}. DISCLAIMER: progetto educativo"

def _groq_chat(message, history=None):
    from groq import Groq
    client = Groq(api_key=GROQ_KEY)
    system = {"role": "system", "content": "Sei un assistente AI utile, esperto in finanza ed economia. Rispondi in modo diretto e conciso, senza presentarti ogni volta. Parla italiano."}
    messages = [system]
    if history:
        for h in history[-6:]:
            messages.append({"role": h["role"], "content": h["msg"]})
    messages.append({"role": "user", "content": message})
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7, max_tokens=512,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Errore AI: {str(e)[:100]}"

def _translate(text):
    if not GROQ_KEY:
        return text
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"Traduci in italiano, mantieni il formato (titoli separati da -):\n{text}"}],
            temperature=0.1, max_tokens=1024,
        )
        return resp.choices[0].message.content
    except:
        return text

if GROQ_KEY:
    USE_GROQ = True
    get_advice = _groq_advice
    chat_ai = _groq_chat
else:
    USE_GROQ = False
    from advisor.reporter import get_advice, chat as chat_ai

def _get_analysis(symbol, label):
    df = fetch_data(symbol, 90)
    df = add_indicators(df)
    prices = df["Close"].values[-30:].tolist()
    spk = _sparkline(prices, 8)
    ind = get_latest_indicators(df)
    lstm_pred = 0
    if not USE_GROQ:
        try:
            from models.trainer import train_model, predict
            model_trained, scaler_trained, _ = train_model(symbol)
            lstm_pred = predict(model_trained, scaler_trained, df)
            lstm_pred = round(float(lstm_pred), 2)
        except:
            pass
    analysis = analyse(df, lstm_pred)
    return analysis, ind, spk

_chat_history = {}
_chat_mode = set()

def _get_history(user_id):
    if user_id not in _chat_history:
        _chat_history[user_id] = []
    return _chat_history[user_id]

def authorized(func):
    async def wrapper(update, context):
        uid = update.effective_user.id
        if uid not in AUTHORIZED_USERS:
            _load_auth()
            if uid not in AUTHORIZED_USERS:
                await update.message.reply_text("Bot privato. Non sei autorizzato. Contatta l'admin.")
                return
        return await func(update, context)
    return wrapper

def format_asset_list():
    lines = ["📋 *Asset disponibili:*\n"]
    for cat, items in [("💶 Forex", FOREX_PAIRS), ("📈 Stocks", STOCKS), ("🏅 Commodities", COMMODITIES), ("₿ Crypto", CRYPTO)]:
        names = "\n".join(f"  • `{n}`" for n in items)
        lines.append(f"*{cat}*\n{names}\n")
    return "\n".join(lines)

def format_news():
    from data.market import get_news
    news = get_news("SPY", 8)
    if not news:
        return "Nessuna notizia disponibile."
    raw = "\n".join(f"- {n['title']} ({n['publisher']})" for n in news)
    translated = _translate(raw)
    lines = ["📰 *Ultime notizie di mercato:*\n"]
    for line in translated.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)
            lines.append("")
    return "\n".join(lines)

async def start(update, context):
    uid = update.effective_user.id
    admin = os.environ.get("ADMIN_ID", "")
    if admin and admin.isdigit() and uid == int(admin):
        _add_user(uid)
        await update.message.reply_text(
            "✅ Sei l'admin. Bot avviato.\n"
            f"`/aggiungi <ID>` - Aggiungi un utente\n"
            f"`/rimuovi <ID>` - Rimuovi un utente\n"
            f"`/lista` - Asset disponibili\n"
            f"`/analizza <nome>` - Analisi\n"
            f"`/notizie` - Ultime notizie\n"
            f"`/notizie <nome>` - Notizie su un asset\n"
            f"`/chat <msg>` - Parla con l'AI"
        )
    else:
        await update.message.reply_text(
            "🤖 Bot privato.\n"
            "Se non sei autorizzato, contatta l'admin."
        )

async def aggiungi(update, context):
    uid = update.effective_user.id
    admin = os.environ.get("ADMIN_ID", "")
    if not (admin and admin.isdigit() and uid == int(admin)):
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usa: /aggiungi <ID>")
        return
    target = int(context.args[0])
    _add_user(target)
    await update.message.reply_text(f"✅ Utente `{target}` autorizzato.")

async def rimuovi(update, context):
    uid = update.effective_user.id
    admin = os.environ.get("ADMIN_ID", "")
    if not (admin and admin.isdigit() and uid == int(admin)):
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usa: /rimuovi <ID>")
        return
    target = int(context.args[0])
    _del_user(target)
    await update.message.reply_text(f"🗑️ Utente `{target}` rimosso.")

@authorized
async def lista(update, context):
    await update.message.reply_text(format_asset_list(), parse_mode="Markdown")

@authorized
async def notizie_cmd(update, context):
    text = " ".join(context.args) if context.args else ""
    if text:
        try:
            label, symbol = resolve_symbol(text)
            raw = get_news_text(symbol)
            if raw:
                translated = _translate(raw)
                await update.message.reply_text(f"📰 *Notizie per {label}:*\n{translated}", parse_mode="Markdown")
            else:
                await update.message.reply_text(f"❌ Nessuna notizia per {label}.")
        except:
            await update.message.reply_text(f"❌ Asset '{text}' non trovato.")
    else:
        msg = format_news()
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
            f"📊 Stoccastico: {ind.get('stoch', 'N/A')} | OBV: {_esc(ind.get('obv_trend', 'N/A'))}",
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
    uid = update.effective_user.id
    _chat_mode.add(uid)
    if not context.args:
        await update.message.reply_text(
            "💬 *Modalità chat attivata!*\n\n"
            "Ora puoi scrivere qualsiasi messaggio e parlerò con l'AI.\n"
            "Usa `/stop` per uscire dalla modalità chat.",
            parse_mode="Markdown"
        )
        return
    msg = " ".join(context.args)
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
async def stop_cmd(update, context):
    uid = update.effective_user.id
    _chat_mode.discard(uid)
    await update.message.reply_text("🚪 Modalità chat disattivata. Usa `/chat` per riattivarla.", parse_mode="Markdown")

@authorized
async def handle_message(update, context):
    uid = update.effective_user.id
    if uid not in _chat_mode:
        return
    text = update.message.text.strip()
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
    app.add_handler(CommandHandler("aggiungi", aggiungi))
    app.add_handler(CommandHandler("rimuovi", rimuovi))
    app.add_handler(CommandHandler("lista", lista))
    app.add_handler(CommandHandler("notizie", notizie_cmd))
    app.add_handler(CommandHandler("analizza", analizza))
    app.add_handler(CommandHandler("chat", chat_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("esci", stop_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    mode = "Groq-Llama3.3" if USE_GROQ else "Ollama"
    print(f" Telegram Bot avviato su @oracle_fx_bot (AI: {mode})")
    app.run_polling(drop_pending_updates=True)

def run():
    start_bot()

if __name__ == "__main__":
    run()
