import os, sys, json, logging, threading, sqlite3, asyncio
from datetime import datetime
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
    except Exception:
        pass

def _add_user(user_id):
    AUTHORIZED_USERS.add(user_id)
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT OR IGNORE INTO auth (user_id) VALUES (?)", (user_id,))
        conn.commit()
        conn.close()
    except Exception:
        pass

def _del_user(user_id):
    AUTHORIZED_USERS.discard(user_id)
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("DELETE FROM auth WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
    except Exception:
        pass

def _esc(text):
    return str(text).replace("_", "\\_").replace("*", "\\*").replace("`", "\\`")

_load_auth()

from data.market import fetch_data, resolve_symbol, get_news_text, get_asset_name, get_current_price
from data.indicators import add_indicators, get_latest_indicators
from advisor.analyser import analyse
from data.chart import create_comparison, create_live_chart, CHART_ENABLED
from data.alerts import add_alert, remove_alert, get_user_alerts, start_checker
from advisor.backtest import run_backtest, format_backtest
from config import FOREX_PAIRS, STOCKS, COMMODITIES, CRYPTO

START_TIME = datetime.now()

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
        f"trend={i['trend']}, voto={analysis['ensemble']['votes']}). "
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
    except Exception:
        return text

if GROQ_KEY:
    USE_GROQ = True
    get_advice = _groq_advice
    chat_ai = _groq_chat
else:
    USE_GROQ = False
    from advisor.reporter import get_advice, chat as chat_ai

def _get_analysis(symbol, label):
    df = fetch_data(symbol, 365)
    df = add_indicators(df)
    prices = df["Close"].values[-30:].tolist()
    spk = _sparkline(prices, 8)
    ind = get_latest_indicators(df)
    lstm_pred = None
    if not USE_GROQ:
        try:
            from models.trainer import train_model, predict
            model_trained, scaler_trained, _ = train_model(symbol)
            lstm_pred = predict(model_trained, scaler_trained, df)
            lstm_pred = round(float(lstm_pred), 2)
        except Exception:
            pass
    analysis = analyse(df, lstm_pred)
    return analysis, ind, spk

_chat_history = {}
_chat_mode = set()
_live_streams = {}

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
            f"`/aggiungi <ID>` - Aggiungi utente\n"
            f"`/rimuovi <ID>` - Rimuovi utente\n"
            f"`/lista` - Asset disponibili\n"
            f"`/prezzo <nome>` - Prezzo in tempo reale\n"
            f"`/grafico <nome>` - Grafico candlestick\n"
            f"`/live <nome>` - Grafico live 2 min\n"
            f"`/stoplive` - Ferma il live\n"
            f"`/analizza <nome>` - Analisi completa\n"
            f"`/confronta <a1> <a2>` - Confronto due asset\n"
            f"`/backtest <nome>` - Backtest 2 anni\n"
            f"`/top` - I migliori\n"
            f"`/flop` - I peggiori\n"
            f"`/riepilogo` - Riepilogo mercati\n"
            f"`/avvisa <nome> <prezzo>` - Allerta prezzo\n"
            f"`/avvisi` - Lista alert\n"
            f"`/disattiva <id>` - Rimuovi alert\n"
            f"`/notizie` - Ultime notizie\n"
            f"`/notizie <nome>` - Notizie su un asset\n"
            f"`/chat <msg>` - Parla con l'AI\n"
            f"`/status` - Stato del bot"
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
async def prezzo(update, context):
    if not context.args:
        await update.message.reply_text("Usa: `/prezzo oro` o `/prezzo BTC`", parse_mode="Markdown")
        return
    try:
        label, symbol = resolve_symbol(" ".join(context.args))
    except Exception:
        await update.message.reply_text("❌ Asset non trovato.")
        return
    await update.message.reply_text(f"🔍 Cerco {_esc(label)}...")
    try:
        df = fetch_data(symbol, 5)
        price = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else price
        high = float(df["High"].max())
        low = float(df["Low"].min())
        chg = (price / prev - 1) * 100
        arrow = "📈" if chg > 0 else "📉" if chg < 0 else "➡️"
        await update.message.reply_text(
            f"*{_esc(label)}* {arrow} ${price:.2f} ({chg:+.2f}%)\n"
            f"📊 Max: ${high:.2f}  Min: ${low:.2f}",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {str(e)[:150]}")

@authorized
async def status_cmd(update, context):
    uptime = datetime.now() - START_TIME
    d, rem = divmod(int(uptime.total_seconds()), 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    uptime_str = f"{d}g {h}h {m}m"
    lines = [
        "🤖 *Stato Bot*",
        f"📅 Uptime: {uptime_str}",
        f"👥 Utenti autorizzati: {len(AUTHORIZED_USERS)}",
        f"📡 AI: {'Groq' if USE_GROQ else 'Ollama'}",
        f"📊 Asset: {len(ALL_ASSETS)}",
        f"📈 Grafici: {'✅' if CHART_ENABLED else '❌'} (mplfinance)",
        f"🎯 Alert: attivo",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

@authorized
async def top_flop(update, context):
    cmd = update.message.text.split()[0].lower()
    is_top = cmd == "/top"
    msg = await update.message.reply_text("⏳ Calcolo performance...")
    results = []
    for name, sym in ALL_ASSETS:
        try:
            df = fetch_data(sym, 10)
            if len(df) >= 3:
                chg = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1) * 100
                results.append((chg, name))
        except Exception:
            pass
    results.sort(reverse=True)
    items = results[:5] if is_top else results[-5:]
    label = "🏆 *TOP 5*" if is_top else "🍂 *FLOP 5*"
    lines = [f"{label} performance:\n"]
    for chg, name in items:
        arrow = "📈" if chg > 0 else "📉"
        lines.append(f"{arrow} `{_esc(name)}`: {chg:+.2f}%")
    await msg.edit_text("\n".join(lines), parse_mode="Markdown")

@authorized
async def confronta(update, context):
    args = context.args
    if not args or len(args) < 2:
        await update.message.reply_text("Usa: `/confronta oro bitcoin`", parse_mode="Markdown")
        return
    try:
        l1, s1 = resolve_symbol(args[0])
        l2, s2 = resolve_symbol(args[1])
    except Exception:
        await update.message.reply_text("❌ Asset non trovato.")
        return
    await update.message.reply_text(f"⏳ Confronto {_esc(l1)} vs {_esc(l2)}...")
    try:
        df1 = fetch_data(s1, 30)
        df2 = fetch_data(s2, 30)
        r1 = (float(df1["Close"].iloc[-1]) / float(df1["Close"].iloc[0]) - 1) * 100
        r2 = (float(df2["Close"].iloc[-1]) / float(df2["Close"].iloc[0]) - 1) * 100
        v1 = float(df1["Close"].pct_change().std() * (252 ** 0.5) * 100)
        v2 = float(df2["Close"].pct_change().std() * (252 ** 0.5) * 100)
        t1 = float(df1["Close"].iloc[-1])
        t2 = float(df2["Close"].iloc[-1])
        lines = [
            f"📊 *Confronto 30gg:*",
            f"",
            f"*{_esc(l1)}* vs *{_esc(l2)}*",
            f"💰 ${t1:.2f}  vs  ${t2:.2f}",
            f"📈 Rendimento: {r1:+.2f}% {'📈' if r1>r2 else '📉'} {r2:+.2f}%",
            f"📊 Volatilità: {v1:.1f}%  vs  {v2:.1f}%",
            f"🔀 Correlazione: {float(df1['Close'].corr(df2['Close'])):.2f}",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        chart = create_comparison(s1, s2, l1, l2, 60)
        if chart:
            with open(chart, "rb") as f:
                await update.message.reply_photo(f)
            os.unlink(chart)
    except Exception as e:
        await update.message.reply_text(f"❌ Errore confronto: {str(e)[:200]}")

@authorized
async def backtest_cmd(update, context):
    if not context.args:
        await update.message.reply_text("Usa: `/backtest NVDA` o `/backtest oro`", parse_mode="Markdown")
        return
    try:
        label, symbol = resolve_symbol(" ".join(context.args))
    except Exception:
        await update.message.reply_text("❌ Asset non trovato.")
        return
    await update.message.reply_text(f"⏳ Backtest {_esc(label)} su 2 anni...")
    try:
        result = run_backtest(symbol, 2)
        await update.message.reply_text(f"📊 *Backtest {_esc(label)}*\n```\n{format_backtest(result)}\n```", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore backtest: {str(e)[:200]}")

@authorized
async def grafico(update, context):
    if not context.args:
        await update.message.reply_text("Usa: `/grafico oro` o `/grafico BTC`", parse_mode="Markdown")
        return
    if not CHART_ENABLED:
        await update.message.reply_text("❌ Grafici non disponibili (mplfinance non installato).")
        return
    try:
        label, symbol = resolve_symbol(" ".join(context.args))
    except Exception:
        await update.message.reply_text("❌ Asset non trovato.")
        return
    await update.message.reply_text(f"📈 Genero grafico intraday per {_esc(label)}...")
    chart = create_live_chart(symbol, label)
    if not chart:
        await update.message.reply_text("❌ Impossibile generare il grafico intraday.")
        return
    price = get_current_price(symbol) or 0
    with open(chart, "rb") as f:
        await update.message.reply_photo(f, caption=f"📈 {_esc(label)} — ${price:.2f}")
    os.unlink(chart)

@authorized
async def live(update, context):
    if not context.args:
        await update.message.reply_text("Usa: `/live NVDA` o `/live oro`", parse_mode="Markdown")
        return
    if not CHART_ENABLED:
        await update.message.reply_text("❌ Grafici non disponibili.")
        return
    try:
        label, symbol = resolve_symbol(" ".join(context.args))
    except Exception:
        await update.message.reply_text("❌ Asset non trovato.")
        return
    msg = await update.message.reply_text(f"📊 Live {_esc(label)} — `/stoplive` per fermare, si aggiorna ogni 10s", parse_mode="Markdown")
    cid = update.effective_chat.id
    _live_streams[cid] = True
    stopped = False
    photo_msg = None
    for i in range(12):
        if not _live_streams.get(cid):
            stopped = True
            break
        try:
            chart = create_live_chart(symbol, f"{label}")
            if not chart:
                await msg.edit_text(f"❌ Errore grafico #{i+1}")
            price = get_current_price(symbol) or 0
            caption = f"📊 {_esc(label)} — Live {i+1}/12 — ${price:.2f}"
            if photo_msg:
                await photo_msg.delete()
            with open(chart, "rb") as f:
                photo_msg = await context.bot.send_photo(
                    chat_id=cid, photo=f, caption=caption
                )
            try: os.unlink(chart)
            except: pass
            if i < 11:
                await asyncio.sleep(10)
        except Exception as e:
            await update.message.reply_text(f"❌ Live interrotto: {str(e)[:150]}")
            break
    _live_streams.pop(cid, None)
    if photo_msg:
        end = " fermato" if stopped else " terminato"
        await photo_msg.edit_caption(caption=f"📊 {_esc(label)} — Live{end} ✅")
    await msg.delete()

@authorized
async def avvisa(update, context):
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Usa: `/avvisa oro 210`", parse_mode="Markdown")
        return
    target = context.args[-1]
    asset_text = " ".join(context.args[:-1])
    if not target.replace(".", "").isdigit():
        await update.message.reply_text("Il prezzo target non è valido.")
        return
    try:
        label, symbol = resolve_symbol(asset_text)
    except Exception:
        await update.message.reply_text("❌ Asset non trovato.")
        return
    target_price = float(target)
    aid = add_alert(update.effective_user.id, symbol, label, target_price, update.effective_chat.id)
    await update.message.reply_text(
        f"🔔 Alert #{aid} creato!\n{_esc(label)} ti avviserò quando supera ${target_price:.2f}"
    )

@authorized
async def avvisi(update, context):
    alerts = get_user_alerts(update.effective_user.id)
    if not alerts:
        await update.message.reply_text("Nessun alert attivo.")
        return
    lines = ["🔔 *I tuoi alert:*\n"]
    for a in alerts:
        lines.append(f"`#{a['id']}` — {_esc(a['label'])} > ${a['target_price']:.2f}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

@authorized
async def disattiva(update, context):
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usa: `/disattiva <ID>` — `/avvisi` per vedere gli ID", parse_mode="Markdown")
        return
    aid = int(context.args[0])
    if remove_alert(aid, update.effective_user.id):
        await update.message.reply_text(f"🗑️ Alert #{aid} disattivato.")
    else:
        await update.message.reply_text(f"❌ Alert #{aid} non trovato o non tuo.")

@authorized
async def riepilogo(update, context):
    await update.message.reply_text("⏳ Genero riepilogo mercati...")
    sample = list(STOCKS.items())[:5] + list(COMMODITIES.items()) + list(CRYPTO.items())[:2] + list(FOREX_PAIRS.items())[:2]
    results = []
    for name, sym in sample:
        try:
            df = fetch_data(sym, 5)
            if len(df) >= 2:
                chg = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100
                price = float(df["Close"].iloc[-1])
                results.append((name, price, chg))
        except Exception:
            pass
    lines = ["📊 *Riepilogo Mercati:*\n"]
    for name, price, chg in results:
        arrow = "📈" if chg > 0 else "📉"
        lines.append(f"{arrow} `{_esc(name)}`: ${price:.2f} ({chg:+.2f}%)")
    lines.append("\n🧠 *Commento AI...*")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    summary = "\n".join(f"{n}: {c:+.2f}%" for n, _, c in results)
    if USE_GROQ:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_KEY)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"Sei un analista. Riassumi il mercato in 3 righe in italiano basandoti su queste performance:\n{summary}"}],
                temperature=0.5, max_tokens=256,
            )
            await update.message.reply_text(f"🧠 *Analisi:*\n{resp.choices[0].message.content}")
        except Exception as e:
            await update.message.reply_text(f"❌ AI non disponibile: {str(e)[:100]}")
    else:
        await update.message.reply_text("❌ AI non configurata (manca GROQ_API_KEY)")

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
        except Exception:
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
    except Exception:
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
            f"🧩 Confluenza: 🟢{e['votes']['buy']}  🔴{e['votes']['sell']}  ⚪{e['votes']['neutral']} esperti",
            f"📊 Accuratezza backtest: {acc}%" if acc else "",
        ]
        await status_msg.edit_text("\n".join(filter(None, lines)), parse_mode="Markdown")
        news = get_news_text(symbol)
        if CHART_ENABLED:
            chart = create_live_chart(symbol, label)
            if chart:
                price = get_current_price(symbol) or ind['price']
                with open(chart, "rb") as f:
                    await update.message.reply_photo(f, caption=f"📈 {_esc(label)} — ${price:.2f}")
                os.unlink(chart)
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
async def stoplive(update, context):
    cid = update.effective_chat.id
    if cid in _live_streams:
        _live_streams[cid] = False
        await update.message.reply_text("⏹️ Live stream fermato.")
    else:
        await update.message.reply_text("Nessun live attivo in questa chat.")

_nuke_armed = False

async def nukebomb(update, context):
    global _nuke_armed
    uid = update.effective_user.id
    admin = os.environ.get("ADMIN_ID", "")
    if not (admin and admin.isdigit() and uid == int(admin)):
        return
    text = " ".join(context.args).upper() if context.args else ""
    if text == "CONFERMA":
        import shutil
        bot_dir = os.path.dirname(os.path.abspath(__file__))
        for f in ["auth.db", "alerts.db", "token.txt"]:
            p = os.path.join(bot_dir, f)
            if os.path.exists(p):
                os.remove(p)
        marker = os.path.join(bot_dir, ".distrutto")
        with open(marker, "w") as f:
            f.write(f"Nuked at {datetime.now()}")
        _live_streams.clear()
        _chat_history.clear()
        _chat_mode.clear()
        AUTHORIZED_USERS.clear()
        _nuke_armed = False
        await update.message.reply_text("💥 Bot distrutto. Railway Redeploy per riattivare.")
        await asyncio.sleep(1)
        os._exit(0)
    elif text == "ANNULLA":
        _nuke_armed = False
        await update.message.reply_text("❌ Operazione annullata.")
    else:
        _nuke_armed = True
        await update.message.reply_text(
            "💣 *NUKEBOMB ARMATA!*\n\n"
            "Manda `/nukebomb CONFERMA` per distruggere tutto.\n"
            "Manda `/nukebomb ANNULLA` per annullare.",
            parse_mode="Markdown"
        )

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
    app.add_handler(CommandHandler("prezzo", prezzo))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("top", top_flop))
    app.add_handler(CommandHandler("flop", top_flop))
    app.add_handler(CommandHandler("confronta", confronta))
    app.add_handler(CommandHandler("backtest", backtest_cmd))
    app.add_handler(CommandHandler("grafico", grafico))
    app.add_handler(CommandHandler("live", live))
    app.add_handler(CommandHandler("stoplive", stoplive))
    app.add_handler(CommandHandler("nukebomb", nukebomb))
    app.add_handler(CommandHandler("avvisa", avvisa))
    app.add_handler(CommandHandler("avvisi", avvisi))
    app.add_handler(CommandHandler("disattiva", disattiva))
    app.add_handler(CommandHandler("riepilogo", riepilogo))
    app.add_handler(CommandHandler("lista", lista))
    app.add_handler(CommandHandler("notizie", notizie_cmd))
    app.add_handler(CommandHandler("analizza", analizza))
    app.add_handler(CommandHandler("chat", chat_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("esci", stop_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    start_checker(app)
    mode = "Groq-Llama3.3" if USE_GROQ else "Ollama"
    print(f" Telegram Bot avviato su @oracle_fx_bot (AI: {mode})")
    app.run_polling(drop_pending_updates=True)

def run():
    start_bot()

if __name__ == "__main__":
    run()
