import os, subprocess, time
import requests

OLLAMA_API = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

FINANCE_SYSTEM = """Sei un analista finanziario. Rispondi in italiano.
Inizia con INVESTI o NON INVESTIRE. Poi spiega brevemente perche'.
Usa le notizie recenti per motivare la tua analisi se disponibili.
Termina con: DISCLAIMER: progetto educativo. Non ripetere lo stesso messaggio ogni volta."""

CHAT_SYSTEM = """Sei un assistente AI utile, esperto in finanza ed economia.
Rispondi in modo diretto e conciso, senza presentarti ogni volta.
Parla italiano. Sii naturale e conversazionale."""


def _ensure_ollama():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except requests.exceptions.ConnectionError:
        pass
    ollama_path = os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"
    )
    if os.path.exists(ollama_path):
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen([ollama_path, "serve"], **kwargs)
        for _ in range(15):
            time.sleep(2)
            try:
                requests.get("http://localhost:11434/api/tags", timeout=2)
                return True
            except requests.exceptions.ConnectionError:
                continue
    return False


def query_ollama(messages, system, timeout=30):
    if not _ensure_ollama():
        return None
    try:
        resp = requests.post(
            OLLAMA_API,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    *messages,
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 512,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ReadTimeout:
        return None
    except Exception:
        return None


def get_advice(asset_name, symbol, analysis, news_text=""):
    i = analysis["indicators"]
    signal = analysis["signal"]
    conf = analysis["confidence"]
    verdict = "INVESTI" if signal == "BUY" and conf >= 50 else "NON INVESTIRE"

    prompt = (
        f"Analisi {asset_name}: segnale {verdict} "
        f"(segnale={signal}, conf={conf}%, "
        f"prezzo=${i['price']}, RSI={i['rsi']}, MACD={i['macd_status']}, "
        f"trend={i['trend']}, AI pred={analysis['lstm_prediction_pct']}%, "
        f"ensemble={analysis['ensemble']['consensus']}/4). "
        f"{news_text} "
        f"Conferma o meno '{verdict}' e spiega perche'."
    )
    reply = query_ollama([{"role": "user", "content": prompt}], FINANCE_SYSTEM, timeout=15)
    if reply:
        return reply
    return f"{verdict} (conf: {conf}%) | {asset_name} a ${i['price']}, trend {i['trend']}. DISCLAIMER: progetto educativo"


def chat(message, history=None):
    messages = []
    if history:
        for h in history[-6:]:
            messages.append({"role": h["role"], "content": h["msg"]})
    messages.append({"role": "user", "content": message})

    reply = query_ollama(messages, CHAT_SYSTEM, timeout=60)
    if reply:
        return reply
    return "Non riesco a contattare Ollama. Verifica che sia in esecuzione."
