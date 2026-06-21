import os, base64, subprocess
import requests

VISION_MODEL = "llama3.2-vision"


def _model_available():
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        return VISION_MODEL in r.stdout
    except Exception:
        return False


def analyse_chart(chart_path):
    if not _model_available():
        return None
    try:
        with open(chart_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": VISION_MODEL,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Analizza questo grafico finanziario. "
                        "Descrivi trend (rialzista/ribassista/laterale), "
                        "pattern (testa e spalle, doppio massimo, bandiera, triangolo, ecc), "
                        "supporti e resistenze visibili. "
                        "Rispondi in italiano, max 4 righe."
                    ),
                    "images": [b64],
                }],
                "stream": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception:
        return None
