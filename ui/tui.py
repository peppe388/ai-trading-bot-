import os, sys, time, threading
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import config
from data.market import fetch_data, resolve_symbol, get_asset_name, get_news_text
from data.indicators import add_indicators
from models.trainer import train_model, predict
from advisor.analyser import analyse, record_prediction
from advisor.reporter import get_advice, chat
from advisor.chart import generate_chart
from advisor.backtest import run_backtest, format_backtest
from data.market import get_news
from advisor.vision import analyse_chart
from advisor.alerter import Alerter
from data.streamer import PriceStreamer

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.markdown import Markdown
from rich.align import Align

console = Console()

ALL_ASSETS = (list(config.FOREX_PAIRS.items()) + list(config.STOCKS.items()) +
              list(config.COMMODITIES.items()) + list(config.CRYPTO.items()))

SPARK_CHARS = ["_", "\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"]


def sparkline(data, width=10):
    if not data or len(data) < 2:
        return ""
    step = max(1, len(data) // width)
    sampled = data[::step][:width]
    mn, mx = min(sampled), max(sampled)
    rng = mx - mn if mx != mn else 1
    return "".join(SPARK_CHARS[min(int((v - mn) / rng * (len(SPARK_CHARS) - 1)), len(SPARK_CHARS) - 1)] for v in sampled)


def _color(sig):
    return "green" if sig == "BUY" else "red" if sig == "SELL" else "yellow"


def _icon(sig):
    return "^" if sig == "BUY" else "v" if sig == "SELL" else "*"


class TradingTUI:
    def __init__(self):
        self.tab = "dashboard"
        self.chat_history = []
        self.asset_cache = {}
        self.ollama_ok = False
        self.ollama_checked = False
        self.scan_done = False
        self.scan_progress = ""
        self.alerter = Alerter()
        self.streamer = PriceStreamer()
        self.backtest_result = None
        self.alert_banner = ""

    def startup(self):
        self._print_loading("Avvio Trading Advisor Pro...")
        self._check_ollama_bg()
        self._scan_all_bg()
        self.streamer.start(interval=30)
        self.alerter.start(interval=300)
        self.streamer.on_price(self._on_price_update)

    def _on_price_update(self, symbol, info):
        if symbol in self.asset_cache:
            self.asset_cache[symbol]["price"] = info["price"]
            self.asset_cache[symbol]["change"] = info["change"]

    def _print_loading(self, msg):
        os.system("cls" if os.name == "nt" else "clear")
        console.print(Panel(Align.center(Text("AI TRADING ADVISOR PRO", style="bold cyan")),
                            style="cyan", box=box.HEAVY))
        console.print(Panel(Align.center(Text(f"  {msg}  ", style="cyan italic")),
                            box=box.SIMPLE, padding=(1, 0)))

    def _check_ollama_bg(self):
        def check():
            try:
                import requests
                requests.get("http://localhost:11434/api/tags", timeout=2)
                self.ollama_ok = True
            except:
                self.ollama_ok = False
            self.ollama_checked = True
        threading.Thread(target=check, daemon=True).start()

    def _scan_all_bg(self):
        def scan():
            from advisor.analyser import _ta_ensemble, _volume_score, _multi_tf_score
            for i, (label, symbol) in enumerate(ALL_ASSETS):
                self.scan_progress = f"Scarico {label} ({i+1}/{len(ALL_ASSETS)})..."
                try:
                    df = fetch_data(symbol, 90)
                    df = add_indicators(df)
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) >= 2 else latest
                    change = (latest["Close"] / prev["Close"] - 1) * 100
                    from data.indicators import get_latest_indicators
                    ind = get_latest_indicators(df)
                    prices = df["Close"].values[-30:].tolist()

                    # Quick signal without LSTM (TA 35% + Volume 15% + MultiTF 15%)
                    s_ta = _ta_ensemble(ind)
                    s_vol = _volume_score(df)
                    s_tf = _multi_tf_score(df)
                    raw = s_ta * 0.35 + s_vol * 0.15 + s_tf * 0.15
                    norm = raw / 0.65 if 0.65 > 0 else 0

                    if norm > 0.2:
                        sig, c = "BUY", min(90, int(abs(norm) * 100))
                    elif norm < -0.2:
                        sig, c = "SELL", min(90, int(abs(norm) * 100))
                    else:
                        sig, c = "HOLD", max(15, int(abs(norm) * 100))

                    self.asset_cache[symbol] = {
                        "label": label, "price": round(float(latest["Close"]), 2),
                        "change": round(change, 2), "rsi": ind["rsi"],
                        "signal": sig, "conf": c,
                        "spark": sparkline(prices, 8), "trend": ind["trend"],
                    }
                except:
                    self.asset_cache[symbol] = {
                        "label": label, "price": None, "change": None,
                        "rsi": None, "signal": None, "conf": None,
                        "spark": "", "trend": "",
                    }
            self.scan_done = True
        threading.Thread(target=scan, daemon=True).start()

    def analyse_asset(self, msg):
        yf_symbol = None
        label = None
        ml = msg.lower()
        for name, sym in ALL_ASSETS:
            nl = name.lower()
            if ml in nl or nl in ml:
                yf_symbol, label = sym, name
                break
        if not yf_symbol:
            for common_name, yf_sym in config.COMMON_NAMES.items():
                if ml in common_name or common_name in ml or yf_sym.lower() in ml:
                    yf_symbol = yf_sym
                    for name, sym in ALL_ASSETS:
                        if sym == yf_sym:
                            label = name
                            break
                    if not label:
                        label = common_name.upper()
                    break
        if not yf_symbol:
            return None, None, None, ""

        try:
            df = fetch_data(yf_symbol, config.LOOKBACK_DAYS)
            df = add_indicators(df)
            model, scaler, _ = train_model(yf_symbol)
            lstm_pred = predict(model, scaler, df)
            analysis = analyse(df, lstm_pred)
            later = 0
            if len(df) >= 2:
                later = float(df["Close"].tail(2).diff().iloc[-1] / df["Close"].iloc[-2] * 100)
            record_prediction(yf_symbol, lstm_pred, later, (lstm_pred > 0) == (later > 0))
            i = analysis["indicators"]
            e = analysis["ensemble"]
            sig = analysis["signal"]
            conf = analysis["confidence"]
            news_text = get_news_text(yf_symbol)
            try:
                ollama_reply = get_advice(label, yf_symbol, analysis, news_text)
            except:
                ollama_reply = None
            prices = df["Close"].values[-30:].tolist() if len(df) >= 30 else df["Close"].values.tolist()
            self.asset_cache[yf_symbol] = {
                "label": label, "price": round(float(i["price"]), 2),
                "change": round(float(i["price_change_pct"]), 2), "rsi": i["rsi"],
                "signal": sig, "conf": conf, "trend": i["trend"],
                "spark": sparkline(prices, 8),
                "ai_pred": analysis["lstm_prediction_pct"], "macd": i["macd_status"],
                "stoch": i["stoch"], "obv": i["obv_trend"], "consensus": e["consensus"],
                "stop_loss": analysis["stop_loss"], "target": analysis["target"],
                "accuracy": analysis["backtest_accuracy"], "news": news_text,
                "ollama": ollama_reply,
            }
            return analysis, label, yf_symbol, news_text
        except Exception as e:
            return {"error": str(e)}, label, yf_symbol, ""

    def _make_header(self):
        o = "Ollama: OK" if self.ollama_ok else "Ollama: OFF" if self.ollama_checked else "Ollama: ..."
        lines = [Align.center(Text("AI TRADING ADVISOR PRO", style="bold cyan"))]
        if self.alert_banner:
            lines.append(Text(f"  {self.alert_banner}  ", style="bold yellow"))
        if self.alerter.alerts:
            a = self.alerter.alerts[0]
            lines.append(Text(f"  Alert: {a['label']} {a['from']} -> {a['to']} @ ${a.get('price',0)}", style="bold yellow"))
        return Panel(Group(*lines), style="cyan", box=box.HEAVY, padding=(0, 0),
                     subtitle=Text(f"  {o}  ", style="italic"))

    def _make_tabs(self):
        t = []
        for i, (k, lb) in enumerate([
            ("dashboard", "Dashboard"), ("chat", "Chat"), ("assets", "Assets"),
            ("backtest", "Backtest"), ("notizie", "Notizie")
        ], 1):
            s = "> " if self.tab == k else "  "
            st = "bold cyan" if self.tab == k else "dim white"
            t.append(f"[{st}]{s}[{i}] {lb}[/]")
        return Panel("   ".join(t), style="cyan", box=box.SQUARE, padding=(0, 1))

    def _make_dashboard(self):
        table = Table(box=box.SIMPLE, header_style="bold cyan")
        table.add_column("Asset", style="bold white", no_wrap=True)
        table.add_column("Prezzo", justify="right")
        table.add_column("Var", justify="right")
        table.add_column("Trend", no_wrap=True)
        table.add_column("RSI")
        table.add_column("Segnale", justify="center")
        table.add_column("Conf", justify="right")

        if not self.scan_done:
            table.add_row("[italic dim]Caricamento...[/]", "", "", "", "", "[italic dim]...[/]", "")

        for symbol, info in sorted(self.asset_cache.items(),
                                   key=lambda x: (x[1].get("change") or 0) if self.tab == "dashboard" else 0,
                                   reverse=True):
            if info.get("error"):
                continue
            ch = info.get("change")
            sig = info.get("signal")
            table.add_row(
                info["label"],
                f"[bold]{info['price']:.2f}[/]" if info.get("price") else "[dim]?[/]",
                f"[green]{ch:+.2f}%" if ch and ch >= 0 else f"[red]{ch:+.2f}%" if ch else "[dim]?[/]",
                f"[cyan]{info.get('spark', '')}[/]",
                str(info.get("rsi", "?")) if info.get("rsi") else "?",
                f"[{_color(sig)}]{_icon(sig)} {sig}[/]" if sig else "[dim]--[/]",
                f"{info.get('conf', 0)}%" if info.get("conf") else "",
            )
        return Panel(Group(Text("DASHBOARD  |  Asset Monitorati", style="bold white"), table,
                           Text(f"\n  {self.scan_progress}", style="dim italic") if not self.scan_done else Text("")),
                     box=box.HEAVY, border_style="cyan")

    def _make_chat_view(self):
        el = [Text("CHAT — Parla con l'AI", style="bold white")]
        if not self.chat_history:
            el.append(Text("  Nessun messaggio. Scrivi qualcosa.", style="dim italic"))
        for e in self.chat_history[-15:]:
            if e["role"] == "user":
                el.append(Panel(Text(e["msg"], style="bold cyan"), box=box.SIMPLE, padding=(0, 1)))
            else:
                el.append(Panel(Markdown(e["msg"]), box=box.SIMPLE, padding=(0, 1), border_style="green"))
        return Panel(Group(*el), box=box.HEAVY, border_style="cyan")

    def _make_assets_view(self):
        table = Table(box=box.SIMPLE, header_style="bold cyan")
        table.add_column("Categoria", style="bold white")
        table.add_column("Asset", style="bold white")
        table.add_column("Simbolo")
        table.add_column("Prezzo", justify="right")
        table.add_column("Stato")
        for cat, pairs in [("Forex", config.FOREX_PAIRS), ("Azioni", config.STOCKS),
                           ("Commodities", config.COMMODITIES), ("Crypto", config.CRYPTO)]:
            for lb, sym in pairs.items():
                info = self.asset_cache.get(sym, {})
                sig = info.get("signal")
                table.add_row(cat, lb, sym,
                              f"[bold]{info['price']:.2f}[/]" if info.get("price") else "[dim]?[/]",
                              f"[{_color(sig)}]{sig}[/]" if sig else "[dim]--[/]")
        return Panel(Group(Text("ASSETS — Elenco completo", style="bold white"), table),
                     box=box.HEAVY, border_style="cyan")

    def _make_backtest_view(self):
        if not self.backtest_result:
            return Panel(Text("Nessun backtest eseguito. Scrivi il nome di un asset per analizzarlo.", style="dim italic"),
                         box=box.HEAVY, border_style="cyan")
        r = self.backtest_result
        t = Table(box=box.HEAVY, show_header=False, padding=(0, 2))
        t.add_column("Metrica", style="bold cyan")
        t.add_column("Valore")
        ret_color = "green" if r["total_return"] >= 0 else "red"
        t.add_row("Simbolo", r["symbol"])
        t.add_row("Rendimento", f"[bold {ret_color}]{r['total_return']:+.2f}%[/]")
        t.add_row("Operazioni vincenti", f"{r['win_rate']}% ({r['num_trades']} trades)")
        t.add_row("Max Drawdown", f"[red]{r['max_drawdown']:.2f}%[/]")
        t.add_row("Sharpe Ratio", f"{'[green]' if r['sharpe_ratio'] >= 1 else '[yellow]'}{r['sharpe_ratio']}[/]")
        t.add_row("Capitale finale", f"${r['final_capital']:.2f}")
        return Panel(t, box=box.HEAVY, border_style="cyan", title="  BACKTEST  ", title_align="center")

    def _make_news_view(self):
        news_items = get_news("SPY", 12) or []
        t = Table(box=box.HEAVY, show_header=False, padding=(0, 1), border_style="cyan")
        for cat_name, sym in [("📈 Mercati", "SPY"), ("₿ Crypto", "BTC-USD"), ("🏅 Commodities", "GLD")]:
            items = get_news(sym, 4) or []
            if items:
                t.add_row(f"[bold cyan]{cat_name}[/] ", "")
                for n in items:
                    t.add_row(f"  {n['title']}", f"[dim]{n.get('publisher', '')}[/]")
        if not t.row_count:
            return Panel("Nessuna notizia disponibile.", box=box.HEAVY, border_style="cyan")
        return Panel(t, box=box.HEAVY, border_style="cyan")

    def _render_tab(self, tab):
        self.tab = tab
        os.system("cls" if os.name == "nt" else "clear")
        console.print(self._make_header())
        console.print(self._make_tabs())
        if tab == "dashboard":
            console.print(self._make_dashboard())
        elif tab == "chat":
            console.print(self._make_chat_view())
        elif tab == "assets":
            console.print(self._make_assets_view())
        elif tab == "backtest":
            console.print(self._make_backtest_view())
        elif tab == "notizie":
            console.print(self._make_news_view())
        console.print(self._make_footer())

    def _make_footer(self):
        return Panel(
            "[bold cyan][1][/] Dashboard  [bold cyan][2][/] Chat  [bold cyan][3][/] Assets  "
            "[bold cyan][4][/] Backtest  [bold cyan][5][/] Notizie  "
            "[bold yellow]q[/]=quit  [bold yellow]h[/]=help",
            style="dim white", box=box.SQUARE, padding=(0, 1),
        )

    def run_analysis_and_show(self, msg):
        os.system("cls" if os.name == "nt" else "clear")
        console.print(self._make_header())
        with console.status("[bold cyan]Analizzo... caricamento dati e LSTM...", spinner="dots"):
            result, label, yf_symbol, news_text = self.analyse_asset(msg)
        if result is None or "error" in (result or {}):
            err = result.get("error", "") if isinstance(result, dict) else ""
            console.print(Panel(Text(
                f"Asset non riconosciuto: {err}" if err else "Asset non riconosciuto. Prova: EUR/USD, oro, TSLA, BTC",
                style="bold yellow"), box=box.HEAVY, border_style="yellow"))
            console.print(self._make_footer())
            return
        info = self.asset_cache.get(yf_symbol, {})
        i = result["indicators"]
        sig = result["signal"]
        conf = result["confidence"]
        inv = "INVESTI" if sig == "BUY" and conf >= 50 else "NON INVESTIRE"
        t = Table(box=box.HEAVY, show_header=False, padding=(0, 2))
        t.add_column("Metrica", style="bold cyan")
        t.add_column("Valore")
        ld = info.get("label", label or "?")
        t.add_row("Asset", ld)
        t.add_row("Prezzo", f"[bold]${i['price']}[/] ({i['price_change_pct']:+.2f}%)")
        t.add_row("RSI", f"{i['rsi']} ({i['rsi_signal']})")
        t.add_row("MACD", i["macd_status"])
        t.add_row("Stocastico", str(i["stoch"]))
        t.add_row("OBV", i["obv_trend"])
        t.add_row("Trend AI", i["trend"])
        t.add_row("Predizione LSTM", f"{result['lstm_prediction_pct']:+.2f}%")
        t.add_row("Ensemble", f"{result['ensemble']['consensus']}/4")
        t.add_row("Stop Loss", f"${result['stop_loss']}")
        t.add_row("Target", f"${result['target']}")
        t.add_row("Backtest Acc", f"{result.get('backtest_accuracy', '?')}%")
        t.add_row("VERDETTO", f"[{'green' if inv == 'INVESTI' else 'red'}][bold]{inv}[/bold][/] (conf: {conf}%)")
        console.print(Panel(t, box=box.HEAVY, border_style="cyan", title=f"  {ld}  ", title_align="center"))
        if news_text:
            console.print(Panel(Text(news_text, style="cyan"), box=box.SIMPLE, border_style="cyan",
                                padding=(0, 1), title="  Notizie  ", title_align="left"))
        chart_path = None
        try:
            dfc = fetch_data(yf_symbol, config.LOOKBACK_DAYS)
            chart_path = generate_chart(dfc, ld, result["signal"])
            console.print(f"  Grafico: [bold cyan]{chart_path}[/]")
            console.print(f"  [dim]Avvia 'ollama pull llama3.2-vision' per analisi visiva[/]")
        except:
            pass
        if chart_path:
            try:
                from advisor.vision import analyse_chart as vision_analyse
                with console.status("[bold cyan]Analisi visiva con AI...", spinner="dots"):
                    vision_reply = vision_analyse(chart_path)
                if vision_reply:
                    console.print(Panel(Markdown(f"**Analisi visiva:** {vision_reply}"),
                                        box=box.SIMPLE, border_style="magenta", padding=(0, 1)))
            except:
                pass
        oi = info.get("ollama")
        if oi:
            console.print(Panel(Markdown(oi), box=box.SIMPLE, border_style="green", padding=(0, 1)))
        # Run backtest in background
        def bt():
            try:
                self.backtest_result = run_backtest(yf_symbol, years=2)
            except:
                pass
        threading.Thread(target=bt, daemon=True).start()
        console.print(self._make_footer())

    def run_chat_and_show(self, msg):
        self.chat_history.append({"role": "user", "msg": msg})
        os.system("cls" if os.name == "nt" else "clear")
        console.print(self._make_header())
        self.tab = "chat"
        with console.status("[bold cyan]Ollama sta pensando...", spinner="dots"):
            reply = chat(msg, self.chat_history)
        if reply:
            self.chat_history.append({"role": "assistant", "msg": reply})
        else:
            self.chat_history.append({"role": "assistant", "msg": "Non ho ricevuto risposta."})
        console.print(self._make_chat_view())
        console.print(self._make_footer())

    def run(self):
        self.startup()
        time.sleep(1.5)
        self._render_tab("dashboard")
        while True:
            try:
                cmd = input(f"\033[96m> \033[0m").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold yellow]Arrivederci![/]")
                break
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit", "esci", "q"):
                console.print("[bold yellow]Arrivederci![/]")
                break
            if cmd == "1":
                self._render_tab("dashboard")
                continue
            if cmd == "2":
                self._render_tab("chat")
                continue
            if cmd == "3":
                self._render_tab("assets")
                continue
            if cmd == "4":
                self._render_tab("backtest")
                continue
            if cmd == "5":
                self._render_tab("notizie")
                continue
            if cmd == "h":
                os.system("cls" if os.name == "nt" else "clear")
                console.print(self._make_header())
                console.print(Panel(
                    "[bold cyan]Comandi:[/]\n"
                    "  [bold]1[/] Dashboard    [bold]4[/] Backtest\n"
                    "  [bold]2[/] Chat         [bold]5[/] Notizie\n"
                    "  [bold]3[/] Assets       [bold]q[/] Esci\n\n"
                    "Oppure scrivi: EUR/USD, oro, TSLA per analisi\n"
                    "  o qualsiasi domanda per parlare con l'AI",
                    box=box.HEAVY, border_style="cyan"))
                console.print(self._make_footer())
                continue
            from cli import is_asset_query
            if is_asset_query(cmd):
                self.run_analysis_and_show(cmd)
            else:
                self.run_chat_and_show(cmd)


def main():
    TradingTUI().run()


if __name__ == "__main__":
    main()
