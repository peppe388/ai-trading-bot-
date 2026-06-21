#!/usr/bin/env python3
"""AI Trading Advisor Pro - Terminal UI
Usage: python run_tui.py
"""
import os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from ui.tui import main

if __name__ == "__main__":
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        from bot.telegram_bot import start_bot
        t = threading.Thread(target=start_bot, daemon=True)
        t.start()
        print(" Telegram Bot avviato in background")
    main()
