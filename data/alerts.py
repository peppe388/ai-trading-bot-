import sqlite3, os, threading, time
from datetime import datetime
from data.market import get_current_price

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bot', 'alerts.db')

def _db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, symbol TEXT, label TEXT, target_price REAL, chat_id INTEGER)")
    conn.row_factory = sqlite3.Row
    return conn

def add_alert(user_id, symbol, label, target_price, chat_id):
    conn = _db()
    conn.execute("INSERT INTO alerts (user_id, symbol, label, target_price, chat_id) VALUES (?, ?, ?, ?, ?)",
                 (user_id, symbol, label, target_price, chat_id))
    conn.commit()
    aid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return aid

def remove_alert(alert_id, user_id):
    conn = _db()
    c = conn.execute("DELETE FROM alerts WHERE id = ? AND user_id = ?", (alert_id, user_id))
    conn.commit()
    r = c.rowcount
    conn.close()
    return r > 0

def get_user_alerts(user_id):
    conn = _db()
    rows = conn.execute("SELECT id, symbol, label, target_price FROM alerts WHERE user_id = ?", (user_id,)).fetchall()
    conn.close()
    return rows

def get_all_alerts():
    conn = _db()
    rows = conn.execute("SELECT id, user_id, symbol, label, target_price, chat_id FROM alerts").fetchall()
    conn.close()
    return rows

def start_checker(app):
    async def _notify(chat_id, text):
        try:
            await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
        except Exception:
            pass
    def _check():
        while True:
            try:
                alerts = get_all_alerts()
                for row in alerts:
                    try:
                        price = get_current_price(row['symbol'])
                        if price and price >= row['target_price']:
                            text = f" Alert attivato!\n{row['label']} ha raggiunto ${price:.2f} (target: ${row['target_price']:.2f})"
                            app.create_task(_notify(row['chat_id'], text))
                            remove_alert(row['id'], row['user_id'])
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(300)
    t = threading.Thread(target=_check, daemon=True)
    t.start()
