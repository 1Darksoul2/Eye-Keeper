"""
EyeKeeper — database.py
SQLite user management: create tables, CRUD for users.
"""

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

DB_PATH  = Path(__file__).parent / "eyekeeper.db"
_db_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────
# Connection helper
# ──────────────────────────────────────────────────────────────────
def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


# ──────────────────────────────────────────────────────────────────
# Schema init (call once at startup)
# ──────────────────────────────────────────────────────────────────
def init_db() -> None:
    with _db_lock, _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    NOT NULL,
                plan          TEXT    NOT NULL DEFAULT 'free',
                credits       INTEGER NOT NULL DEFAULT 0
            )
        """)
        c.commit()
    print("[DB] Initialised — eyekeeper.db")


# ──────────────────────────────────────────────────────────────────
# Read helpers
# ──────────────────────────────────────────────────────────────────
def get_user_by_email(email: str) -> dict | None:
    with _db_lock, _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with _db_lock, _conn() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


# ──────────────────────────────────────────────────────────────────
# Write helpers
# ──────────────────────────────────────────────────────────────────
def create_user(email: str, password_hash: str) -> int:
    """Insert a new user and return their id."""
    with _db_lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO users (email, password_hash, created_at, plan, credits) "
            "VALUES (?, ?, ?, 'free', 0)",
            (email, password_hash, datetime.utcnow().isoformat()),
        )
        c.commit()
    return cur.lastrowid


def update_user_plan(user_id: int, plan: str) -> None:
    with _db_lock, _conn() as c:
        c.execute("UPDATE users SET plan = ? WHERE id = ?", (plan, user_id))
        c.commit()