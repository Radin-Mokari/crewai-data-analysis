import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path


class SessionDB:
    def __init__(self, path="./results/sessions.db"):
        Path(path).parent.mkdir(exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                dataset_path TEXT NOT NULL,
                prompt TEXT NOT NULL,
                status TEXT DEFAULT 'running',
                result_json TEXT,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT
            )
        """)
        try:
            self.db.execute("ALTER TABLE sessions ADD COLUMN messages_json TEXT DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass
        self.db.commit()

    def create_session(self, dataset_path, prompt):
        sid = str(uuid.uuid4())[:8]
        # Initialize with the user's first prompt as a message
        initial_messages = [{"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()}]
        self.db.execute(
            "INSERT INTO sessions (id, dataset_path, prompt, messages_json) VALUES (?,?,?,?)",
            (sid, dataset_path, prompt, json.dumps(initial_messages)),
        )
        self.db.commit()
        return sid

    def append_message(self, session_id, role, content):
        row = self.db.execute("SELECT messages_json FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row:
            messages = json.loads(row["messages_json"] or "[]")
            messages.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
            self.db.execute(
                "UPDATE sessions SET messages_json=? WHERE id=?",
                (json.dumps(messages), session_id)
            )
            self.db.commit()

    def save_result(self, session_id, result):
        self.db.execute(
            "UPDATE sessions SET status='complete', result_json=?, completed_at=? WHERE id=?",
            (json.dumps(result, default=str), datetime.now().isoformat(), session_id),
        )
        self.db.commit()

    def save_error(self, session_id, error):
        self.db.execute(
            "UPDATE sessions SET status='error', error=?, completed_at=? WHERE id=?",
            (error, datetime.now().isoformat(), session_id),
        )
        self.db.commit()

    def get_sessions(self, limit=20):
        rows = self.db.execute(
            "SELECT id, dataset_path, prompt, status, created_at FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id):
        row = self.db.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        if row:
            result = dict(row)
            if result.get("result_json"):
                result["result"] = json.loads(result["result_json"])
            if result.get("messages_json"):
                result["messages"] = json.loads(result["messages_json"])
            else:
                result["messages"] = []
            return result
        return None
