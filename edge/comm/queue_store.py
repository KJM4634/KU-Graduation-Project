"""
네트워크 단절 시 전송 실패한 이벤트를 로컬 SQLite에 쌓아두고, 재연결 시 순서대로
재전송하기 위한 저장소 (PRD 6장: "로컬 SQLite로 오프라인 큐잉 대응").

같은 파일에 clientEventId 중복 전송 방지를 위한 영속 seq 카운터도 함께 둔다 -
재시작해도 이어서 증가해야 서버가 이전에 받은 이벤트를 다시 새 것으로 오인하지 않는다.
"""
import json
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "offline_queue.sqlite3"


class LocalStore:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_envelopes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seq_counter (
                name TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    # --- 오프라인 큐 (FIFO) ---

    def enqueue(self, payload: dict):
        self._conn.execute(
            "INSERT INTO pending_envelopes (created_at, payload_json) VALUES (datetime('now'), ?)",
            (json.dumps(payload, ensure_ascii=False),),
        )
        self._conn.commit()

    def peek_oldest(self):
        """가장 오래 쌓인 항목 1건을 (id, payload) 형태로 반환. 없으면 None."""
        row = self._conn.execute(
            "SELECT id, payload_json FROM pending_envelopes ORDER BY id ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return row[0], json.loads(row[1])

    def remove(self, item_id: int):
        self._conn.execute("DELETE FROM pending_envelopes WHERE id = ?", (item_id,))
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM pending_envelopes").fetchone()[0]

    # --- seq 카운터 ---

    def next_seq(self, name: str) -> int:
        row = self._conn.execute(
            "SELECT value FROM seq_counter WHERE name = ?", (name,)
        ).fetchone()
        value = (row[0] if row else 0) + 1
        self._conn.execute(
            "INSERT INTO seq_counter (name, value) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value = excluded.value",
            (name, value),
        )
        self._conn.commit()
        return value

    def close(self):
        self._conn.close()
