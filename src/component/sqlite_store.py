# -*- coding: utf-8 -*-
"""SqliteContextStore：SQLite 后端实现（默认）。

建表 DDL 统一存放在项目根目录 sql/create_table.sql，
本模块初始化时加载执行。
"""
import os
import sqlite3
import uuid
from typing import Dict, List, Optional

from component.context_store import ContextStore, StoredMessage, _now
from component.logger import get_logger

_log = get_logger("sqlite_store")

# 项目根目录（src/component/sqlite_store.py -> 上溯三级）
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# 兼容源码运行与已安装包运行（安装到 site-packages 时 __file__ 无 sql/ 目录）
_DDL_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "sql", "create_table.sql"),
    os.path.join(os.getcwd(), "sql", "create_table.sql"),
]


def _load_ddl() -> str:
    for p in _DDL_CANDIDATES:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(
        f"建表 DDL 不存在，尝试路径: {_DDL_CANDIDATES}（应在项目根目录 sql/create_table.sql）"
    )


class SqliteContextStore(ContextStore):
    """SQLite 后端实现（默认）。"""

    def __init__(self, path: str = "data/context_cache.db"):
        self.path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_load_ddl())
        self._conn.commit()

    def create_session(self, meta: Optional[str] = None) -> str:
        sid = str(uuid.uuid4())
        now = _now()
        with self._conn:
            self._conn.execute(
                "INSERT INTO sessions(id, created_at, updated_at, meta) VALUES (?, ?, ?, ?)",
                (sid, now, now, meta),
            )
        return sid

    def append(self, session_id: str, message: StoredMessage) -> None:
        now = _now()
        with self._conn:
            self._conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, message.role, message.to_json(), now),
            )
            # updated_at 仅写入时更新
            self._conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?", (
                    now, session_id)
            )

    def get_messages(self, session_id: str) -> List[StoredMessage]:
        rows = self._conn.execute(
            "SELECT content, created_at FROM messages WHERE session_id=? "
            "ORDER BY created_at ASC, id ASC",
            (session_id,),
        ).fetchall()
        return [StoredMessage.from_json(r["content"], r["created_at"]) for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM sessions WHERE id=?", (session_id,))

    def get_summary(self, session_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT summary FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        return row["summary"] if row else None

    def set_summary(self, session_id: str, summary: Optional[str]) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE sessions SET summary=? WHERE id=?", (
                    summary, session_id)
            )
        _log.info("set_summary session=%s summary_len=%d",
                  session_id, len(summary) if summary else 0)

    def prune(self, session_id: str, keep: int) -> None:
        cnt = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()[0]
        excess = cnt - keep
        if excess > 0:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM messages WHERE session_id=? AND id IN ("
                    " SELECT id FROM messages WHERE session_id=? "
                    " ORDER BY created_at ASC, id ASC LIMIT ?)",
                    (session_id, session_id, excess),
                )
        _log.info("prune session=%s before=%d keep=%d removed=%d",
                  session_id, cnt, keep, max(excess, 0))

    def enforce_caps(self, max_sessions: int, max_messages_per_session: int) -> None:
        with self._conn:
            # 会话数上限：按 updated_at 最旧 LRU 淘汰（外键级联删除消息）
            n = self._conn.execute(
                "SELECT COUNT(*) FROM sessions").fetchone()[0]
            excess_sessions = n - max_sessions
            if excess_sessions > 0:
                self._conn.execute(
                    "DELETE FROM sessions WHERE id IN ("
                    " SELECT id FROM sessions ORDER BY updated_at ASC, created_at ASC LIMIT ?)",
                    (excess_sessions,),
                )
            # 单会话消息数上限：删除最旧消息，保留最新 max_messages_per_session 条
            for (sid,) in self._conn.execute("SELECT id FROM sessions").fetchall():
                cnt = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id=?", (sid,)
                ).fetchone()[0]
                excess = cnt - max_messages_per_session
                if excess > 0:
                    self._conn.execute(
                        "DELETE FROM messages WHERE session_id=? AND id IN ("
                        " SELECT id FROM messages WHERE session_id=? "
                        " ORDER BY created_at ASC, id ASC LIMIT ?)",
                        (sid, sid, excess),
                    )

    def clear_all(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM messages")
            self._conn.execute("DELETE FROM sessions")

    def list_sessions(self) -> Dict[str, float]:
        rows = self._conn.execute(
            "SELECT id, updated_at FROM sessions").fetchall()
        return {r["id"]: r["updated_at"] for r in rows}

    def close(self) -> None:
        self._conn.close()
