# -*- coding: utf-8 -*-
"""SqliteVecRagIndex：基于 SQLite db 的对话轮次向量索引实现（默认，自包含无需向量扩展）。

- 每轮切片以 JSON 数组向量存入 rag_turns 表（DDL 统一在 sql/create_table.sql）
- 检索：余弦相似度 + 近因加权（recent_boost），限定同一 session_id
- search 返回含 turn_idx，供上层按最近窗口去重
"""
import json
import math
import os
import sqlite3
import time
from typing import Callable, List, Tuple

from component.logger import get_logger
from component.rag_index import RagIndex

_log = get_logger("rag_sqlite_vec")

# 项目根目录（src/component/rag_sqlite_vec.py -> 上溯三级）
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


def _cosine(a: List[float], b: List[float]) -> float:
    """余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


class SqliteVecRagIndex(RagIndex):
    """基于现有 SQLite db 的向量表实现（默认，自包含无需向量扩展）。"""

    def __init__(
        self,
        path: str,
        embed_fn: Callable[[str], List[float]],
        recent_boost: float = 0.3,
        now: Callable[[], float] = time.time,
    ):
        self.path = path
        self.embed_fn = embed_fn
        self.recent_boost = recent_boost
        self._now = now
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_load_ddl())
        self._conn.commit()

    def add_turn(self, session_id: str, turn_idx: int, role: str, text: str) -> None:
        vec = self.embed_fn(text)
        with self._conn:
            self._conn.execute(
                "INSERT INTO rag_turns(session_id, turn_idx, role, text, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, turn_idx, role, text, json.dumps(vec), self._now()),
            )
        _log.info("rag.add_turn session=%s turn_idx=%d role=%s len=%d",
                  session_id, turn_idx, role, len(text))

    def search(
        self, query: str, session_id: str, top_k: int = 4
    ) -> List[Tuple[int, str, str, float, float]]:
        qv = self.embed_fn(query)
        rows = self._conn.execute(
            "SELECT turn_idx, role, text, embedding, created_at FROM rag_turns WHERE session_id=?",
            (session_id,),
        ).fetchall()
        if not rows:
            return []

        times = [r["created_at"] for r in rows]
        tmin = min(times)
        span = (max(times) - tmin) or 1.0

        scored = []
        for r in rows:
            sim = _cosine(qv, json.loads(r["embedding"]))
            recency = (r["created_at"] - tmin) / span      # 0..1，越大越新
            score = sim + self.recent_boost * recency
            scored.append((r["turn_idx"], r["role"], r["text"], score, sim))
        scored.sort(key=lambda x: x[3], reverse=True)
        top = scored[:top_k]
        _log.info(
            "rag.search session=%s top_k=%d hits=%d",
            session_id, top_k, len(top),
        )
        return top

    def max_similarity(self, query: str, session_id: str) -> float:
        """返回查询与该会话历史轮次的最大余弦相似度（纯相似度，不含近因加权）。"""
        qv = self.embed_fn(query)
        rows = self._conn.execute(
            "SELECT embedding FROM rag_turns WHERE session_id=?", (session_id,)
        ).fetchall()
        best = 0.0
        for r in rows:
            sim = _cosine(qv, json.loads(r["embedding"]))
            if sim > best:
                best = sim
        return best

    def close(self) -> None:
        self._conn.close()
