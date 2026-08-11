# -*- coding: utf-8 -*-
"""InMemoryContextStore：进程内存后端实现（backend="memory"）。"""
import uuid
from typing import Dict, List, Optional

from component.context_store import ContextStore, StoredMessage, _now
from component.logger import get_logger

_log = get_logger("memory_store")


class InMemoryContextStore(ContextStore):
    """进程内存后端实现（backend="memory"）。"""

    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._messages: Dict[str, List[StoredMessage]] = {}

    def create_session(self, meta: Optional[str] = None) -> str:
        sid = str(uuid.uuid4())
        now = _now()
        self._sessions[sid] = {
            "created_at": now, "updated_at": now, "meta": meta, "summary": None,
        }
        self._messages[sid] = []
        return sid

    def append(self, session_id: str, message: StoredMessage) -> None:
        message.created_at = _now()
        self._messages.setdefault(session_id, []).append(message)
        if session_id in self._sessions:
            self._sessions[session_id]["updated_at"] = message.created_at

    def get_messages(self, session_id: str) -> List[StoredMessage]:
        return list(self._messages.get(session_id, []))

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._messages.pop(session_id, None)

    def get_summary(self, session_id: str) -> Optional[str]:
        info = self._sessions.get(session_id)
        return info.get("summary") if info else None

    def set_summary(self, session_id: str, summary: Optional[str]) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["summary"] = summary
            _log.info("set_summary session=%s summary_len=%d",
                      session_id, len(summary) if summary else 0)

    def prune(self, session_id: str, keep: int) -> None:
        msgs = self._messages.get(session_id)
        if not msgs:
            return
        before = len(msgs)
        if before > keep:
            self._messages[session_id] = msgs[before - keep:]
            _log.info("prune session=%s before=%d keep=%d removed=%d",
                      session_id, before, keep, before - keep)

    def enforce_caps(self, max_sessions: int, max_messages_per_session: int) -> None:
        # 单会话消息数上限：保留最新 max_messages_per_session 条
        for sid, msgs in self._messages.items():
            if len(msgs) > max_messages_per_session:
                self._messages[sid] = msgs[len(
                    msgs) - max_messages_per_session:]
        # 会话数上限：按 updated_at 最旧 LRU 淘汰
        if len(self._sessions) > max_sessions:
            ordered = sorted(
                self._sessions.items(),
                key=lambda kv: (kv[1]["updated_at"], kv[1]["created_at"]),
            )
            for sid, _ in ordered[: len(self._sessions) - max_sessions]:
                self.delete_session(sid)

    def clear_all(self) -> None:
        self._sessions.clear()
        self._messages.clear()

    def list_sessions(self) -> Dict[str, float]:
        return {sid: info["updated_at"] for sid, info in self._sessions.items()}

    def close(self) -> None:
        pass
