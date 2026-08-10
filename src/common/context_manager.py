# -*- coding: utf-8 -*-
"""ContextManager 门面：结合 ContextStore + 窗口裁剪 + 数量上限保留。

- 自动创建/复用当前会话；也可按显式 session_id 读写（防止误取其他会话造成上下文浪费）
- get_context: 加载指定会话（缺省当前）历史并按窗口裁剪（保留系统提示、工具消息成对）
- add_user / add_assistant: 保存消息并更新会话
- start_new_session: 创建全新会话（/new），避免复用旧会话导致上下文浪费
- clear_current / clear_all: 手动清理
- enforce_caps_if_needed: 节流执行数量上限（max_sessions LRU / max_messages_per_session）
"""
import json
from typing import List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

from common.context_store import ContextStore, StoredMessage, create_store
from config.config import CacheConfig


# ---------- StoredMessage <-> BaseMessage 转换（薄适配层） ----------

def to_langchain(msg: StoredMessage) -> BaseMessage:
    """将存储消息载荷转换为 langchain BaseMessage。"""
    if msg.role == "user":
        return HumanMessage(content=msg.content)
    if msg.role == "system":
        return SystemMessage(content=msg.content)
    if msg.role == "tool":
        return ToolMessage(content=msg.content, tool_call_id=msg.tool_call_id or "", name=msg.name)
    # assistant
    if msg.name and msg.args:
        try:
            args = json.loads(msg.args)
        except Exception:
            args = msg.args
        return AIMessage(
            content=msg.content,
            tool_calls=[{"id": msg.tool_call_id or "", "name": msg.name, "args": args}],
        )
    return AIMessage(content=msg.content)


def _count_messages(msgs: List[BaseMessage]) -> int:
    """条数计数（token_counter 的简化实现，用于条数窗口裁剪）。"""
    return len(msgs)


class ContextManager:
    """上下文缓存门面。缓存未启用时请勿构造（由调用方判断 enable）。"""

    def __init__(self, cache: CacheConfig):
        self.cache = cache
        self._store: Optional[ContextStore] = None
        self._current_session: Optional[str] = None
        self._start_fresh = False   # 显式清理后下次创建新会话，而非恢复旧会话
        self._write_count = 0

    # ---------- 内部 ----------

    def _ensure_store(self) -> ContextStore:
        if self._store is None:
            self._store = create_store(self.cache.backend, self.cache.path)
            self.enforce_caps_if_needed()   # 初始化时执行一次上限清理
        return self._store

    def _maybe_enforce(self) -> None:
        # 每写入一定次数节流执行上限清理，避免每轮都全表扫描
        self._write_count += 1
        if self._write_count % 20 == 0:
            self.enforce_caps_if_needed()

    # ---------- 会话 ----------

    def current_session(self) -> str:
        """自动创建/复用当前会话。

        - 已有有效会话：直接复用
        - 首次访问：若存储中已有会话，恢复最近写入的会话（进程重启后继续对话）
        - 显式清理（/clear、--clear-cache）后：创建全新会话
        """
        s = self._ensure_store()
        if self._current_session is None or self._current_session not in s.list_sessions():
            sessions = s.list_sessions()
            if sessions and not self._start_fresh:
                # 恢复最近写入的会话（进程重启后继续对话）
                self._current_session = max(sessions.items(), key=lambda kv: kv[1])[0]
            else:
                self._current_session = s.create_session()
                self._start_fresh = False
        assert self._current_session is not None
        return self._current_session

    def start_new_session(self) -> str:
        """创建全新会话并设为当前（/new），返回新会话 id。

        用于显式开启新话题，避免自动恢复旧会话导致上下文浪费。
        """
        s = self._ensure_store()
        self._current_session = s.create_session()
        self._start_fresh = True
        return self._current_session

    def _resolve_session(self, session_id: Optional[str] = None) -> str:
        """解析目标会话：显式 session_id 优先（须已存在）；缺省走当前会话。"""
        if session_id is not None:
            s = self._ensure_store()
            if session_id not in s.list_sessions():
                raise ValueError(
                    f"会话不存在: {session_id}（请通过 current_session()/start_new_session() 获取有效会话 id）"
                )
            return session_id
        return self.current_session()

    # ---------- 读写 ----------

    def get_context(self, session_id: Optional[str] = None) -> List[BaseMessage]:
        """加载指定会话（缺省为当前会话）的历史并按窗口裁剪，返回送入模型的 BaseMessage 列表。

        显式传入 session_id 可精确定位会话，避免误取其他会话造成上下文浪费。
        """
        s = self._ensure_store()
        sid = self._resolve_session(session_id)
        stored = s.get_messages(sid)
        msgs = [to_langchain(m) for m in stored]
        budget = self.cache.max_context_messages
        if budget > 0 and len(msgs) > budget:
            msgs = trim_messages(
                msgs,
                token_counter=_count_messages,   # 条数计数 → 条数窗口
                max_tokens=budget,
                strategy="last",
                include_system=True,             # 系统提示始终保留
            )
        return msgs

    def add_user(self, text: str, session_id: Optional[str] = None) -> None:
        s = self._ensure_store()
        sid = self._resolve_session(session_id)
        s.append(sid, StoredMessage(role="user", content=text))
        self._maybe_enforce()

    def add_assistant(
        self,
        content: str,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
        args: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        s = self._ensure_store()
        sid = self._resolve_session(session_id)
        s.append(
            sid,
            StoredMessage(
                role="assistant",
                content=content,
                tool_call_id=tool_call_id,
                name=name,
                args=args,
            ),
        )
        self._maybe_enforce()

    # ---------- 清理与上限 ----------

    def clear_current(self) -> None:
        """清空当前会话（/clear）。"""
        s = self._ensure_store()
        if self._current_session is not None:
            s.delete_session(self._current_session)
        self._current_session = None
        self._start_fresh = True   # 下次创建全新会话，而非恢复其他旧会话

    def clear_all(self) -> None:
        """清空整个存储（--clear-cache）。"""
        s = self._ensure_store()
        s.clear_all()
        self._current_session = None
        self._start_fresh = True

    def enforce_caps_if_needed(self) -> None:
        """执行数量上限保留（max_sessions LRU / max_messages_per_session）。"""
        if self._store is not None:
            self._store.enforce_caps(
                self.cache.max_sessions, self.cache.max_messages_per_session
            )

    # ---------- 资源 ----------

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
