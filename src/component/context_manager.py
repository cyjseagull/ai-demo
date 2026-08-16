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

from component.context_store import ContextStore, StoredMessage, create_store
from component.logger import get_logger
from config.config import CacheConfig

_log = get_logger("context_manager")


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
            tool_calls=[{"id": msg.tool_call_id or "",
                         "name": msg.name, "args": args}],
        )
    return AIMessage(content=msg.content)


def _count_messages(msgs: List[BaseMessage]) -> int:
    """条数计数（token_counter 的简化实现，用于条数窗口裁剪）。"""
    return len(msgs)


class ContextManager:
    """上下文缓存门面。缓存未启用时请勿构造（由调用方判断 enable）。"""

    def __init__(self, cache: CacheConfig, summarizer=None, rag_index=None):
        self.cache = cache
        self.summarizer = summarizer   # 可选：Callable[[list[BaseMessage]], str]
        self.rag_index = rag_index     # 可选：component.rag_index.RagIndex
        self.rag_top_k = 4             # 检索命中条数，可由调用方按 RagConfig.top_k 覆盖
        self.min_relevance = 0.25      # 相关性门控阈值：低于则丢弃历史窗口
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

    def _maybe_summarize(self, session_id: str) -> None:
        """历史超过阈值时，把更早轮次折叠进运行摘要并裁剪（仅 summarize 策略且已注入 summarizer）。"""
        if self.summarizer is None or self.cache.trim_strategy != "summarize":
            return
        s = self._ensure_store()
        threshold = self.cache.summary_trigger_messages
        keep = self.cache.recent_window_size
        messages = s.get_messages(session_id)
        if len(messages) < threshold:
            return
        older = messages[: len(messages) - keep]   # 超出最近窗口的旧轮次
        if not older:
            return
        old_summary = s.get_summary(session_id)
        fold: List[BaseMessage] = []
        if old_summary:
            fold.append(SystemMessage(content=f"摘要：{old_summary}"))
        fold += [to_langchain(m) for m in older]
        new_summary = self.summarizer(fold)
        s.set_summary(session_id, new_summary)
        s.prune(session_id, keep)
        _log.info(
            "summarize session=%s folded=%d keep=%d new_summary_len=%d",
            session_id, len(older), keep, len(new_summary),
        )

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
                self._current_session = max(
                    sessions.items(), key=lambda kv: kv[1])[0]
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

    def _index_turn(self, session_id: str, role: str, text: str) -> None:
        """把一轮对话写入向量索引（启用 rag 时）。"""
        if self.rag_index is None:
            return
        s = self._ensure_store()
        turn_idx = len(s.get_messages(session_id)) - 1
        self.rag_index.add_turn(session_id, turn_idx, role, text)

    # ---------- 读写 ----------

    def get_context(self, session_id: Optional[str] = None, question: Optional[str] = None) -> List[BaseMessage]:
        """组装上下文：运行摘要 + topK 最相关的历史轮次 + 当前问题。

        - question 非空且启用 RAG：检索 topK 最相关的历史轮次，**仅注入单条相似度 >= min_relevance**
          的轮次（按时间序）。不再对"全部历史取最大相似度"做整体门控，也不常开工作记忆。
        - 无相关命中时，上下文只含 摘要 + 问题（避免无关历史全量下发）。
        - 发送前按预算 trim_messages 兜底（系统提示与当前问题保留）。
        """
        s = self._ensure_store()
        sid = self._resolve_session(session_id)
        stored = s.get_messages(sid)
        summary = s.get_summary(sid)

        msgs: List[BaseMessage] = []
        if summary:
            msgs.append(SystemMessage(content=f"此前对话摘要：\n{summary}"))

        retrieved_n = 0
        if question is not None and self.rag_index is not None:
            hits = self.rag_index.search(question, sid, top_k=self.rag_top_k)
            # 仅注入"纯相似度"达标的轮次（不含近因加权，避免无关问题的最近轮次被注入）
            hits = [h for h in hits if h[4] >= self.min_relevance]
            hits.sort(key=lambda h: h[0])   # 按时间序注入
            for turn_idx, role, text, _score, _sim in hits:
                msgs.append(AIMessage(content=text) if role ==
                            "assistant" else HumanMessage(content=text))
                retrieved_n += 1
            if retrieved_n == 0:
                _log.info(
                    "get_context session=%s no relevant history injected", sid)

        budget = self.cache.max_context_messages
        before = len(msgs)
        if budget > 0 and len(msgs) > budget:
            msgs = trim_messages(
                msgs,
                token_counter=_count_messages,   # 条数计数 → 条数窗口
                max_tokens=budget,
                strategy="last",
                include_system=True,             # 系统提示始终保留
            )
        if question is not None:
            msgs = msgs + [HumanMessage(content=question)]

        _log.info(
            "get_context session=%s stored=%d summary=%s retrieved=%d sent=%d budget=%d",
            sid, len(stored), "yes" if summary else "no", retrieved_n, len(
                msgs), budget,
        )
        return msgs

    def add_user(self, text: str, session_id: Optional[str] = None) -> None:
        s = self._ensure_store()
        sid = self._resolve_session(session_id)
        s.append(sid, StoredMessage(role="user", content=text))
        _log.info("add_user session=%s len=%d", sid, len(text))
        self._maybe_enforce()
        self._maybe_summarize(sid)
        self._index_turn(sid, "user", text)

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
        _log.info("add_assistant session=%s len=%d", sid, len(content))
        self._maybe_enforce()
        self._maybe_summarize(sid)
        self._index_turn(sid, "assistant", content)

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
