# -*- coding: utf-8 -*-
"""独立可复用的上下文缓存模块（ContextStore）。

- ContextStore: 存储抽象基类
- StoredMessage: 可移植 JSON 载荷的消息模型
- InMemoryContextStore: 进程内存后端，见 common/memory_store.py
- SqliteContextStore: SQLite 后端（默认），见 common/sqlite_store.py
- create_store: 按 backend 配置创建后端实例

设计要点：
- updated_at 仅在写入时更新（读取不 touch）
- 数量上限保留（max_sessions LRU / max_messages_per_session），不做时间 TTL
- 消息以 JSON 载荷存储，保真 tool_call_id/name/args
"""
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class StoredMessage:
    """存储层的消息载荷（可移植 JSON，与 langchain BaseMessage 解耦）。

    - role: system | user | assistant | tool
    - content: 文本内容（tool 消息为工具输出）
    - tool_call_id: 工具消息关联的调用 id
    - name: 工具名
    - args: 工具调用参数（JSON 字符串）
    """
    role: str
    content: str
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    args: Optional[str] = None
    created_at: Optional[float] = None

    def to_json(self) -> str:
        """序列化为可移植 JSON 载荷字符串。"""
        payload: dict = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        if self.args is not None:
            payload["args"] = self.args
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str, created_at: Optional[float] = None) -> "StoredMessage":
        """从 JSON 载荷字符串解析为 StoredMessage。"""
        payload = json.loads(raw)
        return cls(
            role=payload.get("role", ""),
            content=payload.get("content", ""),
            tool_call_id=payload.get("tool_call_id"),
            name=payload.get("name"),
            args=payload.get("args"),
            created_at=created_at,
        )


def _now() -> float:
    return time.time()


class ContextStore(ABC):
    """上下文缓存存储抽象基类。"""

    @abstractmethod
    def create_session(self, meta: Optional[str] = None) -> str:
        """创建一个会话，返回会话 id。"""
        raise NotImplementedError

    @abstractmethod
    def append(self, session_id: str, message: StoredMessage) -> None:
        """向会话追加一条消息，并更新会话 updated_at（仅写入时）。"""
        raise NotImplementedError

    @abstractmethod
    def get_messages(self, session_id: str) -> List[StoredMessage]:
        """按写入顺序返回会话的全部消息。"""
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """删除会话及其全部消息（级联）。"""
        raise NotImplementedError

    @abstractmethod
    def enforce_caps(self, max_sessions: int, max_messages_per_session: int) -> None:
        """数量上限保留：超 max_sessions 按 updated_at 最旧淘汰；单会话超上限删最旧消息。"""
        raise NotImplementedError

    @abstractmethod
    def clear_all(self) -> None:
        """清空全部会话与消息。"""
        raise NotImplementedError

    @abstractmethod
    def list_sessions(self) -> Dict[str, float]:
        """返回 {session_id: updated_at}，供 LRU 排序与测试。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""
        raise NotImplementedError


def create_store(backend: str = "sqlite", path: str = "data/context_cache.db") -> ContextStore:
    """按 backend 配置创建存储后端实例。

    后端实现使用延迟导入，避免模块加载时与具体后端产生循环依赖。
    """
    if backend == "memory":
        from common.memory_store import InMemoryContextStore
        return InMemoryContextStore()
    from common.sqlite_store import SqliteContextStore
    return SqliteContextStore(path)
