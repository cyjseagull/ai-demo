# -*- coding: utf-8 -*-
"""RAG 索引：对话轮次向量化存储与检索的抽象与工厂。

- RagIndex: 抽象基类（add_turn / search）
- SqliteVecRagIndex: 默认实现，见 component/rag_sqlite_vec.py
- create_rag_index: 按 backend 配置创建索引实例

具体实现使用延迟导入，避免模块加载时产生循环依赖。
"""
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

EmbedFn = Callable[[str], List[float]]


class RagIndex(ABC):
    """对话轮次向量索引抽象基类。"""

    @abstractmethod
    def add_turn(self, session_id: str, turn_idx: int, role: str, text: str) -> None:
        """将一轮对话切片嵌入并写入索引。"""
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query: str, session_id: str, top_k: int = 4
    ) -> List[Tuple[int, str, str, float]]:
        """按语义相似度检索同会话相关轮次。

        返回 [(turn_idx, role, text, score)]，按 score 降序；turn_idx 供上层与最近窗口去重。
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""
        raise NotImplementedError


def create_rag_index(
    backend: str = "sqlite_vec",
    path: str = "data/context_cache.db",
    embed_fn: Optional[EmbedFn] = None,
    recent_boost: float = 0.3,
) -> RagIndex:
    """按 backend 配置创建 RAG 索引实例（当前默认 sqlite_vec）。

    :param backend: sqlite_vec（默认） | chroma（预留）
    :param embed_fn: 文本 embedding 函数 Callable[[str], list[float]]
    """
    if backend == "sqlite_vec":
        from component.rag_sqlite_vec import SqliteVecRagIndex
        if embed_fn is None:
            raise ValueError("sqlite_vec 后端需要提供 embed_fn")
        return SqliteVecRagIndex(path, embed_fn, recent_boost=recent_boost)
    raise ValueError(f"不支持的 RAG 后端: {backend}")


def __getattr__(name: str):
    # 延迟重导出 SqliteVecRagIndex，保持 from component.rag_index import SqliteVecRagIndex 可用
    if name == "SqliteVecRagIndex":
        from component.rag_sqlite_vec import SqliteVecRagIndex
        return SqliteVecRagIndex
    raise AttributeError(
        f"module 'component.rag_index' has no attribute {name!r}")
