# -*- coding: utf-8 -*-
"""本地 embedding 工厂：优先 sentence-transformers，缺依赖时回退确定性哈希向量。

make_embed_fn(model) 返回 `Callable[[str], list[float]]`。
"""
import hashlib
from typing import Callable, List


def make_embed_fn(model: str = "local:bge-small-zh") -> Callable[[str], List[float]]:
    """构造文本 embedding 函数。

    优先使用本地 sentence-transformers 模型（如 bge-small-zh）；
    若未安装该依赖，则回退到确定性字符哈希向量，保证 demo 可离线运行（质量较低但稳定）。
    """
    name = model.split(":", 1)[-1] if ":" in model else model
    try:
        # 可选依赖：未安装时回退哈希向量；此处忽略静态导入检查
        # type: ignore[reportMissingImports]
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(name)

        def embed(text: str) -> List[float]:
            vec = encoder.encode(text, normalize_embeddings=True)
            return [float(x) for x in vec.tolist()]

        return embed
    except Exception:
        return _hash_embed


def _hash_embed(text: str, dim: int = 128) -> List[float]:
    """确定性哈希 embedding：按 3-gram 哈希到维度桶并做 L2 归一化。"""
    vec = [0.0] * dim
    t = " ".join(text.lower().split())
    for i in range(len(t)):
        gram = t[i:i + 3]
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]
