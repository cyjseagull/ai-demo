# -*- coding: utf-8 -*-
"""运行摘要（compaction）工厂：把历史消息压缩成要点摘要。

make_summarizer(llm) 返回一个 `Callable[[list[BaseMessage]], str]`：
入参为需要压缩的历史消息列表（首条若为"摘要：..."开头的 SystemMessage，
视为既有摘要，做增量合并），返回压缩后的摘要文本。
"""
from typing import Callable, List

from langchain_core.messages import BaseMessage

_SUMMARY_PROMPT = (
    "请把以下对话压缩成一份简洁的要点摘要，用于在长对话中保留关键上下文。\n"
    "要求：\n"
    "- 保留关键事实、结论、用户偏好、未完成事项\n"
    "- 人名 / 术语 / 数字必须准确\n"
    "- 语言与对话一致，正文控制在 200 字以内\n"
    "- 若对话中包含'此前摘要'，请在此基础上增量合并，不要重复已记录的内容\n\n"
    "对话内容：\n{conversation}"
)


def make_summarizer(llm) -> Callable[[List[BaseMessage]], str]:
    """基于 LLM 构造摘要函数。

    :param llm: 支持 `.invoke(prompt)` 并返回带 `.content` 的模型实例。
    """
    def summarize(messages: List[BaseMessage]) -> str:
        lines = []
        for m in messages:
            content = getattr(m, "content", None)
            if not content:
                continue
            lines.append(f"{m.type}: {content}")
        conversation = "\n".join(lines)
        prompt = _SUMMARY_PROMPT.format(conversation=conversation)
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)

    return summarize
