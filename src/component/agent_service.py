# -*- coding: utf-8 -*-
"""AgentService：把 LLM + Agent + 上下文缓存（含 RAG/摘要）封装为可复用的服务类。

用法：
    service = AgentService(config, clear_cache=False)
    reply = service.handle("你好", session_id=service.session_id)
    service.close()
"""
import os
from typing import Any, List, Optional, Tuple, cast

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from component.context_manager import ContextManager
from component.embeddings import make_embed_fn
from component.file_tool import PendingWrite
from component.logger import get_logger
from component.rag_index import create_rag_index
from component.summarizer import make_summarizer
from component.token_usage import TokenUsageCallback
from config.config import AppConfig

_log = get_logger("agent_service")


class AgentService:
    """Agent 服务：装配 LLM + Agent + 上下文缓存，并提供一轮对话处理。"""

    def __init__(self, config: AppConfig, clear_cache: bool = False):
        self.config = config
        tools_cfg = getattr(config, "tools", None)
        self._base_dir = os.path.abspath(getattr(
            getattr(tools_cfg, "save_file", None), "base_dir", "outputs"))
        # 待写暂存队列：save_research 只入队，人工审核确认后才落盘
        self._pending_writes: List[PendingWrite] = []
        self.llm, self.agent = self.init_agent()
        self.context_manager, self.session_id = self.recover_context(
            clear_cache=clear_cache)

    # ---------- 装配 ----------

    def init_agent(self) -> Tuple[Any, CompiledStateGraph]:
        """创建 LLM 与 langgraph Agent（按配置注入联网搜索工具）。"""
        llm = ChatOpenAI(model=self.config.llm.model,
                         temperature=self.config.llm.temperature,
                         base_url=self.config.llm.base_url,
                         api_key=self.config.llm.api_key,
                         stream_usage=True)   # 流式时也返回 usage，供 token 用量监控
        tools = []
        search = getattr(self.config, "search", None)
        if search is not None and search.enable:
            from component.search_tool import make_search_tool
            tools = [make_search_tool(search.provider, search.max_results)]
        # 写文件工具：功能默认开启（无 enable 开关），始终注入；入队不写盘
        from component.file_tool import SAVE_IDENTIFY_RULE, make_file_tool
        tools = tools + \
            [make_file_tool(self._base_dir, enqueue=self._enqueue_write)]
        system_prompt = self.config.agent.system_prompt + SAVE_IDENTIFY_RULE
        agent: CompiledStateGraph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            debug=self.config.agent.debug)
        return llm, agent

    def recover_context(self, clear_cache: bool = False):
        """装配上下文缓存（RAG 索引 + 运行摘要）并恢复会话。

        :param clear_cache: 是否在启动时清空整个存储
        :return: (context_manager, session_id)；未启用缓存时 context_manager 为 None
        """
        if not self.config.cache.enable:
            return None, None

        rag_index = None
        if self.config.rag.enable:
            embed_fn = make_embed_fn(self.config.rag.embed_model)
            rag_index = create_rag_index(self.config.rag.backend, self.config.cache.path,
                                         embed_fn, recent_boost=self.config.rag.recent_boost)
        summarizer = make_summarizer(
            self.llm) if self.config.cache.trim_strategy == "summarize" else None
        context_manager = ContextManager(
            self.config.cache, summarizer=summarizer, rag_index=rag_index)
        context_manager.rag_top_k = self.config.rag.top_k
        context_manager.min_relevance = getattr(
            self.config.rag, "min_relevance", 0.5)
        if clear_cache:
            context_manager.clear_all()
        session_id = context_manager.current_session()
        return context_manager, session_id

    # ---------- 对话 ----------

    def handle(self, user_query: str, session_id: Optional[str] = None,
               stream: bool = False, on_token=None) -> str:
        """处理一轮对话：三层组装上下文 → invoke/流式 → 保存回复。

        :param stream: 为 True 时用 agent.stream(stream_mode="messages") 逐 token 输出，避免停顿感
        :param on_token: 流式回调 Callable[[str], None]；缺省直接 print 到 stdout
        """
        sid = self.session_id if session_id is None else session_id
        messages: List[BaseMessage]
        if self.context_manager is not None:
            # 把当前问题传给 get_context，组装"摘要+检索命中+最近窗口+当前问题"
            messages = self.context_manager.get_context(sid, user_query)
        else:
            messages = [HumanMessage(content=user_query)]

        agent_cfg = self.config.agent
        # 模型用量监控：回调采集每次 LLM 调用 usage（覆盖 Agent 内部多次调用）
        usage_cb = TokenUsageCallback(
            session_id=sid, model=self.config.llm.model)
        invoke_config = cast(Any, {
            "callbacks": [usage_cb],
            "max_iterations": agent_cfg.max_iterations,
            "handle_parsing_errors": agent_cfg.handle_parsing_errors,
            "return_intermediate_steps": agent_cfg.return_intermediate_steps,
            "max_execution_time": agent_cfg.max_execution_time,
        })

        if stream:
            reply = self._stream_invoke(
                cast(Any, {"messages": messages}), invoke_config, on_token)
        else:
            res = self.agent.invoke(
                cast(Any, {"messages": messages}), config=invoke_config)
            reply = res["messages"][-1].text

        # 会话级汇总：总 token / 缓存命中率 / 成本估算（日志输出，不落库）
        usage_cb.log_round_summary()

        if self.context_manager is not None:
            self.context_manager.add_user(user_query, session_id=sid)
            self.context_manager.add_assistant(reply, session_id=sid)
        return reply

    def _stream_invoke(self, agent_input, invoke_config, on_token=None) -> str:
        """用 agent.stream(stream_mode="messages") 逐 token 输出并返回完整文本。"""
        from langchain_core.messages import AIMessageChunk
        parts: List[str] = []

        def emit(text: str) -> None:
            parts.append(text)
            if on_token:
                on_token(text)
            else:
                print(text, end="", flush=True)

        for chunk, _meta in self.agent.stream(
            agent_input, config=invoke_config, stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                text = chunk.content if isinstance(
                    chunk.content, str) else str(chunk.content)
                if text:
                    emit(text)
        return "".join(parts)

    def chat_handler(self, stream: bool = True):
        """返回兼容 cli.ChatHandler 的回调：入参(agent, config, text[, session_id])。

        stream=True 时流式输出（逐 token 打印）。
        """
        if stream:
            return lambda agent, agent_config, user_query, session_id=None: self.handle(
                user_query, session_id, stream=True)
        return lambda agent, agent_config, user_query, session_id=None: self.handle(
            user_query, session_id)

    # ---------- 写文件：暂存 + 人工审核 ----------

    def _enqueue_write(self, file_path: str, content: str) -> None:
        """save_research 工具入暂存回调：仅入队，不写盘。"""
        self._pending_writes.append(
            PendingWrite(path=file_path, content=content))

    def pending_writes(self) -> List[dict]:
        """返回待写项快照（供 CLI 展示与审核）。"""
        return [{"path": w.path, "content": w.content} for w in self._pending_writes]

    def confirm_write(self, index: int, mode: str) -> str:
        """执行人工审核结果；这是唯一写盘入口（人工审核强制，无自动写入分支）。

        :param mode: overwrite 覆盖 / append 追加 / discard 丢弃
        """
        if index < 0 or index >= len(self._pending_writes):
            return f"待写项索引无效：{index}"
        item = self._pending_writes[index]
        if mode == "discard":
            self._pending_writes.pop(index)
            _log.info("丢弃待写项: %s", item.path)
            return f"已丢弃待写项：{item.path}"
        if mode not in ("overwrite", "append"):
            return f"未知模式：{mode}（overwrite / append / discard）"
        # 落盘前再次安全校验（防御纵深：工具内已校验，这里再兜底一次）
        from component.file_tool import validate_path
        norm = validate_path(item.path, self._base_dir)
        if norm is None:
            self._pending_writes.pop(index)
            return f"路径校验失败，已丢弃：{item.path}"
        full = os.path.join(self._base_dir, norm)
        parent = os.path.dirname(full)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full, "w" if mode == "overwrite" else "a",
                  encoding="utf-8") as f:
            f.write(item.content)
        self._pending_writes.pop(index)
        _log.info("已写入文件: %s（%s）", full, mode)
        return f"已写入：{full}（{mode}）"

    def clear_pending(self) -> None:
        """清空全部待写项（例如退出时丢弃未确认内容）。"""
        self._pending_writes.clear()

    # ---------- 资源 ----------

    def close(self) -> None:
        if self.context_manager is not None:
            self.context_manager.close()
