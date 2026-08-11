# -*- coding: utf-8 -*-
"""AgentService：把 LLM + Agent + 上下文缓存（含 RAG/摘要）封装为可复用的服务类。

用法：
    service = AgentService(config, clear_cache=False)
    reply = service.handle("你好", session_id=service.session_id)
    service.close()
"""
from typing import Any, List, Optional, Tuple, cast

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from component.context_manager import ContextManager
from component.embeddings import make_embed_fn
from component.rag_index import create_rag_index
from component.summarizer import make_summarizer
from config.config import AppConfig


class AgentService:
    """Agent 服务：装配 LLM + Agent + 上下文缓存，并提供一轮对话处理。"""

    def __init__(self, config: AppConfig, clear_cache: bool = False):
        self.config = config
        self.llm, self.agent = self.init_agent()
        self.context_manager, self.session_id = self.recover_context(
            clear_cache=clear_cache)

    # ---------- 装配 ----------

    def init_agent(self) -> Tuple[Any, CompiledStateGraph]:
        """创建 LLM 与 langgraph Agent。"""
        llm = ChatOpenAI(model=self.config.llm.model,
                         temperature=self.config.llm.temperature,
                         base_url=self.config.llm.base_url,
                         api_key=self.config.llm.api_key)
        agent: CompiledStateGraph = create_agent(
            model=llm,
            system_prompt=self.config.agent.system_prompt,
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
        if clear_cache:
            context_manager.clear_all()
        session_id = context_manager.current_session()
        return context_manager, session_id

    # ---------- 对话 ----------

    def handle(self, user_query: str, session_id: Optional[str] = None) -> str:
        """处理一轮对话：三层组装上下文（摘要+检索+窗口+问题）→ invoke → 保存回复。

        显式传入 session_id 可精确定位会话；未启用缓存时保持无上下文行为。
        """
        sid = self.session_id if session_id is None else session_id
        messages: List[BaseMessage]
        if self.context_manager is not None:
            # 把当前问题传给 get_context，组装"摘要+检索命中+最近窗口+当前问题"
            messages = self.context_manager.get_context(sid, user_query)
        else:
            messages = [HumanMessage(content=user_query)]

        agent_cfg = self.config.agent
        # 输入与 config 为 langgraph 动态状态类型，cast 到 Any 以适配 invoke 签名
        res = self.agent.invoke(
            cast(Any, {"messages": messages}),
            config=cast(Any, {
                "max_iterations": agent_cfg.max_iterations,
                "handle_parsing_errors": agent_cfg.handle_parsing_errors,
                "return_intermediate_steps": agent_cfg.return_intermediate_steps,
                "max_execution_time": agent_cfg.max_execution_time,
            }),
        )
        reply = res["messages"][-1].text

        if self.context_manager is not None:
            self.context_manager.add_user(user_query, session_id=sid)
            self.context_manager.add_assistant(reply, session_id=sid)
        return reply

    def chat_handler(self):
        """返回兼容 cli.ChatHandler 的回调：入参(agent, config, text[, session_id])。"""
        return lambda agent, agent_config, user_query, session_id=None: self.handle(user_query, session_id)

    # ---------- 资源 ----------

    def close(self) -> None:
        if self.context_manager is not None:
            self.context_manager.close()
