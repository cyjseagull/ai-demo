import os, sys
import argparse
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(root_path, "../"))

from config.config import load_config, AppConfig, AgentConfig
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from common.cli import cli_chat
from common.context_manager import ContextManager

def agent_handle(agent, agent_config: AgentConfig, user_query: str,
                 context_manager=None, session_id=None) -> str:
    """处理一轮对话：加载指定会话（缺省当前会话）历史 → invoke → 保存回复。

    显式传入 session_id 可精确定位会话，避免误取其他会话造成上下文浪费；
    context_manager 为 None 时保持无上下文行为。
    """
    messages = []
    if context_manager is not None:
        messages = context_manager.get_context(session_id)
    messages.append(HumanMessage(content=user_query))

    res = agent.invoke({"messages": messages},
                        config = {"max_iterations": agent_config.max_iterations,
                                    "handle_parsing_errors": agent_config.handle_parsing_errors,
                                    "return_intermediate_steps": agent_config.return_intermediate_steps,
                                    "max_execution_time": agent_config.max_execution_time,
                                    })
    reply_msg = res["messages"][-1]
    reply = reply_msg.text

    if context_manager is not None:
        context_manager.add_user(user_query, session_id=session_id)
        context_manager.add_assistant(reply, session_id=session_id)
    return reply

def init_agent(config: AppConfig):
    llm = ChatOpenAI(model=config.llm.model, 
        temperature=config.llm.temperature, 
        base_url=config.llm.base_url, 
        api_key=config.llm.api_key)
    agent = create_agent(
        model=llm,
        system_prompt=config.agent.system_prompt,
        debug=config.agent.debug)
    return (llm, agent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI chat demo")
    parser.add_argument("-c", "--config", default="config.toml",
                        help="Path to the config TOML file (default: config.toml)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear the entire context cache store before starting")
    args = parser.parse_args()

    config = load_config(args.config)

    # init context cache（enable 默认启用，由 [context.cache] 控制）
    context_manager = None
    session_id = None
    if config.cache.enable:
        context_manager = ContextManager(config.cache)
        if args.clear_cache:
            context_manager.clear_all()
        # 获取/恢复当前会话（重启后自动续聊）；会话 id 精确传递给 agent_handle
        session_id = context_manager.current_session()

    # init the agent
    (llm, agent) = init_agent(config)

    # chat with agent_handle：按 session_id 精确读写上下文，防止上下文浪费
    chat_handler = (lambda a, c, u, sid=None: agent_handle(a, c, u, context_manager, sid))
    cli_chat(agent = agent, agent_config = config.agent,
             chat_handler = chat_handler, context_manager = context_manager,
             session_id = session_id)

    # cleanup
    if context_manager is not None:
        context_manager.close()
