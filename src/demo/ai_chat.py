from component.agent_service import AgentService
from component.cli import cli_chat
from config.config import load_config
import os
import sys
import argparse
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(root_path, "../"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI chat demo")
    parser.add_argument("-c", "--config", default="config.toml",
                        help="Path to the config TOML file (default: config.toml)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear the entire context cache store before starting")
    args = parser.parse_args()

    config = load_config(args.config)

    # 构造 Agent 服务：装配 LLM + Agent + 上下文缓存（含 RAG/摘要），并恢复会话
    service = AgentService(config, clear_cache=args.clear_cache)

    # chat：按 session_id 精确读写，开启 RAG/摘要时为三层组装
    cli_chat(agent=service.agent, agent_config=config.agent,
             chat_handler=service.chat_handler(),
             context_manager=service.context_manager,
             session_id=service.session_id)

    # cleanup
    service.close()
