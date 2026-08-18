import os
import sys
import argparse

# 在导入项目包之前，先把 src 加入 sys.path 最前，优先使用本地 src 副本，
# 避免命中已安装的 site-packages 旧副本（旧副本缺新字段/无法定位 sql/ 目录）。
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(root_path, "../"))


def main():
    # 项目包在函数内导入：autopep8 不会把函数内的 import 提到顶部，
    # 从而保证上面的 sys.path 设置先执行、始终使用本地 src/ 副本。
    from config.config import load_config
    from component.cli import cli_chat
    from component.logger import setup_logging
    from component.agent_service import AgentService

    parser = argparse.ArgumentParser(description="AI chat demo")
    parser.add_argument("-c", "--config", default="config.toml",
                        help="Path to the config TOML file (default: config.toml)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear the entire context cache store before starting")
    args = parser.parse_args()

    config = load_config(args.config)

    # 按配置初始化日志（级别 + 可选文件路径）
    setup_logging(config.log.level, config.log.path)

    # 构造 Agent 服务：装配 LLM + Agent + 上下文缓存（含 RAG/摘要），并恢复会话
    service = AgentService(config, clear_cache=args.clear_cache)

    # chat：按 session_id 精确读写，开启 RAG/摘要时为三层组装
    cli_chat(agent=service.agent, agent_config=config.agent,
             chat_handler=service.chat_handler(),
             context_manager=service.context_manager,
             session_id=service.session_id,
             agent_service=service)

    # cleanup
    service.close()


if __name__ == "__main__":
    main()
