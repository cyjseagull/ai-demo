from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from config.config import AgentConfig
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from typing import Callable, Optional
from langgraph.graph.state import CompiledStateGraph
from rich import print

# 修正函数签名：回调(agent,配置,用户字符串[,会话id]) → 返回回答字符串
ChatHandler = Callable[[CompiledStateGraph, AgentConfig, str, Optional[str]], str]
def cli_chat(agent: CompiledStateGraph,
             agent_config: AgentConfig,
             chat_handler: ChatHandler,
             context_manager=None,
             session_id: Optional[str] = None):
    """
    通用命令行交互终端
    :param agent: langgraph 编译后的Agent实例
    :param agent_config: agent运行配置
    :param chat_handler: 处理句柄，入参(agent,config,用户提问[,会话id])，返回回答文本
    :param context_manager: 可选上下文缓存门面；提供 /clear、/new 会话控制
    :param session_id: 当前会话 id；有值时会话相关命令会更新它，并透传给 chat_handler
    """
    history = FileHistory(".chat_history")
    print("[green]AI命令行问答终端：输入 q 退出、clear 清空输入历史、/clear 清空当前会话、/new 开启新会话[/green]")

    while True:
        user_text = prompt(
            "👦 > ",
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            multiline=False
        )
        cmd = user_text.strip().lower()
        if cmd in ("q", "quit", "exit"):
            print("[red]会话结束[/red]")
            break
        if cmd == "clear":
            history._loaded_strings.clear()
            continue
        if cmd == "/clear":
            if context_manager is not None:
                context_manager.clear_current()
                session_id = context_manager.start_new_session()   # 清空后立即开启新会话
                print("[yellow]当前会话上下文已清空，已开启新会话[/yellow]")
            else:
                print("[yellow]上下文缓存未启用，无需清空[/yellow]")
            continue
        if cmd == "/new":
            if context_manager is not None:
                session_id = context_manager.start_new_session()
                print(f"[yellow]已开启新会话：{session_id}[/yellow]")
            else:
                print("[yellow]上下文缓存未启用，无需新建会话[/yellow]")
            continue

        answer = chat_handler(agent, agent_config, user_text, session_id)
        print(f"[blue]🤖 {answer}[/blue]\n")