from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from config.config import AgentConfig
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from typing import Callable
from langgraph.graph.state import CompiledStateGraph
from rich import print

# 修正函数签名：回调(agent,配置,用户字符串) → 返回回答字符串
ChatHandler = Callable[[CompiledStateGraph, AgentConfig, str], str]
def cli_chat(agent: CompiledStateGraph,
             agent_config: AgentConfig,
             chat_handler: ChatHandler):
    """
    通用命令行交互终端
    :param agent: langgraph 编译后的Agent实例
    :param agent_config: agent运行配置
    :param chat_handler: 处理句柄，入参(agent,config,用户提问)，返回回答文本
    """
    history = FileHistory(".chat_history")
    print("[green]AI命令行问答终端，输入 q 退出、clear 清空输入历史[/green]")

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

        answer = chat_handler(agent, agent_config, user_text)
        print(f"[blue]🤖 {answer}[/blue]\n")