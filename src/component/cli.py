from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from config.config import AgentConfig
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from typing import Callable, Optional
from langgraph.graph.state import CompiledStateGraph
from rich import print

# 修正函数签名：回调(agent,配置,用户字符串[,会话id]) → 返回回答字符串
ChatHandler = Callable[[CompiledStateGraph,
                        AgentConfig, str, Optional[str]], str]


def cli_chat(agent: CompiledStateGraph,
             agent_config: AgentConfig,
             chat_handler: ChatHandler,
             context_manager=None,
             session_id: Optional[str] = None,
             stream: bool = True,
             agent_service=None):
    """
    通用命令行交互终端
    :param agent: langgraph 编译后的Agent实例
    :param agent_config: agent运行配置
    :param chat_handler: 处理句柄，入参(agent,config,用户提问[,会话id])，返回回答文本
    :param context_manager: 可选上下文缓存门面；提供 /clear、/new 会话控制
    :param session_id: 当前会话 id；有值时会话相关命令会更新它，并透传给 chat_handler
    :param stream: 为 True 时流式显示（先打印 🤖 前缀，handler 逐 token 输出，避免长时间停顿）
    :param agent_service: 可选 AgentService；提供写文件待写项的展示与人工审核（y/e/d）
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

        if stream:
            print("[blue]🤖 [/blue]", end="", flush=True)
            chat_handler(agent, agent_config, user_text,
                         session_id)   # 流式：handler 内部逐 token 输出
            print("", flush=True)
        else:
            answer = chat_handler(agent, agent_config, user_text, session_id)
            print(f"[blue]🤖 {answer}[/blue]\n")

        # 写文件人工审核：本轮若产生待写项，逐个展示并 y/e/d 确认后才落盘
        _review_pending(agent_service)


def _review_pending(agent_service) -> None:
    """人工审核待写项：展示路径 + 内容预览（截断），按 y/e/d 确认后落盘。

    y=overwrite 覆盖写入 / e=append 追加写入 / d=discard 丢弃；回车跳过剩余。
    写盘唯一入口在 AgentService.confirm_write（人工审核强制，无自动写入分支）。
    """
    if agent_service is None:
        return
    if not agent_service.pending_writes():
        return
    print("[yellow]📝 检测到待写入文件（需人工审核）[/yellow]")
    while agent_service.pending_writes():
        item = agent_service.pending_writes()[0]
        path = item["path"]
        content = item["content"]
        preview = content if len(content) <= 200 else content[:200] + "…（预览截断）"
        print(f"[cyan]  待写: {path}[/cyan]（{len(content)} 字符）")
        for line in preview.splitlines():
            print(f"      {line}")
        choice = prompt(
            "  处理（y=写入覆盖 / e=追加 / d=丢弃，回车=跳过剩余）> "
        ).strip().lower()
        if choice in ("y", "e", "d"):
            mode = {"y": "overwrite", "e": "append", "d": "discard"}[choice]
            print(f"[green]  {agent_service.confirm_write(0, mode)}[/green]")
        elif choice == "":
            break
        else:
            print("[red]  无效输入，请输入 y / e / d（回车跳过剩余）[/red]")
