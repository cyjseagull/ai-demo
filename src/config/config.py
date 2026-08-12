import tomllib
from dataclasses import dataclass, field
from pydantic import SecretStr


@dataclass
class LLMConfig:
    model: str
    base_url: str
    api_key: SecretStr
    temperature: float


@dataclass
class AgentConfig:
    system_prompt: str = "You are a helpful assistant."
    debug: bool = False
    verbose: bool = True
    max_iterations: int = 5
    max_execution_time: int = 60
    handle_parsing_errors: bool = True
    return_intermediate_steps: bool = True


@dataclass
class CacheConfig:
    """上下文缓存配置（[context.cache] 段，可选）。"""
    enable: bool = True
    backend: str = "sqlite"                  # sqlite | memory
    path: str = "data/context_cache.db"      # sqlite 后端 db 文件路径
    max_sessions: int = 50                   # 会话数上限，超出按 updated_at 最旧淘汰
    max_messages_per_session: int = 200      # 单会话存储层消息上限
    max_context_messages: int = 30           # 送入模型的窗口裁剪条数
    # --- 运行摘要（compaction） ---
    trim_strategy: str = "sliding_window"    # sliding_window | summarize
    recent_window_size: int = 20             # 摘要模式下保留的原始消息窗口
    summary_trigger_messages: int = 50       # 历史超过该条数时触发摘要


@dataclass
class RagConfig:
    """上下文检索配置（[context.rag] 段）。"""
    enable: bool = False
    backend: str = "sqlite_vec"              # sqlite_vec | chroma
    embed_model: str = "local:bge-small-zh"  # embedding 模型
    top_k: int = 4                           # 检索命中条数
    recent_boost: float = 0.3                # 近因加权系数


@dataclass
class LogConfig:
    """日志配置（[log] 段）。"""
    level: str = "INFO"          # DEBUG | INFO | WARNING | ERROR
    path: str = ""               # 日志文件路径；为空则仅输出到控制台


@dataclass
class AppConfig:
    llm: LLMConfig
    agent: AgentConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    log: LogConfig = field(default_factory=LogConfig)


def load_config(config_path: str = "config.toml"):
    with open(config_path, "rb") as f:
        d = tomllib.load(f)

    # agent：仅取已知字段，容忍未知子段（如历史 [agent.cache]），避免破坏现有配置
    agent_fields = set(AgentConfig.__dataclass_fields__.keys())
    agent_cfg = AgentConfig(
        **{k: v for k, v in d.get("agent", {}).items() if k in agent_fields})

    # [context.cache]：可选段，缺失或部分字段缺失时回退默认值，不破坏既有配置
    cache_defaults = dict(
        enable=True,
        backend="sqlite",
        path="data/context_cache.db",
        max_sessions=50,
        max_messages_per_session=200,
        max_context_messages=30,
        trim_strategy="sliding_window",
        recent_window_size=20,
        summary_trigger_messages=50,
    )
    cache_sec = d.get("context", {}).get("cache", {})
    cache_cfg = CacheConfig(**{**cache_defaults, **cache_sec})

    # [context.rag]：可选段，缺失或部分字段缺失时回退默认值
    rag_defaults = dict(
        enable=False,
        backend="sqlite_vec",
        embed_model="local:bge-small-zh",
        top_k=4,
        recent_boost=0.3,
    )
    rag_sec = d.get("context", {}).get("rag", {})
    rag_cfg = RagConfig(**{**rag_defaults, **rag_sec})

    # [log]：可选段，缺失或部分字段缺失时回退默认值
    log_defaults = dict(level="INFO", path="")
    log_sec = d.get("log", {})
    log_cfg = LogConfig(**{**log_defaults, **log_sec})

    return AppConfig(
        llm=LLMConfig(**d["llm"]),
        agent=agent_cfg,
        cache=cache_cfg,
        rag=rag_cfg,
        log=log_cfg,
    )
