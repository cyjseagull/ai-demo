import tomllib
from dataclasses import dataclass
from pydantic import SecretStr

@dataclass
class LLMConfig:
    model: str
    base_url: str
    api_key: SecretStr
    temperature: float

@dataclass
class AgentConfig:
    system_prompt: str
    debug: bool
    verbose: bool
    max_iterations: int
    max_execution_time: int
    handle_parsing_errors: bool
    return_intermediate_steps: bool

@dataclass
class AppConfig:
    llm: LLMConfig
    agent: AgentConfig
def load_config(config_path: str = "config.toml"):
    with open(config_path, "rb") as f:
        d = tomllib.load(f)

    return AppConfig(
        llm=LLMConfig(**d["llm"]),
        agent=AgentConfig(**d["agent"]),
    )