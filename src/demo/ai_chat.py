import os, sys
import argparse
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(root_path, "../"))

from config.config import load_config, AppConfig, AgentConfig
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from common.cli import cli_chat
def agent_handle(agent, agent_config: AgentConfig, user_query: str) -> str:
    res = agent.invoke({"messages": [HumanMessage(content=user_query)]},
                        config = {"max_iterations": agent_config.max_iterations,
                                    "handle_parsing_errors": agent_config.handle_parsing_errors,
                                    "return_intermediate_steps": agent_config.return_intermediate_steps,
                                    "max_execution_time": agent_config.max_execution_time,
                                    })
    reply_msg = res["messages"][-1]
    return reply_msg.text

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
    args = parser.parse_args()

    config = load_config(args.config)
    # init the agent
    (llm, agent) = init_agent(config)
    # chat with agent_handle
    cli_chat(agent = agent, agent_config = config.agent, chat_handler = agent_handle)