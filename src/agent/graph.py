"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from agent.human import add_human_in_the_loop

checkpointer = InMemorySaver()


class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


@dataclass
class State(AgentState):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str = "example"


def _load_mcp_config() -> dict:
    mcp_config_path = Path(__file__).parent / "mcp.json"
    return json.loads(mcp_config_path.read_text())


async def graph(config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime configuration to alter behavior.
    """
    configuration = config["configurable"]

    mcp_config = await asyncio.to_thread(_load_mcp_config)
    servers = mcp_config.get("servers", {})
    for server in servers.values():
        server.setdefault("transport", "stdio")
    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()
    manned_tools = [add_human_in_the_loop(tool) for tool in tools]
    # Define the graph
    agent = create_react_agent(
        "",
        tools=manned_tools,
        state_schema=State,
        config_schema=Configuration,
        checkpointer=checkpointer,
        name="New Graph",
    )
    return agent
