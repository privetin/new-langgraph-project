"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    model: Optional[str]


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str = "example"


async def graph(config: RunnableConfig):
    """Define the graph."""
    logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
    configuration = config["configurable"]
    model = init_chat_model(
        configuration.get("model", "google_genai:gemini-2.0-flash"),
    )
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["/Users/privetin/Projects/agent/src/agent/math.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(
        model=model,
        tools=tools,
        # state_schema=State,
        config_schema=Configuration,
        name="New Graph",
    )
    return agent
