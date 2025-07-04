"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langmem import create_manage_memory_tool, create_search_memory_tool
from langmem.short_term import SummarizationNode


class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    model: Optional[str]


@dataclass
class State(AgentState):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    context: dict[str, Any]


async def graph(config: RunnableConfig):
    """Define the graph."""
    logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
    configuration = config["configurable"]
    model = init_chat_model(
        configuration.get("model", "google_genai:gemini-2.0-flash"),
    )
    summarization_node = SummarizationNode(
        token_counter=count_tokens_approximately,
        model=model,
        max_tokens=384,
        max_summary_tokens=128,
        output_messages_key="llm_input_messages",
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
        tools=[
            *tools,
            create_manage_memory_tool(namespace=("memories",)),
            create_search_memory_tool(namespace=("memories",)),
        ],
        pre_model_hook=summarization_node,
        state_schema=State,
        config_schema=Configuration,
        name="New Graph",
    )
    return agent
