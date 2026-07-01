"""Agent 1 - Search agent.

A ReAct agent given the `web_search` tool. It decides when to call
Tavily and returns recent web information as text.
"""

from langchain.agents import create_agent
from ..tools import web_search
from .llm import llm


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
    )
