"""Agent 1 - Search agent.

A ReAct agent given the `web_search` tool. It decides when to call
Tavily and returns recent web information as text.
"""

from langchain.agents import create_agent
from ..tools import web_search
from .llm import llm, with_resilience


def build_search_agent():
    # Wrap the whole agent so a transient upstream 504 retries the run
    # instead of crashing the pipeline.
    return with_resilience(create_agent(
        model=llm,
        tools=[web_search],
    ))
