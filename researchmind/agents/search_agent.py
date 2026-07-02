"""Agent 1 - Search agent.

A ReAct agent given the `web_search` tool. It decides when to call
Tavily and returns recent web information as text.
"""

from langchain.agents import create_agent
from ..tools import web_search
from .llm import llm, fallback_llm, with_resilience


def build_search_agent():
    # Primary agent on GitHub Models; if it errors (e.g. daily cap), fall back
    # to a Groq-backed agent. Wrap the whole thing so a transient 504 retries
    # the run instead of crashing the pipeline.
    agent = create_agent(model=llm, tools=[web_search])
    if fallback_llm is not None:
        agent = agent.with_fallbacks([create_agent(model=fallback_llm, tools=[web_search])])
    return with_resilience(agent)
