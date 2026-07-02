"""Agent 2 - Reader agent.

A ReAct agent given the `scrape_url` tool. It picks the most relevant
URL from the search output and scrapes deeper content.
"""

from langchain.agents import create_agent
from ..tools import scrape_url
from .llm import llm, fallback_llm, with_resilience


def build_reader_agent():
    # Primary agent on GitHub Models; if it errors (e.g. daily cap), fall back
    # to a Groq-backed agent. Wrap the whole thing so a transient 504 retries
    # the run instead of crashing the pipeline.
    agent = create_agent(model=llm, tools=[scrape_url])
    if fallback_llm is not None:
        agent = agent.with_fallbacks([create_agent(model=fallback_llm, tools=[scrape_url])])
    return with_resilience(agent)
