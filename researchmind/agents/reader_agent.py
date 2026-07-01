"""Agent 2 - Reader agent.

A ReAct agent given the `scrape_url` tool. It picks the most relevant
URL from the search output and scrapes deeper content.
"""

from langchain.agents import create_agent
from ..tools import scrape_url
from .llm import llm, with_resilience


def build_reader_agent():
    # Wrap the whole agent so a transient upstream 504 retries the run
    # instead of crashing the pipeline.
    return with_resilience(create_agent(
        model=llm,
        tools=[scrape_url],
    ))
