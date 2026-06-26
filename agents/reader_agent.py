"""Agent 2 - Reader agent.

A ReAct agent given the `scrape_url` tool. It picks the most relevant
URL from the search output and scrapes deeper content.
"""

from langchain.agents import create_agent
from tools import scrape_url
from .llm import llm


def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
    )
