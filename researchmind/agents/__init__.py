"""Agents package.

Each agent/chain lives in its own file for easy reading and editing:

    agents/llm.py            -> shared LLM config
    agents/search_agent.py   -> build_search_agent  (web search)
    agents/reader_agent.py   -> build_reader_agent  (scrape URL)
    agents/writer_agent.py   -> writer_chain        (drafts report)
    agents/critic_agent.py   -> critic_chain        (reviews report)
    agents/chat_agent.py     -> chat_chain          (Q&A over context)

Everything is re-exported here so existing imports keep working, e.g.:

    from agents import build_search_agent, writer_chain, chat_chain
"""

from .llm import llm
from .search_agent import build_search_agent
from .reader_agent import build_reader_agent
from .writer_agent import writer_chain, ResearchReport
from .critic_agent import critic_chain, parse_score, CriticReview
from .chat_agent import chat_chain, ChatResponse

__all__ = [
    "llm",
    "build_search_agent",
    "build_reader_agent",
    "writer_chain",
    "ResearchReport",
    "critic_chain",
    "CriticReview",
    "parse_score",
    "chat_chain",
    "ChatResponse",
    "chat_chain",
]
