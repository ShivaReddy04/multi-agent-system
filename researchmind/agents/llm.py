"""Shared LLM configuration.

Primary model is GitHub Models (OpenAI gpt-4o-mini). When the GitHub daily
rate cap is hit, calls transparently fall back to Groq's free tier. Everything
is configured here once and imported by the agents/chains.
"""

import os
import openai
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from dotenv import load_dotenv

load_dotenv()


def _build(model: str, base_url: str, api_key: str | None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        # Give the writer enough room to finish its JSON report, or it stops
        # early and the report JSON gets cut off mid-string.
        max_tokens=3000,
        # Fail a stuck upstream call instead of hanging, and let the OpenAI
        # client retry HTTP-level 5xx/429 with backoff.
        timeout=90,
        max_retries=3,
    )


# --- Primary: GitHub Models (OpenAI gpt-4o-mini) ---
# OpenAI-compatible endpoint, authenticated with a GitHub token. Hosted, so it
# works on Render. Great quality but a tight daily rate cap — hence the Groq
# fallback below. Override the model via GITHUB_MODEL without a code change.
MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-4o-mini")
llm = _build(MODEL, "https://models.github.ai/inference", os.getenv("GITHUB_TOKEN"))

# --- Fallback: Groq free tier (more generous limits) ---
# Only wired up when a Groq key is present; otherwise we run on the primary
# alone. Used automatically when the GitHub primary errors (e.g. 429 daily cap).
_groq_key = os.getenv("GROQ_API_KEY")
fallback_llm = (
    _build(os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
           "https://api.groq.com/openai/v1", _groq_key)
    if _groq_key else None
)

# A chat runnable that transparently retries the primary on transient errors and
# switches to Groq when the primary keeps failing. Use this INSIDE chains:
#   writer_chain = writer_prompt | chat_llm | parser
# (Agents need a raw model for create_agent, so they compose `llm` /
# `fallback_llm` directly — see build_search_agent / build_reader_agent.)
chat_llm: Runnable = llm.with_fallbacks([fallback_llm]) if fallback_llm else llm


# Transient failures worth retrying: upstream gateway timeouts (504 "operation
# was aborted"), rate limits, dropped connections, and the ValueError LangChain
# raises when a provider returns an error object in the response body instead of
# a completion (the OpenAI client's own retries miss it, since the HTTP status
# is 200). OutputParserException subclasses ValueError, so a garbled/truncated
# response also triggers a fresh generation.
_RETRYABLE_ERRORS = (openai.APIError, ValueError, ConnectionError, TimeoutError)


def with_resilience(runnable: Runnable, attempts: int = 4) -> Runnable:
    """Wrap a chain/agent so transient LLM failures are retried with backoff."""
    return runnable.with_retry(
        retry_if_exception_type=_RETRYABLE_ERRORS,
        wait_exponential_jitter=True,
        stop_after_attempt=attempts,
    )
