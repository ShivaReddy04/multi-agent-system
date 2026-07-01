"""Shared LLM configuration.

Every agent/chain imports the same `llm` from here so the model is
configured in exactly one place.
"""

import os
import openai
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from dotenv import load_dotenv

load_dotenv()

# Pin a concrete free model instead of the generic "openrouter/free" router,
# which hops between whatever free model is available and is frequently
# overloaded (the source of the 504s). Override via OPENROUTER_MODEL without a
# code change if this one gets rate-limited or retired.
MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

llm = ChatOpenAI(
    model=MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    # Give the writer enough room to finish its JSON report. Without this the
    # free model stops early and the report JSON gets cut off mid-string,
    # which the parser then can't read.
    max_tokens=3000,
    # Fail a stuck upstream call instead of hanging the whole request, and let
    # the OpenAI client retry HTTP-level 5xx/429 with backoff.
    timeout=90,
    max_retries=3,
)


# Transient failures worth retrying: OpenRouter/upstream gateway timeouts
# (the 504 "operation was aborted"), rate limits, dropped connections, and the
# ValueError LangChain raises when OpenRouter returns an error object in the
# response body instead of a completion (which the OpenAI client's own retries
# don't catch, since the HTTP status is 200). OutputParserException subclasses
# ValueError, so a garbled/truncated response also triggers a fresh generation.
_RETRYABLE_ERRORS = (openai.APIError, ValueError, ConnectionError, TimeoutError)


def with_resilience(runnable: Runnable, attempts: int = 4) -> Runnable:
    """Wrap a chain/agent so transient LLM failures are retried with backoff."""
    return runnable.with_retry(
        retry_if_exception_type=_RETRYABLE_ERRORS,
        wait_exponential_jitter=True,
        stop_after_attempt=attempts,
    )
