"""Shared LLM configuration.

Every agent/chain imports the same `llm` from here so the model is
configured in exactly one place.
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    # Give the writer enough room to finish its JSON report. Without this the
    # free model stops early and the report JSON gets cut off mid-string,
    # which the parser then can't read.
    max_tokens=3000,
)
