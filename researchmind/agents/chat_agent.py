"""Agent 5 - Chat chain.

An LCEL chain that answers questions using ONLY the provided context
(used for chatting over stored reports / RAG) with structured JSON output.
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from .custom_parser import RobustPydanticParser
from .llm import llm


class ChatResponse(BaseModel):
    """Structured response to a user question."""
    answer: str = Field(description="The answer to the user's question based only on provided context")

parser = RobustPydanticParser(pydantic_object=ChatResponse)

chat_prompt = ChatPromptTemplate.from_template(
    """Use only this context to answer. Return only this JSON:

Context:
{context}

Question:
{question}

Return only: {{"answer": "your answer"}}"""
)

chat_chain = chat_prompt | llm | parser
