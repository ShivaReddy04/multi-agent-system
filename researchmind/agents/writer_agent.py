"""Agent 3 - Writer chain.

An LCEL chain that drafts a structured research report from the combined
research using Pydantic models and JsonOutputParser.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from .custom_parser import RobustPydanticParser
from .llm import llm


class ResearchReport(BaseModel):
    """Structured research report with sections."""
    introduction: str = Field(description="Introduction section explaining the topic and scope")
    key_findings: List[str] = Field(description="List of key findings from the research (minimum 3)")
    conclusion: Optional[str] = Field(default="", description="Conclusion summarizing the research")
    sources: Optional[List[str]] = Field(default_factory=list, description="List of source URLs referenced in the report")

parser = RobustPydanticParser(pydantic_object=ResearchReport)

writer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert research writer. Respond ONLY with valid JSON, no other text."),
        ("human", """Write a detailed research report on the topic: {topic}

Research Gathered:
{research}

Return ONLY this JSON structure (no markdown, no code blocks):
{{"introduction": "2-3 sentence intro", "key_findings": ["Finding 1", "Finding 2", "Finding 3"], "conclusion": "Summary", "sources": ["url1", "url2"]}}"""),
    ]
)

writer_chain = writer_prompt | llm | parser
