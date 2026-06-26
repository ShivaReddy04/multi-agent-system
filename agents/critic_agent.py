"""Agent 4 - Critic chain.

An LCEL chain that evaluates the report and returns a structured
score + feedback.
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from .custom_parser import RobustPydanticParser
from .llm import llm


class CriticReview(BaseModel):
    """Structured critic review of a research report."""
    score: int = Field(description="Score out of 10 (1-10)")
    strengths: List[str] = Field(default_factory=list, description="List of strengths of the report")
    areas_to_improve: List[str] = Field(default_factory=list, description="List of areas that need improvement")
    verdict: Optional[str] = Field(default="", description="One line overall verdict of the report quality")


def parse_score(feedback_obj):
    """Extract numeric score from critic feedback.

    Accepts either a CriticReview object or a string (for backward compatibility).
    Returns the score as an int, or None if not available.
    """
    if isinstance(feedback_obj, CriticReview):
        return feedback_obj.score
    if isinstance(feedback_obj, str):
        match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", feedback_obj, re.IGNORECASE)
        if match:
            return int(float(match.group(1)))
    return None

parser = RobustPydanticParser(pydantic_object=CriticReview)

critic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a critic. Respond ONLY with valid JSON, no other text."),
        ("human", """Review this report:

{report}

Return ONLY this JSON (no markdown, no code blocks):
{{"score": 8, "strengths": ["Strength 1"], "areas_to_improve": ["Area 1"], "verdict": "Good report"}}"""),
    ]
)

critic_chain = critic_prompt | llm | parser
