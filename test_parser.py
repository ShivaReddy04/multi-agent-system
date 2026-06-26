#!/usr/bin/env python
"""Test custom parser with various JSON formats."""

from agents.custom_parser import RobustPydanticParser
from agents.writer_agent import ResearchReport
from agents.critic_agent import CriticReview

parser_report = RobustPydanticParser(pydantic_object=ResearchReport)
parser_critic = RobustPydanticParser(pydantic_object=CriticReview)

# Test 1: JSON in markdown code block
text1 = """```json
{
  "introduction": "Test intro",
  "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
  "conclusion": "Test conclusion",
  "sources": ["https://example.com"]
}
```"""

result1 = parser_report.parse(text1)
print("[PASS] Test 1 (markdown block):", result1.introduction[:20] + "...")

# Test 2: Raw JSON without markdown
text2 = '{"introduction": "Raw intro", "key_findings": ["F1", "F2"], "conclusion": "Done", "sources": []}'
result2 = parser_report.parse(text2)
print("[PASS] Test 2 (raw JSON):", result2.introduction)

# Test 3: JSON with extra text
text3 = """Here's the report:
{"introduction": "With text", "key_findings": ["F1"], "conclusion": "x", "sources": []}
Extra text here"""
result3 = parser_report.parse(text3)
print("[PASS] Test 3 (JSON with extra text):", result3.introduction)

# Test 4: CriticReview with minimal fields
text4 = '{"score": 7}'
result4 = parser_critic.parse(text4)
print("[PASS] Test 4 (minimal critic):", f"Score={result4.score}, strengths={len(result4.strengths)}")

print("\n[SUCCESS] All parser tests passed!")
