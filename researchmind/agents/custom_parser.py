"""Custom output parser that handles JSON wrapped in markdown and other common issues."""

import json
import re
from typing import Any, TypeVar, Type
from pydantic import BaseModel, ValidationError, ConfigDict, field_serializer
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

T = TypeVar("T", bound=BaseModel)


class RobustPydanticParser(BaseOutputParser[T]):
    """
    A robust parser that:
    1. Extracts JSON from markdown code blocks (```json ... ```)
    2. Handles common JSON formatting issues
    3. Falls back to field extraction if JSON parsing fails
    4. Provides helpful error messages
    """

    pydantic_object: Type[T]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, pydantic_object: Type[T]):
        super().__init__(pydantic_object=pydantic_object)

    def parse_result(self, result: Any) -> T:
        """Parse the LLM result into a Pydantic model."""
        # Extract text from result
        if hasattr(result, "text"):
            text = result.text
        elif isinstance(result, list) and len(result) > 0:
            text = result[0].text if hasattr(result[0], "text") else str(result[0])
        else:
            text = str(result)

        # Try to extract and parse JSON
        json_obj = self._extract_json(text)
        if json_obj is None:
            raise OutputParserException(
                f"Could not extract valid JSON from LLM output.\n\nOutput:\n{text[:500]}",
                llm_output=text,
            )

        # Try to parse into Pydantic model
        try:
            return self.pydantic_object(**json_obj)
        except ValidationError as e:
            # If validation fails, try to fix common issues
            fixed_obj = self._fix_common_issues(json_obj)
            try:
                return self.pydantic_object(**fixed_obj)
            except ValidationError as e2:
                raise OutputParserException(
                    f"Failed to parse into {self.pydantic_object.__name__}: {str(e2)}",
                    llm_output=text,
                ) from e2

    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()

        # Try 1: Look for ```json ... ``` blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Try 2: Look for raw JSON object at start/end
        # Find first { and last }
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx : end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        first_brace = text.find("{")
        first_bracket = text.find("[")
        # Whether the top-level value is an object (starts with '{') rather than
        # a bare array. An inner array inside an object must NOT be mistaken for
        # the whole result — that would return a list and break `Model(**list)`.
        top_level_is_object = first_brace != -1 and (
            first_bracket == -1 or first_brace < first_bracket
        )

        # Try 3: The response was cut off mid-generation (common with the free
        # model hitting its output limit). Best-effort repair: close any open
        # string / brackets so the salvageable part of the object still parses.
        # Run this before bare-array extraction so a truncated object isn't
        # misread as the inner array it happens to contain.
        if top_level_is_object:
            repaired = self._repair_truncated_json(text)
            if repaired is not None:
                return repaired

        # Try 4: The whole output is a bare JSON array.
        end_idx = text.rfind("]")
        if first_bracket != -1 and end_idx != -1 and end_idx > first_bracket:
            json_str = text[first_bracket : end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        return None

    def _repair_truncated_json(self, text: str) -> dict | None:
        """Recover a JSON object that stopped mid-generation.

        Takes the text from the first ``{`` and closes whatever is still open
        (an unterminated string, then any open ``[``/``{``). If that doesn't
        parse, it trims back to the last structurally complete point (end of a
        string/bracket, or before a trailing comma) and retries.
        """
        start = text.find("{")
        if start == -1:
            return None
        s = text[start:]

        # Single pass: find what's open at the end, and record "safe" cut
        # points where the JSON could be closed cleanly.
        stack: list[str] = []
        in_string = False
        escape = False
        cut_points: list[int] = []
        for i, ch in enumerate(s):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                if not in_string:
                    cut_points.append(i + 1)  # just closed a string value
                continue
            if in_string:
                continue
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack:
                    stack.pop()
                cut_points.append(i + 1)
            elif ch == ",":
                cut_points.append(i)  # drop the comma and the incomplete tail

        candidates: list[str] = []

        # Strategy 1: close the currently-open string and brackets in place.
        repaired = s + ('"' if in_string else "")
        repaired += "".join(reversed(stack))
        candidates.append(repaired)

        # Strategy 2: trim back to a safe cut point (latest first) and re-close.
        for cut in list(reversed(cut_points))[:10]:
            prefix = s[:cut].rstrip().rstrip(",")
            depth: list[str] = []
            ins = esc = False
            for ch in prefix:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    ins = not ins
                    continue
                if ins:
                    continue
                if ch in "{[":
                    depth.append("}" if ch == "{" else "]")
                elif ch in "}]":
                    if depth:
                        depth.pop()
            candidates.append(prefix + "".join(reversed(depth)))

        for cand in candidates:
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        return None

    def _fix_common_issues(self, obj: dict) -> dict:
        """Fix common JSON issues that Pydantic might reject."""
        # Convert string lists to actual lists if needed
        for key, value in obj.items():
            if isinstance(value, str) and value.startswith("["):
                try:
                    obj[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass

            # Handle key_findings with embedded newlines and list markers
            if key == "key_findings" and isinstance(value, list):
                fixed_findings = []
                for item in value:
                    if isinstance(item, str):
                        # Split on bullet points but keep the content
                        cleaned = item.strip()
                        if cleaned and not cleaned.startswith("]"):
                            fixed_findings.append(cleaned)
                if fixed_findings:
                    obj[key] = fixed_findings

        return obj

    def parse(self, text: str) -> T:
        """Parse text directly (for string input)."""
        json_obj = self._extract_json(text)
        if json_obj is None:
            raise OutputParserException(
                f"Could not extract valid JSON from text.\n\nText:\n{text[:500]}",
                llm_output=text,
            )
        try:
            return self.pydantic_object(**json_obj)
        except ValidationError as e:
            fixed_obj = self._fix_common_issues(json_obj)
            try:
                return self.pydantic_object(**fixed_obj)
            except ValidationError as e2:
                raise OutputParserException(
                    f"Failed to parse into {self.pydantic_object.__name__}: {str(e2)}",
                    llm_output=text,
                ) from e2
