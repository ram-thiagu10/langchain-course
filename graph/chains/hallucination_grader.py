import json
import os
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

class GradeHallucination(BaseModel):
    """Binary score for hallucination check on generated answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


def parse_hallucination_grade(value: Any) -> GradeHallucination:
    """Accept plain text, JSON, or fenced JSON responses for the hallucination grader."""
    if isinstance(value, GradeHallucination):
        return value

    if isinstance(value, dict):
        return GradeHallucination(**value)

    if hasattr(value, "content"):
        value = value.content

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty hallucination grade response")

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE).strip()

        if text.startswith("{"):
            payload = json.loads(text)
            return GradeHallucination(**payload)

        normalized = text.lower()
        if normalized in {"yes", "no"}:
            return GradeHallucination(binary_score=normalized)

        if "safety" in normalized or "unsafe" in normalized:
            return GradeHallucination(binary_score="no")

    raise ValueError(f"Could not parse hallucination grade response: {value!r}")


system = """You are a grader assessing whether the answer is grounded in /supported by the set of documents. \n
Give a binary score of 'yes' if the answer is supported by the documents, and 'no' if it is not. \n
Return exactly one JSON object with the schema {{"binary_score": "yes"}} or {{"binary_score": "no"}}.
"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system
        ),
        (
            "human",
            "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"
        ),
    ]
)


hallucination_grader: RunnableSequence = hallucination_prompt | llm | RunnableLambda(parse_hallucination_grade)
