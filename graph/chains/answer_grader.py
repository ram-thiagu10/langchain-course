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

class GradeAnswer(BaseModel):
    """Binary score for answer check on generated answer."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def parse_answer_grade(value: Any) -> GradeAnswer:
    """Accept plain text, JSON, or fenced JSON responses for the answer grader."""
    if isinstance(value, GradeAnswer):
        return value

    if isinstance(value, dict):
        return GradeAnswer(**value)

    if hasattr(value, "content"):
        value = value.content

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty answer grade response")

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE).strip()

        if text.startswith("{"):
            payload = json.loads(text)
            return GradeAnswer(**payload)

        normalized = text.lower()
        if normalized in {"yes", "no"}:
            return GradeAnswer(binary_score=normalized)

        if "safety" in normalized or "unsafe" in normalized:
            return GradeAnswer(binary_score="no")

    raise ValueError(f"Could not parse answer grade response: {value!r}")


system = """You are a grader assessing whether the answer addresses / resolves a question. \n
Give a binary score of 'yes' if the answer addresses the question, and 'no' if it is not. \n
Return exactly one JSON object with the schema {{"binary_score": "yes"}} or {{"binary_score": "no"}}.
"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system
        ),
        (
            "human",
            "User question: \n\n {question} \n\n LLM generation: {generation}"
        ),
    ]
)


answer_grader: RunnableSequence = answer_prompt | llm | RunnableLambda(parse_answer_grade)
