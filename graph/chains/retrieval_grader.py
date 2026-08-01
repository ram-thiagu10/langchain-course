import json
import os
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def parse_grade_documents(value: Any) -> GradeDocuments:
    """Accept either JSON-like structured output or plain yes/no text."""
    if isinstance(value, GradeDocuments):
        return value

    if isinstance(value, dict):
        return GradeDocuments(**value)

    if hasattr(value, "content"):
        value = value.content

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty grade response")

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE).strip()

        if text.startswith("{"):
            payload = json.loads(text)
            return GradeDocuments(**payload)

        normalized = text.lower()
        if normalized in {"yes", "no"}:
            return GradeDocuments(binary_score=normalized)

        if "safety" in normalized or "unsafe" in normalized:
            return GradeDocuments(binary_score="no")

    raise ValueError(f"Could not parse grade response: {value!r}")


system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Return exactly one JSON object with the schema {{"binary_score": "yes"}} or {{"binary_score": "no"}}."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | llm | RunnableLambda(parse_grade_documents)