from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os, re, json
from langchain_core.runnables import RunnableLambda, RunnableSequence


class RouteQuery(BaseModel):
    """Route the user query to appropriate datasource"""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given the user query, choose the appropriate datasource is websearch or vectorstore.",
    )

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

structured_llm = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to the appropriate datasource. \n
Given a user question, choose the appropriate datasource is websearch or vectorstore. \n
The vectorstore is a collection of documents that are relevant to agents, prompt engineering, and adversarial attacks. \n
Use vectorstore for questions related to these topics. For all else, use websearch.\n"""

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system
        ),
        (
            "human",
            "{question}",
        ),
    ]
)

question_router = router_prompt | structured_llm