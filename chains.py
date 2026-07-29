from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet"
            "Always provide detailed recommendations, including requests for length, virality, style, etc."
        ),
        MessagesPlaceholder(variable_name = "messages")
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writting excellent twitter posts."
            "Generate a best twitter post possible for user's request."
            "If the user provides critique, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name = "messages")
    ]
)

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature = 0
)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm