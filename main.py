from dotenv import load_dotenv
import os
load_dotenv()
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response"""
    answer: str = Field(description="The answer from the agent")
    sources: List[Source] = Field(default_factory = list, description="The sources used by the agent")

# from tavily import TavilyClient

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Search for a query and return the results.
#     Args:
#         query (str): The search query.
#     Returns:
#         str: The search results.
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query=query)
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
    )
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
def main():
    print("Hello from langchain-course!")

    response = agent.invoke({
        "messages": [
            HumanMessage(content="search for 3 job opening for 5+ years ai engineer in hyderabad"),
        ]
    })
    print("Agent response:", response)
    

if __name__ == "__main__":
    main()
