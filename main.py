from dotenv import load_dotenv
import os
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
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
    model_name="qwen/qwen3-32b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
    )
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)
def main():
    print("Hello from langchain-course!")

    response = agent.invoke({
        "messages": [
            HumanMessage(content="search for 3 job opening for ai engineer in hyderabad"),
        ]
    })
    print(f"Agent result: {response['messages'][-1].content}")
    

if __name__ == "__main__":
    main()
