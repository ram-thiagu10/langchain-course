import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


async def main():
    print("Hello from langchain_client.py!")


if __name__ == "__main__":
    asyncio.run(main())
