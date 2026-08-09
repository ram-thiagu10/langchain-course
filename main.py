import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # Load environment variables from .env file

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

stdio_server_params = StdioServerParameters(
    command = "python",
    args = ["C:/Users/Ramki/Documents/udemy_courses/mcp-crash-course/mcp-crash-course/servers/weather_server.py"],
)

async def main():
    print("Hello from mcp-crash-course!")


if __name__ == "__main__":
    asyncio.run(main())
