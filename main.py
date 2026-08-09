import asyncio
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
print("Environment variables loaded successfully.")
print("OPENROUTER_API_KEY:", os.getenv("OPENROUTER_API_KEY"))
async def main():
    print("Hello from mcp-crash-course!")


if __name__ == "__main__":
    asyncio.run(main())
