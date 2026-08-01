from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch
# from langchain_community.tools.tavily_search import TavilySearchResults
import sys, os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    # web_results = Document(page_content=tavily_results)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


if __name__ == "__main__":
     web_search(state={"question": "agent memory", "documents": None})