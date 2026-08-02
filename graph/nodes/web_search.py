import json
import os
import sys
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_tavily import TavilySearch
# from langchain_community.tools.tavily_search import TavilySearchResults

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def _normalize_search_results(results: Any) -> List[str]:
    """Convert Tavily search results into a flat list of text snippets."""
    if not results:
        return []

    if isinstance(results, str):
        return [results]

    if isinstance(results, dict):
        if isinstance(results.get("results"), list):
            results = results["results"]
        elif isinstance(results.get("content"), str):
            return [results["content"]]
        else:
            return [json.dumps(results)]

    if not isinstance(results, list):
        return [str(results)]

    normalized: List[str] = []
    for item in results:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                normalized.append(content)
            else:
                title = item.get("title")
                if isinstance(title, str) and title.strip():
                    normalized.append(title)
                else:
                    normalized.append(json.dumps(item))
        else:
            normalized.append(str(item))

    return normalized


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents")

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_result = "\n".join(_normalize_search_results(tavily_results))
    web_results = Document(page_content=joined_tavily_result)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


if __name__ == "__main__":
     web_search(state={"question": "agent memory", "documents": None})