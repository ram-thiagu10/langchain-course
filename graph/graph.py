from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.const import RETRIEVE, GRADE_DOCUMENTS, WEBSEARCH, GENERATE
from graph.nodes import retrieve, grade_documents, web_search, generate
from graph.state import GraphState

load_dotenv()

def decide_to_generate(state: GraphState) -> str:
    if state["web_search"]:
        print("Deciding to perform web search...")
        return WEBSEARCH
    else:
        print("Deciding to generate answer...")
        return GENERATE

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GENERATE, generate)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")