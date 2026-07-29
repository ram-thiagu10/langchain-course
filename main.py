from dotenv import load_dotenv
from typing import TypedDict, Annotated
import os
load_dotenv()
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from chains import generation_chain, reflection_chain

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]} 

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content = res.content)]}

def should_continue(state:MessageGraph):
    if len(state["messages"]) > 5:
        return END
    return REFLECT

builder = StateGraph(MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)
builder.add_conditional_edges(GENERATE, should_continue, {END:END, REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

if __name__ == "__main__":
    inputs = {"messages": [
        HumanMessage(
        
        content = """Make this twitter better:
        new news in the market.
        After long time this is happening.
        Never imagined that this will happen in this time.
        Hope you too will love to enjoy it.
        let's jump in and have fun.
"""
    )]}
    response = graph.invoke(inputs)
    print(response)
