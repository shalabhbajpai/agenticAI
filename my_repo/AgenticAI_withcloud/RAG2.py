from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

# Define a simple retriever tool with logging to confirm usage
@tool
def retrieve_docs(query: str) -> str:
    """Retrieve relevant documents based on the query."""
    print(f"Tool 'retrieve_docs' called with query: {query}")
    # In real setup: Use vectorstore.as_retriever().invoke(query)
    return "Retrieved docs: Info about LangGraph is a library for building agentic apps."

# Tools list
tools = [retrieve_docs]

# LLM setup
llm = ChatOllama(model="llama3.2:latest", temperature=0)

# Agent state using BaseMessage and add_messages
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Create the ReAct agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful assistant. Use the retrieve_docs tool for questions about LangGraph."
)

# Define a node that calls the agent
def agent_node(state: AgentState) -> AgentState:
    print("Agent node called")
    result = agent.invoke(state)
    return result

# Define the LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)

workflow.set_entry_point("agent")

workflow.add_edge("agent", END)

# Compile the graph
app = workflow.compile()

# Run the agent with HumanMessage (via the graph)
input_data = {"messages": [HumanMessage(content="What is LangGraph?")]}
result = app.invoke(input_data)

# Inspect the full message history to confirm tool usage
print("Full message history:")
for message in result["messages"]:
    print(f"{message.type}: {message.content}")

print("\nFinal response:", result["messages"][-1].content)