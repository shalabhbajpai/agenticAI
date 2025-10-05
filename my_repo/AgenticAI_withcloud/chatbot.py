from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

llm = ChatOllama(model="llama3.2:latest")

def chat_node(state:ChatState) :
    # take user query from state
    messages = state['messages']
    #sending it to llm
    response = llm.invoke(messages)
    # response store into the state
    
    return {'messages': [response]}

checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START,"chat_node")

graph.add_edge("chat_node", END)

#graph.compile()

workflow = graph.compile(checkpointer=checkpointer)

#print(workflow)

#initial_state={
#   'messages': HumanMessage(content="What is the capital of France?")
#}

#print(workflow.invoke(initial_state)['messages'][-1].content)