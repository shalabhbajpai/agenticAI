from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- Tools ---
def retrieve_places(country: str) -> str:
    catalog = {
        "india": ["Goa", "Manali", "Jaipur"],
        "france": ["Paris", "Nice", "Lyon"],
    }
    return f"Top places in {country}: {', '.join(catalog.get(country.lower(), ['No data']))}"

def weather_info(city: str) -> str:
    return f"The weather in {city} is sunny 28°C (dummy)."

# --- State ---
class TravelState(Dict[str, Any]):
    messages: List
    preferences: List[str] = []

# --- Router ---
def router(state: TravelState) -> str:
    last_msg = state["messages"][-1].content.lower()

    # store preference
    if "i like" in last_msg:
        pref = last_msg.split("i like")[-1].strip()
        state.setdefault("preferences", []).append(pref)
        return "llm_node"

    if "weather" in last_msg:
        return "weather_node"

    if "recommend" in last_msg or "places" in last_msg:
        return "places_node"

    # default: stop instead of looping forever
    return "end"

# --- Nodes ---
llm = ChatOllama(model="llama3.2:latest")

def llm_node(state: TravelState) -> Dict[str, List[AIMessage]]:
    prefs = ", ".join(state.get("preferences", [])) or "none"
    user_msg = state["messages"][-1]
    ai_msg = llm.invoke([
        user_msg,
        AIMessage(content=f"(Your preferences so far: {prefs})")
    ])
    return {"messages": [ai_msg]}

def weather_node(state: TravelState) -> Dict[str, List[AIMessage]]:
    text = state["messages"][-1].content.lower()
    # naive city detection = last word
    city = text.split()[-1]
    return {"messages": [AIMessage(content=weather_info(city))]}

def places_node(state: TravelState) -> Dict[str, List[AIMessage]]:
    text = state["messages"][-1].content.lower()
    countries = ["india", "france"]
    country = None
    for c in countries:
        if c in text:
            country = c
            break
    if country:
        msg = retrieve_places(country)
    else:
        # fallback: let LLM handle it
        prefs = ", ".join(state.get("preferences", [])) or "none"
        msg = llm.invoke([
            state["messages"][-1],
            AIMessage(content=f"(No country match, your preferences so far: {prefs})")
        ]).content
    return {"messages": [AIMessage(content=msg)]}

# --- Graph ---
workflow = StateGraph(TravelState)

workflow.add_node("llm_node", llm_node)
workflow.add_node("weather_node", weather_node)
workflow.add_node("places_node", places_node)

workflow.set_entry_point("llm_node")

workflow.add_conditional_edges("llm_node", router, {
    "llm_node": "llm_node",
    "weather_node": "weather_node",
    "places_node": "places_node",
    "end": END
})

workflow.add_edge("weather_node", END)
workflow.add_edge("places_node", END)

memory = MemorySaver()
workflow = workflow.compile(checkpointer=memory)
