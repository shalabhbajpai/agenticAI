from typing import TypedDict, List, Annotated
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import operator
import re

# === Helper to clean LLM output ===
def extract_json_string(text: str) -> str:
    """Extract JSON from LLM response, even if wrapped in markdown or code fences."""
    return re.sub(r"```(?:json)?|```", "", text).strip()

# === Schema for LLM response ===
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Feedback on the cover letter")
    score: int = Field(ge=0, le=10, description="Score out of 10")

# === State to hold all data during evaluation ===
class CoverLetterState(TypedDict):
    cover_letter: str
    professionalism_feedback: str
    relevance_feedback: str
    clarity_feedback: str
    individual_scores: Annotated[List[int], operator.add]
    overall_feedback: str
    average_score: float

# === LLM Setup ===
#llm = ChatOllama(model="llama3.2:latest")
llm = ChatOpenAI()

# === Prompt Templates ===
professionalism_prompt = PromptTemplate(
    template="""
    You are an AI assistant tasked with evaluating the professionalism of a cover letter.
    Provide feedback on the cover letter's professionalism. 
    Return your response as JSON with "feedback" (string) and "score" (integer 0-10) fields.
    Here is the cover letter:
    {cover_letter}
    """,
    input_variables=["cover_letter"],
)

relevance_prompt = PromptTemplate(
    template="""
    You are an AI assistant tasked with evaluating the relevance of a cover letter.
    Provide feedback on the cover letter's relevance.
    Return your response as JSON with "feedback" (string) and "score" (integer 0-10) fields.
    Here is the cover letter:
    {cover_letter}
    """,
    input_variables=["cover_letter"],
)

clarity_prompt = PromptTemplate(
    template="""
    You are an AI assistant tasked with evaluating the clarity of a cover letter.
    Provide feedback on the cover letter's clarity.
    Return your response as JSON with "feedback" (string) and "score" (integer 0-10) fields.
    Here is the cover letter:
    {cover_letter}
    """,
    input_variables=["cover_letter"],
)

# === Evaluation Functions ===
def evaluate_professionalism(state: CoverLetterState) -> dict:
    response = llm.invoke(professionalism_prompt.format(cover_letter=state["cover_letter"]))
    raw_json = extract_json_string(response.content)
    evaluation = EvaluationSchema.model_validate_json(raw_json)
    return {
        "professionalism_feedback": evaluation.feedback,
        "individual_scores": [evaluation.score]
    }

def evaluate_relevance(state: CoverLetterState) -> dict:
    response = llm.invoke(relevance_prompt.format(cover_letter=state["cover_letter"]))
    raw_json = extract_json_string(response.content)
    evaluation = EvaluationSchema.model_validate_json(raw_json)
    return {
        "relevance_feedback": evaluation.feedback,
        "individual_scores": [evaluation.score]
    }

def evaluate_clarity(state: CoverLetterState) -> dict:
    response = llm.invoke(clarity_prompt.format(cover_letter=state["cover_letter"]))
    raw_json = extract_json_string(response.content)
    evaluation = EvaluationSchema.model_validate_json(raw_json)
    return {
        "clarity_feedback": evaluation.feedback,
        "individual_scores": [evaluation.score]
    }

def summarize(state: CoverLetterState) -> dict:
    combined_feedback = "\n".join([
        state.get("professionalism_feedback", ""),
        state.get("relevance_feedback", ""),
        state.get("clarity_feedback", "")
    ])
    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])
    return {
        "overall_feedback": combined_feedback,
        "average_score": avg_score
    }

# === Build the Workflow ===
graph = StateGraph(CoverLetterState)

# Add Nodes
graph.add_node("evaluate_professionalism", evaluate_professionalism)
graph.add_node("evaluate_relevance", evaluate_relevance)
graph.add_node("evaluate_clarity", evaluate_clarity)
graph.add_node("summarize", summarize)

# Add Edges
graph.add_edge(START, "evaluate_professionalism")
graph.add_edge(START, "evaluate_relevance")
graph.add_edge(START, "evaluate_clarity")

graph.add_edge("evaluate_professionalism", "summarize")
graph.add_edge("evaluate_relevance", "summarize")
graph.add_edge("evaluate_clarity", "summarize")

graph.add_edge("summarize", END)

# Compile the graph
workflow = graph.compile()

# === Run the Workflow ===
if __name__ == "__main__":
    initial_state = {
        "cover_letter": "I am passionate about software development and have 2 years of experience in Python."
    }

    result = workflow.invoke(initial_state)

    print("=== Overall Feedback ===")
    print(result["overall_feedback"])
    print("\n=== Average Score ===")
    print(result["average_score"])
