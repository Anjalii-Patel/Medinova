from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable
from typing import Dict, List, Optional, TypedDict
from components.vector_store import query_faiss
from components.llm_ollama import query_ollama
import re

# --- Step 1: Define the state schema ---
class MemoryState(TypedDict, total=False):
    symptoms: List[str]
    duration: Optional[str]
    triggers: Optional[str]

class BotState(TypedDict, total=False):
    input: str
    memory: MemoryState
    docs: List[str]
    response: str
    followup_required: bool

# --- Step 2: Node implementations ---

def load_memory(state: BotState) -> BotState:
    state.setdefault("memory", {"symptoms": [], "duration": None, "triggers": None})
    return state

def retrieve_chunks(state: BotState) -> BotState:
    user_query = state["input"]
    docs = query_faiss(user_query)
    state["docs"] = docs
    return state

def query_llm(state: BotState) -> BotState:
    prompt = build_medical_prompt(
        symptoms=state["memory"]["symptoms"],
        input_text=state["input"],
        docs=state["docs"]
    )
    response = query_ollama(prompt)
    state["response"] = response
    return state

def decide_followup(state: BotState) -> BotState:
    mem = state["memory"]
    missing = []

    if not mem.get("duration"):
        missing.append("duration")
    if not mem.get("triggers"):
        missing.append("triggers")

    if missing:
        state["response"] += f"\n\nFollow-up: Could you tell me your symptom {', '.join(missing)}?"
        state["followup_required"] = True
    else:
        state["followup_required"] = False

    return state

def update_memory(state: BotState) -> BotState:
    input_text = state["input"]
    mem = state["memory"]

    # Simple pattern match for duration
    duration_match = re.search(r"(for|since)\s+(\d+\s+\w+)", input_text.lower())
    if duration_match:
        mem["duration"] = duration_match.group(2)

    # Basic keyword matching for triggers
    for word in ["walking", "running", "exercise", "cold", "dust"]:
        if word in input_text.lower():
            mem["triggers"] = word
            break

    # Add current input to symptoms
    if input_text not in mem["symptoms"]:
        mem["symptoms"].append(input_text)

    state["memory"] = mem
    return state

# --- Helper for prompt building ---
def build_medical_prompt(symptoms, input_text, docs):
    doc_context = "\n".join(docs)
    return f"""
You are a medical assistant. Based on the user's input and known symptoms, provide a possible explanation, ask relevant follow-up questions, and refer to the context if useful.

Known symptoms: {', '.join(symptoms) if symptoms else 'None yet'}
User just said: "{input_text}"

Context from medical docs:
{doc_context}

Now, respond conversationally and ask any necessary follow-up questions.
"""

# --- Graph Builder ---
def build_graph() -> Runnable:
    graph = StateGraph(BotState)

    graph.add_node("load_memory", load_memory)
    graph.add_node("retrieve", retrieve_chunks)
    graph.add_node("llm", query_llm)
    graph.add_node("followup_logic", decide_followup)
    graph.add_node("update_memory", update_memory)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "retrieve")
    graph.add_edge("retrieve", "llm")
    graph.add_edge("llm", "followup_logic")
    graph.add_edge("followup_logic", "update_memory")
    graph.set_finish_point("update_memory")

    return graph.compile()
