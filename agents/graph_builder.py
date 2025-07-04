# agent/graph_builder.py
from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable
from typing import Dict, List, Optional, TypedDict
from components.vector_store import query_faiss
from components.llm_ollama import query_ollama
from components.memory_store import get_memory, save_memory
import re

class BotState(TypedDict, total=False):
    session_id: str
    input: str
    memory: dict
    docs: List[str]
    response: str
    followup_required: bool

def load_memory(state: BotState) -> BotState:
    sid = state.get("session_id", "default")
    state["memory"] = get_memory(sid)
    return state

def retrieve_chunks(state: BotState) -> BotState:
    state["docs"] = query_faiss(state["input"])
    return state

def query_llm(state: BotState) -> BotState:
    prompt = build_medical_prompt(
        state["memory"]["symptoms"],
        state["input"],
        state["docs"]
    )
    state["response"] = query_ollama(prompt)
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
    text = state["input"]
    mem = state["memory"]

    match = re.search(r"(for|since)\s+(\d+\s+\w+)", text.lower())
    if match:
        mem["duration"] = match.group(2)

    for word in ["walking", "running", "exercise", "cold", "dust"]:
        if word in text.lower():
            mem["triggers"] = word
            break

    if text not in mem["symptoms"]:
        mem["symptoms"].append(text)

    # Save to Redis
    save_memory(state.get("session_id", "default"), mem)
    state["memory"] = mem
    return state

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
