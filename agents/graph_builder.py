# agents/graph_builder.py
from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable
from typing import Dict, List, Optional, TypedDict
from components.vector_store import query_faiss, get_all_chunks, query_chat_faiss
from components.llm_ollama import query_ollama
from components.memory_store import get_memory
import re

class BotState(TypedDict, total=False):
    session_id: str
    input: str
    memory: dict
    chat_context: List[str]
    docs: List[str]
    response: str
    followup_required: bool

def load_memory(state: BotState) -> BotState:
    sid = state.get("session_id", "default")
    memory = get_memory(sid)
    state["memory"] = memory
    return state

def retrieve_chunks(state: BotState) -> BotState:
    session_id = state.get("session_id", "default")
    question = state["input"].lower()

    # Retrieve relevant previous chat messages
    relevant_chats = query_chat_faiss(session_id, question, top_k=5)
    print(f"[FAISS] Retrieved {len(relevant_chats)} relevant chat messages for session {session_id}")
    print("[FAISS] Relevant chat messages:", relevant_chats)
    state["chat_context"] = relevant_chats

    # If summarization-type command, use recursive summarization
    if any(cmd in question for cmd in ["summarize", "summarise", "explain", "analyze", "extract"]):
        from components.document_loader import recursive_summarize, summarize_chunks_with_llm
        all_chunks = get_all_chunks(f"{session_id}.faiss")
        summary = recursive_summarize(all_chunks, summarize_chunks_with_llm, max_chunks_per_pass=8)
        state["docs"] = [summary]
        return state
    else:
        chunks = query_faiss(question, index_name=f"{session_id}.faiss")
        # Fallback to full doc text if no context found
        if not chunks:
            chunks = get_all_chunks(f"{session_id}.faiss")
        state["docs"] = chunks
        return state

def build_medical_prompt(input_text, docs, chat_context=None):
    doc_context = "\n".join(docs) if docs else "No context found."

    # Filter duplicate or irrelevant messages
    filtered_chat = []
    seen = set()
    input_clean = input_text.strip().lower()

    for msg in (chat_context or []):
        msg_clean = msg.strip().lower()
        if msg_clean and msg_clean != input_clean and msg_clean not in seen:
            filtered_chat.append(msg)
            seen.add(msg_clean)

    # Fallback: ensure at least 2 recent messages are included
    if not filtered_chat and chat_context:
        filtered_chat = chat_context[-2:]  # last 2 messages

    chat_history = "\n".join(filtered_chat) if filtered_chat else "None"

    # DEBUG LOGGING
    print("[Prompt Debug] Final Chat Context for Prompt:")
    for i, msg in enumerate(filtered_chat):
        print(f"  {i+1}. {msg}")

    return f"""
        You are a helpful and friendly medical assistant.

        Always respond in a warm, conversational tone.

        ---

        User input: \"{input_text}\"

        Relevant previous chat history:
        {chat_history}

        Context from uploaded medical documents:
        {doc_context}

        ---

        Instructions:
        1. If the user's input is a **task-oriented command** like \"summarize\", \"explain\", or \"extract\", perform that task using the context above.
        2. If the user greets you or says something vague (e.g., \"hello\", \"how are you?\", \"I need help\"), then politely ask them to describe their medical symptoms.
        3. If the user is already describing symptoms, respond helpfully and ask relevant follow-up questions.
        4. Never ask for symptoms unless you're sure the user hasn’t shared any and isn’t giving a task-oriented command.

        Only respond based on the user input, previous chat, and document context. Do not make up medical information.
    """

def query_llm(state: BotState) -> BotState:
    prompt = build_medical_prompt(
        state["input"],
        state.get("docs", []),
        state.get("chat_context", [])
    )
    print("[LLM Input Debug] Chat context keys in state:", state.get("chat_context"))
    state["response"] = query_ollama(prompt)
    return state

def decide_followup(state: BotState) -> BotState:
    # Remove manual followup message, let LLM decide
    state["followup_required"] = False
    return state

def update_memory(state: BotState) -> BotState:
    text = state["input"]
    mem = state["memory"]

    # Extract duration
    match = re.search(r"(for|since)\s+(\d+\s+\w+)", text.lower())
    if match:
        mem["duration"] = match.group(2)

    # Detect triggers
    for word in ["walking", "running", "exercise", "cold", "dust"]:
        if word in text.lower():
            mem["triggers"] = word
            break

    state["memory"] = mem
    return state

def build_graph() -> Runnable:
    graph = StateGraph(BotState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("retrieve", retrieve_chunks)
    graph.add_node("llm", query_llm)
    graph.add_node("followup_logic", decide_followup)
    graph.add_node("update_memory", update_memory)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "retrieve")
    graph.add_edge("retrieve", "update_memory")
    graph.add_edge("update_memory", "llm")
    graph.add_edge("llm", "followup_logic")
    graph.set_finish_point("followup_logic")
    return graph.compile()
