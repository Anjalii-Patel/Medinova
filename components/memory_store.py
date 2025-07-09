# components/memory_store.py
from redis import Redis
import json
from datetime import datetime
from components.vector_store import delete_faiss_index
import os

UPLOAD_FOLDER = "uploads"  # Set this to your actual uploads directory

redis = Redis(host="localhost", port=6379, decode_responses=True)

def get_session_key(session_id: str) -> str:
    return f"session:{session_id}"

def get_memory(session_id: str):
    key = get_session_key(session_id)
    data = redis.get(key)

    if data:
        memory = json.loads(data)
        # Ensure required keys exist
        memory.setdefault("session_id", session_id)
        memory.setdefault("created", str(datetime.now()))
        memory.setdefault("symptoms", [])
        memory.setdefault("duration", None)
        memory.setdefault("triggers", None)
        memory.setdefault("messages", [])
        memory.setdefault("uploaded_files", [])
        return memory

    # New session
    memory = {
        "session_id": session_id,
        "created": str(datetime.now()),
        "symptoms": [],
        "duration": None,
        "triggers": None,
        "messages": [],
        "uploaded_files": []
    }
    redis.set(key, json.dumps(memory))
    return memory

def save_memory(session_id: str, memory: dict):
    key = get_session_key(session_id)
    redis.set(key, json.dumps(memory))

def list_sessions():
    keys = redis.keys("session:*")
    sessions = []

    for key in sorted(keys):
        session = json.loads(redis.get(key))
        preview = "No messages"
        if session.get("messages"):
            for msg in session["messages"]:
                if msg["role"] == "user":
                    preview = msg["text"]
                    break
        sessions.append({
            "session_id": session["session_id"],
            "created": session.get("created"),
            "preview": preview
        })
    return sessions

def delete_session(session_id: str):
    key = get_session_key(session_id)
    redis.delete(key)

def get_memory(session_id: str):
    key = get_session_key(session_id)
    data = redis.get(key)
    if data:
        return json.loads(data)

    memory = {
        "session_id": session_id,
        "created": str(datetime.now()),
        "symptoms": [],
        "duration": None,
        "triggers": None,
        "messages": [],
        "documents": []
    }
    redis.set(key, json.dumps(memory))
    return memory

def delete_session(session_id: str):
    redis.delete(get_session_key(session_id))
    # Delete FAISS
    delete_faiss_index(f"{session_id}.faiss")
    # Delete uploaded files
    for f in os.listdir(UPLOAD_FOLDER):
        if f.startswith(session_id + "_"):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
