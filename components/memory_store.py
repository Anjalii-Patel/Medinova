# components/memory_store.py
from redis import Redis
import json
from datetime import datetime

redis = Redis(host="localhost", port=6379, decode_responses=True)

def get_session_key(session_id: str) -> str:
    return f"session:{session_id}"

def get_memory(session_id: str):
    key = get_session_key(session_id)
    data = redis.get(key)
    if data:
        return json.loads(data)
    
    # Initialize new session memory
    memory = {
        "session_id": session_id,
        "created": str(datetime.now()),
        "symptoms": [],
        "duration": None,
        "triggers": None,
        "messages": []
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
        sessions.append({
            "session_id": session["session_id"],
            "created": session["created"],
            "preview": session["messages"][0]["text"] if session["messages"] else ""
        })
    return sessions
