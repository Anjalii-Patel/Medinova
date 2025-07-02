from redis import Redis
import json

redis = Redis(host="localhost", port=6379, decode_responses=True)

def get_memory(session_id: str):
    data = redis.get(session_id)
    if data:
        return json.loads(data)
    return {"symptoms": [], "duration": None, "triggers": None}

def save_memory(session_id: str, memory: dict):
    redis.set(session_id, json.dumps(memory))
