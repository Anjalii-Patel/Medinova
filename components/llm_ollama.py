import requests

def query_ollama(prompt: str, model="mistral"):
    url = "http://localhost:11434/api/chat"
    response = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]
