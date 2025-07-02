import requests

def query_ollama(prompt: str, model="mistral"):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = requests.post(url, json=payload)
    data = response.json()
    # Check for a valid assistant response
    if (
        "message" in data and
        isinstance(data["message"], dict) and
        "content" in data["message"] and
        data["message"]["content"].strip() != ""
    ):
        return data["message"]["content"].strip()
    elif "done_reason" in data and data["done_reason"] == "load":
        return "[Model loaded, please ask your question again.]"
    else:
        print("Unexpected response from Ollama API:", data)
        return "[Error: Unexpected response format or empty response from Ollama API]"
