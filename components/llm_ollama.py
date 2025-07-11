# components/llm_ollama.py
import requests

def query_ollama(prompt: str, model="medllama2"):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    print("[Query Ollama] Debugging information:", payload)

    try:
        response = requests.post(url, json=payload)
        data = response.json()

        print("\n==== RAW RESPONSE FROM OLLAMA ====")
        print(data)

        content = data.get("message", {}).get("content", "").strip()

        if content:
            return content
        elif data.get("done_reason") == "load":
            return "[Model is now loaded, please ask your question again.]"
        else:
            return "[No valid response from model. Try rephrasing your question.]"

    except Exception as e:
        print("[query_ollama] Exception:", e)
        return "Sorry, something went wrong while processing your request."
