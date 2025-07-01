from components.llm_ollama import query_ollama

print("Welcome to Medinova!")
while True:
    q = input("You: ")
    if q.lower() == "exit":
        break
    reply = query_ollama(q)
    print("Bot:", reply)
