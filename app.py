# app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from components.document_loader import load_document, chunk_text
from components.vector_store import embed_chunks, save_faiss_index
from components.asr import transcribe_audio, stream_asr
from agents.graph_builder import build_graph
import os, json, uuid
from datetime import datetime

UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "audio"
MEMORY_FOLDER = "memory_store"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(MEMORY_FOLDER, exist_ok=True)


graph = build_graph()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# @app.get("/", response_class=HTMLResponse)
# def index():
#     with open("frontend/index.html") as f:
#         return f.read()

# Mount the static folder
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Serve the main HTML page
@app.get("/")
def get_index():
    return FileResponse("frontend/index.html")

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    path = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    text = load_document(path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    save_faiss_index(embeddings, chunks)
    return {"status": "uploaded and ingested"}

@app.post("/ask")
async def ask(question: str = Form(...), session_id: str = Form("default")):
    if not os.path.exists("vector_store/faiss_index") or not os.path.exists("vector_store/metadata.pkl"):
        return {"error": "Please upload a document before asking questions."}

    # Fresh memory creation if session is new
    memory_path = os.path.join(MEMORY_FOLDER, f"session_{session_id}.json")
    if not os.path.exists(memory_path):
        with open(memory_path, "w") as f:
            json.dump({
                "session_id": session_id,
                "created": str(datetime.now()),
                "messages": []
            }, f)

    result = graph.invoke({"input": question, "session_id": session_id})
    response_text = result["response"]

    # Append messages
    with open(memory_path, "r") as f:
        history = json.load(f)
    history["messages"].append({"role": "user", "text": question})
    history["messages"].append({"role": "bot", "text": response_text})
    with open(memory_path, "w") as f:
        json.dump(history, f, indent=2)

    return {
        "response": response_text,
        "followup": result.get("followup_required", False)
    }

@app.get("/chats")
def list_chats():
    files = [f for f in os.listdir(MEMORY_FOLDER) if f.startswith("session_")]
    sessions = []
    for file in sorted(files):
        with open(os.path.join(MEMORY_FOLDER, file)) as f:
            data = json.load(f)
            sessions.append({
                "session_id": data["session_id"],
                "created": data["created"],
                "preview": data["messages"][0]["text"] if data["messages"] else ""
            })
    return sessions

@app.get("/chat/{session_id}")
def get_chat(session_id: str):
    path = os.path.join(MEMORY_FOLDER, f"session_{session_id}.json")
    if not os.path.exists(path):
        return {"error": "Session not found."}
    with open(path) as f:
        return json.load(f)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    path = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"transcription": transcribe_audio(path)}

@app.websocket("/ws/asr")
async def websocket_asr_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await stream_asr(websocket)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
