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
from components.memory_store import get_memory, save_memory, list_sessions

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
    session = get_memory(session_id)
    result = graph.invoke({"input": question, "session_id": session_id})
    response_text = result["response"]

    # Update messages
    session["messages"].append({"role": "user", "text": question})
    session["messages"].append({"role": "bot", "text": response_text})
    save_memory(session_id, session)

    return {
        "response": response_text,
        "followup": result.get("followup_required", False)
    }

@app.get("/chats")
def list_chats():
    return list_sessions()

@app.get("/chat/{session_id}")
def get_chat(session_id: str):
    session = get_memory(session_id)
    return {"session_id": session_id, "messages": session.get("messages", [])}

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
