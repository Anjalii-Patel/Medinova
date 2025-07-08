# app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from components.document_loader import load_document, chunk_text
from components.vector_store import save_faiss_index, embed_chunks
from components.asr import transcribe_audio, stream_asr
from components.memory_store import get_memory, save_memory, list_sessions
from agents.graph_builder import build_graph

import os

# Setup folders
UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Initialize
graph = build_graph()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def get_index():
    return FileResponse("frontend/index.html")

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...), session_id: str = Form("default")):
    try:
        print("Received file:", file.filename)

        if not file.filename.endswith((".pdf", ".docx")):
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        
        path = f"{UPLOAD_FOLDER}/{session_id}_{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())
        
        print("Saved file to:", path)
        text = load_document(path)
        print("Extracted text length:", len(text))

        chunks = chunk_text(text)
        print("Number of chunks:", len(chunks))

        embeddings = embed_chunks(chunks)
        print("Shape of embeddings:", embeddings.shape)

        save_faiss_index(embeddings, chunks, index_name=f"{session_id}.faiss")
        return {"status": "uploaded and ingested"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(question: str = Form(...), session_id: str = Form("default")):
    memory = get_memory(session_id)
    result = graph.invoke({"input": question, "session_id": session_id})
    response_text = result["response"]

    # Store messages in Redis-backed memory
    memory["messages"].append({"role": "user", "text": question})
    memory["messages"].append({"role": "bot", "text": response_text})
    save_memory(session_id, memory)

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
    return {
        "session_id": session["session_id"],
        "created": session.get("created"),
        "messages": session.get("messages", [])
    }

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
