# app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from components.document_loader import load_document, chunk_text
from components.vector_store import save_faiss_index, embed_chunks, delete_faiss_index
from components.asr import transcribe_audio, stream_asr
from components.memory_store import get_memory, save_memory, list_sessions, delete_session
from agents.graph_builder import build_graph

import os
import shutil

UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

graph = build_graph()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def get_index():
    return FileResponse("frontend/index.html")

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...), session_id: str = Form("default")):
    try:
        if not file.filename.endswith((".pdf", ".docx")):
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        
        path = f"{UPLOAD_FOLDER}/{session_id}_{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())
        
        text = load_document(path)
        if len(text.strip()) < 50:
            os.remove(path)
            raise HTTPException(status_code=400, detail="Document is too short to process.")

        chunks = chunk_text(text)
        if not chunks:
            os.remove(path)
            raise HTTPException(status_code=400, detail="Failed to split document into chunks.")
        
        embeddings = embed_chunks(chunks)
        if embeddings.shape[0] == 0:
            os.remove(path)
            raise HTTPException(status_code=400, detail="Embedding failed.")

        save_faiss_index(embeddings, chunks, index_name=f"{session_id}.faiss")

        # Save document name in memory
        memory = get_memory(session_id)
        memory.setdefault("documents", [])
        memory["documents"].append(file.filename)
        save_memory(session_id, memory)

        return {"status": "uploaded", "filename": file.filename}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/delete_doc")
def delete_document(session_id: str = Form(...), filename: str = Form(...)):
    file_path = f"{UPLOAD_FOLDER}/{session_id}_{filename}"
    if os.path.exists(file_path):
        os.remove(file_path)

    # Delete FAISS index
    delete_faiss_index(f"{session_id}.faiss")

    # Remove from memory
    memory = get_memory(session_id)
    memory["documents"] = [doc for doc in memory.get("documents", []) if doc != filename]
    save_memory(session_id, memory)

    return {"status": "deleted"}

@app.post("/ask")
async def ask(question: str = Form(...), session_id: str = Form("default")):
    memory = get_memory(session_id)
    result = graph.invoke({"input": question, "session_id": session_id})
    response_text = result["response"]

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
        "messages": session.get("messages", []),
        "documents": session.get("documents", [])
    }

@app.post("/delete_session")
def delete_chat(session_id: str = Form(...)):
    delete_session(session_id)
    return {"status": "deleted"}

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
