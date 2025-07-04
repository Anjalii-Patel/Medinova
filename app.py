# === app.py ===
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from components.document_loader import load_document, chunk_text
from components.vector_store import embed_chunks, save_faiss_index
from components.asr import transcribe_audio
from components.asr import stream_asr
from agents.graph_builder import build_graph
import os

UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

graph = build_graph()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", response_class=HTMLResponse)
def index():
    with open("frontend/index.html") as f:
        return f.read()

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

    result = graph.invoke({"input": question, "session_id": session_id})
    return {
        "response": result["response"],
        "followup": result.get("followup_required", False)
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
