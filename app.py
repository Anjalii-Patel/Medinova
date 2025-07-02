from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from components.document_loader import load_document, chunk_text
from components.vector_store import query_faiss, embed_chunks, save_faiss_index
from agents.graph_builder import build_graph
from components.asr_stt import transcribe_audio

app = FastAPI()
graph = build_graph()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def index():
    with open("frontend/index.html", "r") as f:
        return f.read()


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save file to disk
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text
    text = load_document(file_path)
    
    # Chunk & embed
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    # Save to FAISS
    save_faiss_index(embeddings, chunks)

    return {"status": "Ingested successfully", "filename": file.filename}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    state = {"input": question}
    result = graph.invoke(state)
    return {
        "response": result.get("response", ""),
        "followup": result.get("followup_required", False)
    }

@app.post("/transcribe")
async def transcribe_audio_file(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    transcription = transcribe_audio(file_path)
    return {"transcription": transcription}