import os
import faiss
import pickle
from typing import List
from sentence_transformers import SentenceTransformer

VECTOR_STORE_PATH = "vector_store/faiss_index"
METADATA_PATH = "vector_store/metadata.pkl"

# Load or initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    return embedding_model.encode(chunks, show_progress_bar=True)

def save_faiss_index(embeddings, chunks):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, VECTOR_STORE_PATH)

    # Save metadata (chunks) for later use
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index and metadata saved.")

def load_faiss_index():
    if not os.path.exists(VECTOR_STORE_PATH):
        raise ValueError("Index not found!")
    
    index = faiss.read_index(VECTOR_STORE_PATH)
    with open(METADATA_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def query_faiss(query: str, k: int = 3):
    query_embedding = embedding_model.encode([query])
    index, chunks = load_faiss_index()
    scores, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]
