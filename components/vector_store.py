# components/vector_store.py
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

def save_faiss_index(new_embeddings, new_chunks):
    dim = len(new_embeddings[0])

    # Load existing index/chunks if available
    if os.path.exists(VECTOR_STORE_PATH):
        index = faiss.read_index(VECTOR_STORE_PATH)
        with open(METADATA_PATH, "rb") as f:
            old_chunks = pickle.load(f)
        all_chunks = old_chunks + new_chunks
    else:
        index = faiss.IndexFlatL2(dim)
        all_chunks = new_chunks

    index.add(new_embeddings)

    # Save updated index and metadata
    faiss.write_index(index, VECTOR_STORE_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("FAISS index updated with new document.")

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
