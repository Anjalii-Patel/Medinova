# components/vector_store.py
import faiss
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FOLDER = "vector_store"
os.makedirs(INDEX_FOLDER, exist_ok=True)

def save_faiss_index(new_embeddings, new_chunks, index_name="default.faiss"):
    index_path = os.path.join(INDEX_FOLDER, index_name)
    chunks_path = index_path + ".pkl"

    dim = len(new_embeddings[0])
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        # Load existing
        existing_index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            existing_chunks = pickle.load(f)
        
        # Merge
        existing_index.add(new_embeddings)
        all_chunks = existing_chunks + new_chunks
    else:
        # Create new
        existing_index = faiss.IndexFlatL2(dim)
        existing_index.add(new_embeddings)
        all_chunks = new_chunks

    # Save updated
    faiss.write_index(existing_index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)

def load_faiss_index(index_name="default.faiss"):
    index_path = os.path.join(INDEX_FOLDER, index_name)
    if not os.path.exists(index_path):
        return None, []

    faiss_index = faiss.read_index(index_path)
    with open(index_path + ".pkl", "rb") as f:
        chunks = pickle.load(f)
    return faiss_index, chunks

def query_faiss(query, index_name="default.faiss", top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query])
    index, chunks = load_faiss_index(index_name)
    if not index:
        return []

    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return np.array(embeddings).astype("float32")
