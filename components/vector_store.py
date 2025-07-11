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

def delete_faiss_index(index_name):
    index_path = os.path.join(INDEX_FOLDER, index_name)
    chunks_path = index_path + ".pkl"
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(chunks_path):
        os.remove(chunks_path)

def get_all_chunks(index_name="default.faiss"):
    _, chunks = load_faiss_index(index_name)
    return chunks

def save_chat_message_embedding(session_id, message_text):
    """
    Embed a chat message and add it to the session's chat FAISS index.
    """
    index_name = f"{session_id}_chat.faiss"
    index_path = os.path.join(INDEX_FOLDER, index_name)
    chunks_path = index_path + ".pkl"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode([message_text], convert_to_numpy=True, show_progress_bar=False)
    embedding = np.array(embedding).astype("float32")
    # Save or append
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            messages = pickle.load(f)
        index.add(embedding)
        messages.append(message_text)
        print(f"[FAISS] Appended message to index: {message_text}")
    else:
        index = faiss.IndexFlatL2(embedding.shape[1])
        index.add(embedding)
        messages = [message_text]
        print(f"[FAISS] Created new chat index with message: {message_text}")
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(messages, f)
    print(f"[FAISS] Saved chat index and messages for session: {session_id}")

def query_chat_faiss(session_id, query, top_k=5):
    """
    Query the session's chat FAISS index for relevant previous messages.
    """
    index_name = f"{session_id}_chat.faiss"
    index_path = os.path.join(INDEX_FOLDER, index_name)
    chunks_path = index_path + ".pkl"
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return []
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        messages = pickle.load(f)
    D, I = index.search(query_vec, top_k)
    print("[FAISS] Query results:", D, I)
    return [messages[i] for i in I[0] if i < len(messages)]

def delete_chat_faiss_index(session_id):
    index_name = f"{session_id}_chat.faiss"
    index_path = os.path.join(INDEX_FOLDER, index_name)
    chunks_path = index_path + ".pkl"
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(chunks_path):
        os.remove(chunks_path)