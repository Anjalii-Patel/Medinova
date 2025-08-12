# 🩺 Medinova – AI-Powered Medical Assistant

Medinova is an AI-powered healthcare assistant that combines **speech recognition**, **document understanding**, and **conversational AI** to deliver context-aware, retrieval-augmented responses.  
It supports real-time voice input, PDF/DOCX knowledge ingestion, and persistent memory for long-term patient interaction.

---

## 🚀 Features

- **Speech-to-Text (ASR)** – Real-time and file-based transcription using `faster-whisper`.
- **Document Understanding** – Upload PDF/DOCX files, automatically extract and process content.
- **RAG (Retrieval-Augmented Generation)** – FAISS-powered vector search to retrieve relevant context for responses.
- **Conversational Memory** – Persistent memory across sessions using Redis.
- **LLM Integration** – Powered by locally hosted models via Ollama (`medllama2` by default).
- **Graph-Based Orchestration** – Modular chatbot flow using LangGraph.
- **Frontend Interface** – Simple HTML/JS frontend for interaction.

---

## 🛠 Tech Stack

**Backend**  
- FastAPI – REST & WebSocket API
- LangGraph + LangChain Core – Agent orchestration
- Ollama – Local LLM API
- FAISS – Vector database
- Redis – Conversation memory
- faster-whisper – Speech recognition
- SentenceTransformers – Text embeddings

**Frontend**  
- HTML, CSS, JavaScript

**Utilities**  
- PyMuPDF (fitz) & python-docx – Document parsing
- NumPy, SoundFile – Audio processing
- Requests – API calls

---

## 📂 Project Structure

```
Medinova/
│
├── app.py                     # FastAPI backend entry point
├── requirements.txt           # Dependencies
├── agents/
│   └── graph_builder.py       # LangGraph-based chatbot logic
├── components/
│   ├── asr.py                 # Speech-to-text processing
│   ├── document_loader.py     # PDF/DOCX loading & parsing
│   ├── llm_ollama.py          # Ollama API integration
│   ├── memory_store.py        # Redis-based conversation memory
│   └── vector_store.py        # FAISS vector storage
├── frontend/
│   ├── app.js                 # Frontend JavaScript
│   ├── index.html             # UI
│   └── styles.css             # Styling
└── .gitignore
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Anjalii-Patel/Medinova.git
cd Medinova
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Start Redis
Make sure Redis is running locally:
```bash
redis-server
```

### 4️⃣ Start Ollama
Install [Ollama](https://ollama.ai) and pull the model:
```bash
ollama pull medllama2
ollama run medllama2
```

### 5️⃣ Run the backend
```bash
uvicorn app:app --reload
```

### 6️⃣ Open the frontend
Open `frontend/index.html` in your browser.

---

## 📡 API Endpoints

### **Upload Document**
```http
POST /upload
```
Uploads and processes PDF/DOCX for RAG.

### **Transcribe Audio**
```http
POST /transcribe
```
Converts audio files to text.

### **WebSocket Chat**
```
/ws/{session_id}
```
Handles real-time chat with memory.

---

## 📐 Architecture

```plaintext
Frontend (HTML/JS)  →  FastAPI Backend  →  LangGraph Orchestrator
      ↑                       ↓
   WebSocket             Ollama LLM
                          ↑     ↓
                    FAISS Vector Store ← Embeddings ← Documents / Audio
                          ↑
                       Redis Memory
```

---

## 🧪 Example Use Case

1. Upload a **patient report PDF**.
2. Speak to the assistant via **microphone**.
3. Medinova retrieves relevant medical history from FAISS.
4. Generates an **AI-powered response** using local LLM.

---

## 🤝 Contributing

1. Fork this repo.
2. Create a new branch:
```bash
git checkout -b feature-name
```
3. Commit changes and push.
4. Open a Pull Request.

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---
