# components/document_loader.py
import os
import fitz
import docx
from typing import List
from components.llm_ollama import query_ollama  # or your LLM function

def load_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def load_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def load_document(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return load_pdf(file_path)
    elif file_path.endswith(".docx"):
        return load_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def recursive_summarize(chunks, summarize_func, max_chunks_per_pass=8):
    """
    Recursively summarize a list of text chunks using the provided summarize_func.
    summarize_func: function that takes a list of strings and returns a summary string.
    max_chunks_per_pass: how many chunks to summarize at once (fits LLM context window).
    """
    if len(chunks) <= max_chunks_per_pass:
        return summarize_func(chunks)
    # Summarize in batches
    summaries = []
    for i in range(0, len(chunks), max_chunks_per_pass):
        batch = chunks[i:i+max_chunks_per_pass]
        summary = summarize_func(batch)
        summaries.append(summary)
    # Recursively summarize the summaries
    return recursive_summarize(summaries, summarize_func, max_chunks_per_pass)

def summarize_chunks_with_llm(chunks):
    """
    Summarize a list of text chunks using your LLM.
    """
    prompt = "\n\n".join(chunks) + "\n\nSummarize the above medical content in clear, concise bullet points."
    return query_ollama(prompt)
