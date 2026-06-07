import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from messages import get_file_path


class SessionRAG:
    def __init__(self):
        self.chroma_client = chromadb.Client(
            Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="session_documents",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_model = None
        self.uploaded_files = set()

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        return self._embedding_model

    def apply_context(self, current_message, text, rag_files):
        if not rag_files or not text.strip():
            return current_message

        self.index_files_if_changed(rag_files)

        relevant_chunks = self.search_documents(text, top_k=3)
        print(f"Relevant chunks: {relevant_chunks}")

        if not relevant_chunks:
            return current_message

        context = "\n\n".join(relevant_chunks)
        context_text = (
            "Context from documents:\n"
            f"{context}\n\n"
            f"User question: {text}"
        )

        if len(current_message["content"]) > 1:
            current_message["content"] = [
                {"type": "text", "text": context_text}
            ] + current_message["content"][1:]
        else:
            current_message["content"] = [
                {"type": "text", "text": context_text}
            ]

        return current_message

    def index_files_if_changed(self, rag_files):
        current_file_paths = {
            get_file_path(file)
            for file in rag_files
        }

        if current_file_paths == self.uploaded_files:
            return

        self.clear_collection()
        self.uploaded_files.clear()

        for file in rag_files:
            file_path = get_file_path(file)
            ext = Path(file_path).suffix.lower()

            try:
                if ext == ".pdf":
                    doc_text = extract_text_from_pdf(file_path)
                elif ext == ".docx":
                    doc_text = extract_text_from_docx(file_path)
                elif ext == ".txt":
                    doc_text = extract_text_from_txt(file_path)
                else:
                    continue

                doc_id = os.path.basename(file_path)
                self.add_document(doc_id, doc_text)
                self.uploaded_files.add(file_path)
            except Exception as error:
                print(f"Error processing {file_path}: {error}")

    def clear_collection(self):
        all_docs = self.collection.get()

        if all_docs["ids"]:
            self.collection.delete(ids=all_docs["ids"])

    def add_document(self, doc_id, text):
        chunks = chunk_text(text)
        embeddings = self.embedding_model.encode(chunks).tolist()
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
        )

        return len(chunks)

    def search_documents(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        if results["documents"] and results["documents"][0]:
            return results["documents"][0]

        return []


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


def extract_text_from_file(file_path):
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)

    if ext == ".docx":
        return extract_text_from_docx(file_path)

    if ext == ".txt":
        return extract_text_from_txt(file_path)

    raise ValueError(f"Unsupported document type: {ext}")


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""

    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    return text


def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    words = text.split()

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks
