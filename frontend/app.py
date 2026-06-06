import os
import base64
import ast
import html
from pathlib import Path

import gradio as gr
from openai import APIError, OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

# Initialize in-memory ChromaDB and embedding model
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(
    name="session_documents",
    metadata={"hnsw:space": "cosine"}
)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Track uploaded files to avoid re-processing
uploaded_files = set()

client = OpenAI(
    base_url=API_BASE_URL,
    api_key="dummy",
)

MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"

ORANGE = gr.themes.Color(
    c50="#fff3eb",
    c100="#ffe2ce",
    c200="#ffc49d",
    c300="#ffa66b",
    c400="#ff8e45",
    c500="#ff761d",
    c600="#df5d0d",
    c700="#b8470a",
    c800="#91380d",
    c900="#752f0f",
    c950="#3f1605",
    name="nemotron_orange",
)

APP_CSS = """
:root {
    --button-primary-background-fill: #ff761d;
    --button-primary-background-fill-hover: #df5d0d;
    --checkbox-label-background-fill-selected: #ff761d;
    --checkbox-label-border-color-selected: #ff761d;
    --slider-color: #ff761d;
}

.tab-nav button.selected,
.tab-nav button[aria-selected="true"] {
    background: #ff761d !important;
    color: #ffffff !important;
}

.label-wrap,
.block-label {
    background: #ff761d !important;
    color: #ffffff !important;
    border-color: #ff761d !important;
}

input[type="range"] {
    accent-color: #ff761d;
}

input[type="checkbox"] {
    accent-color: #ff761d;
}
"""


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def add_document_to_chroma(doc_id, text):
    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks).tolist()
    
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks
    )
    
    return len(chunks)


def search_documents(query, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    if results['documents'] and results['documents'][0]:
        return results['documents'][0]
    return []


def media_type_from_path_or_mime(path=None, mime=None):

    ext = Path(path).suffix.lower() if path else ""

    if ext in [".png", ".jpg", ".jpeg"] or mime in [
        "image/png",
        "image/jpeg",
    ]:
        return "image_url"

    if ext in [".mp4"] or mime == "video/mp4":
        return "video_url"

    if ext in [".mp3", ".wav"] or mime in [
        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
    ]:
        return "audio_url"

    return None


def file_to_data_uri(filepath):

    filepath = Path(filepath)

    ext = filepath.suffix.lower()

    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".mp4": "video/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
    }

    mime = mime_map.get(ext)

    if mime is None:
        raise ValueError(f"Unsupported file: {ext}")

    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    return f"data:{mime};base64,{encoded}"


def get_file_path(file):

    if isinstance(file, (str, Path)):
        return file

    if isinstance(file, dict):
        return (
            file.get("path")
            or file.get("name")
            or file.get("orig_name")
        )

    return getattr(file, "path", file)


def build_message(text, files):

    content = []

    text = text or ""
    files = files or []

    if text.strip():
        content.append({
            "type": "text",
            "text": text
        })

    for file in files:

        file_part = build_file_part(file)

        if file_part is not None:
            content.append(file_part)

    return {
        "role": "user",
        "content": content
    }


def build_file_part(file):

    url = None
    mime = None

    if isinstance(file, dict):
        url = file.get("url")
        mime = (
            file.get("mime_type")
            or file.get("mime")
            or file.get("type")
        )

    filepath = get_file_path(file)

    media_type = media_type_from_path_or_mime(filepath, mime)

    if url and str(url).startswith(("data:", "http://", "https://")):

        if media_type is None:
            return None

        return {
            "type": media_type,
            media_type: {
                "url": url
            }
        }

    if filepath is None:
        return None

    if media_type is None:
        raise ValueError(f"Unsupported file: {Path(filepath).suffix.lower()}")

    uri = file_to_data_uri(filepath)

    if media_type == "image_url":
        return {
            "type": "image_url",
            "image_url": {
                "url": uri
            }
        }

    if media_type == "video_url":
        return {
            "type": "video_url",
            "video_url": {
                "url": uri
            }
        }

    if media_type == "audio_url":
        return {
            "type": "audio_url",
            "audio_url": {
                "url": uri
            }
        }

    return None


def normalize_content_parts(parts):

    normalized = []

    for part in parts:

        if not isinstance(part, dict):
            normalized.append(part)
            continue

        part_type = part.get("type")

        if part_type in [
            "text",
            "image_url",
            "video_url",
            "audio_url",
        ]:
            normalized.append(part)
            continue

        if part_type == "file":
            file_part = build_file_part(
                part.get("file")
                or part.get("path")
                or part.get("url")
                or part
            )

            if file_part is not None:
                normalized.append(file_part)

            continue

        if get_file_path(part):
            file_part = build_file_part(part)

            if file_part is not None:
                normalized.append(file_part)

            continue

    return normalized


def normalize_user_content(content):

    if isinstance(content, list):
        return normalize_content_parts(content)

    if isinstance(content, dict):

        if "text" in content or "files" in content:
            return build_message(
                content.get("text", ""),
                content.get("files", []),
            )["content"]

        if get_file_path(content):
            return build_message("", [content])["content"]

        return build_message(
            content.get("text", ""),
            content.get("files", []),
        )["content"]

    return content


def extract_text(content):

    if content is None:
        return ""

    if isinstance(content, str):
        stripped_content = content.strip()

        if stripped_content.startswith("[") and stripped_content.endswith("]"):
            try:
                parsed_content = ast.literal_eval(stripped_content)
            except (SyntaxError, ValueError):
                parsed_content = None

            if isinstance(parsed_content, list):
                parsed_text = extract_text(parsed_content)

                if parsed_text:
                    return parsed_text

        if stripped_content.startswith("{") and stripped_content.endswith("}"):
            try:
                parsed_content = ast.literal_eval(stripped_content)
            except (SyntaxError, ValueError):
                parsed_content = None

            if isinstance(parsed_content, dict):
                parsed_text = extract_text(parsed_content)

                if parsed_text:
                    return parsed_text

        return content

    if isinstance(content, list):
        text_parts = []

        for part in content:

            if isinstance(part, str):
                text_parts.append(part)
                continue

            if not isinstance(part, dict):
                continue

            if part.get("type") == "text":
                text = part.get("text")

                if text:
                    text_parts.append(text)

        result = "".join(text_parts)
        return result if result else content

    if isinstance(content, dict):

        if content.get("type") == "text":
            return content.get("text", "")

        return content.get("text", "")

    return str(content)


def extract_reasoning(delta):

    for field_name in [
        "reasoning_content",
        "reasoning",
    ]:
        reasoning = getattr(delta, field_name, None)

        if reasoning:
            return extract_text(reasoning)

    model_extra = getattr(delta, "model_extra", None) or {}

    for field_name in [
        "reasoning_content",
        "reasoning",
    ]:
        reasoning = model_extra.get(field_name)

        if reasoning:
            return extract_text(reasoning)

    return ""


def split_thinking_tags(text):

    if not text:
        return "", ""

    start_tag = "<think>"
    end_tag = "</think>"
    start_index = text.find(start_tag)

    if start_index == -1:
        return "", text

    before = text[:start_index]
    remainder = text[start_index + len(start_tag):]
    end_index = remainder.find(end_tag)

    if end_index == -1:
        return remainder, before

    thinking = remainder[:end_index]
    after = remainder[end_index + len(end_tag):]

    return thinking, before + after


def render_response(thinking, answer):

    if not thinking:
        return answer

    escaped_thinking = html.escape(thinking).replace("\n", "<br>")

    thinking_box = (
        '<div style="background:#050505;color:#ffffff;'
        'border:1px solid #d1d5db;border-radius:4px;'
        'padding:10px 12px;margin-bottom:12px;">'
        '<div style="font-weight:700;margin-bottom:6px;">Thinking</div>'
        f'<div style="white-space:normal;">{escaped_thinking}</div>'
        '</div>'
    )

    return f"{thinking_box}\n\n{answer}"


def normalize_history(history):

    messages = []

    for item in history or []:

        if isinstance(item, dict):

            role = item.get("role")
            content = item.get("content")

            if role == "user":
                content = normalize_user_content(content)
            elif role == "assistant":
                content = extract_text(content)

            if role and content is not None:
                messages.append({
                    "role": role,
                    "content": content,
                })

            continue

        try:
            user_msg, assistant_msg = item
        except (TypeError, ValueError):
            continue

        if user_msg is not None:
            messages.append({
                "role": "user",
                "content": normalize_user_content(user_msg),
            })

        if assistant_msg is not None:
            messages.append({
                "role": "assistant",
                "content": extract_text(assistant_msg),
            })

    return messages


def chat(message, history, max_tokens, enable_thinking, enable_rag, rag_files):

    text = message.get("text", "")

    files = message.get("files", [])

    try:
        messages = normalize_history(history)

        current_message = build_message(text, files)
    except Exception as error:
        yield f"Could not prepare request: {error}"
        return

    if enable_rag and rag_files and text.strip():
        # Check if new files were uploaded
        current_file_paths = set()
        for file in rag_files:
            file_path = get_file_path(file)
            current_file_paths.add(file_path)
        
        # Only clear and re-add if new files were uploaded
        if current_file_paths != uploaded_files:
            all_docs = collection.get()
            if all_docs['ids']:
                collection.delete(ids=all_docs['ids'])
            
            uploaded_files.clear()
            
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
                    add_document_to_chroma(doc_id, doc_text)
                    uploaded_files.add(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        # Vector search using ChromaDB
        relevant_chunks = search_documents(text, top_k=3)
        print(f"Relevant chunks: {relevant_chunks}")
        if relevant_chunks:
            context = "\n\n".join(relevant_chunks)
            if len(current_message["content"]) > 1:
                current_message["content"] = [
                    {"type": "text", "text": f"Context from documents:\n{context}\n\nUser question: {text}"}
                ] + current_message["content"][1:]
            else:
                current_message["content"] = [
                    {"type": "text", "text": f"Context from documents:\n{context}\n\nUser question: {text}"}
                ]

    print(current_message)
    messages.append(current_message)

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,

            temperature=0.2,

            max_tokens=int(max_tokens),

            stream=True,

            extra_body={
                "top_k": 1,
                "chat_template_kwargs": {
                    "enable_thinking": bool(enable_thinking)
                },
                "mm_processor_kwargs": {
                    "use_audio_in_video": True
                },
            }
        )
    except APIError as error:
        yield f"Backend request failed: {error.message}"
        return
    except Exception as error:
        yield f"Request failed: {error}"
        return

    thinking = ""
    answer = ""

    try:
        for chunk in stream:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            reasoning_delta = extract_reasoning(delta)
            text_delta = getattr(delta, "content", None)
            text_delta = extract_text(text_delta)

            if reasoning_delta:
                thinking += reasoning_delta

            if text_delta:
                tagged_thinking, final_text = split_thinking_tags(text_delta)
                thinking += tagged_thinking
                answer += final_text

            if reasoning_delta or text_delta:
                yield render_response(thinking, answer)
    except APIError as error:
        yield f"Backend stream failed: {error.message}"
        return
    except Exception as error:
        yield f"Stream failed: {error}"
        return

    if not thinking and not answer:
        yield ""


theme = gr.themes.Soft(
    primary_hue=ORANGE,
    secondary_hue=ORANGE,
)

demo = gr.ChatInterface(
    fn=chat,

    multimodal=True,

    # theme=theme,

    title="Nemotron Omni",

    description=None,

    textbox=gr.MultimodalTextbox(
        file_types=[
            ".png",
            ".jpg",
            ".jpeg",
            ".mp4",
            ".mp3",
            ".wav",
        ],
        file_count="multiple",
        placeholder="Ask anything..."
    ),

    additional_inputs=[
        gr.Slider(
            minimum=2048,
            maximum=16000,
            value=2048,
            step=256,
            label="Max tokens",
        ),
        gr.Checkbox(
            value=False,
            label="Show reasoning",
        ),
        gr.Checkbox(
            value=False,
            label="Enable RAG",
        ),
        gr.File(
            file_types=[".pdf", ".txt", ".docx"],
            file_count="multiple",
            label="Upload documents for RAG (PDF, TXT, DOCX)",
        ),
    ],

    additional_inputs_accordion="Generation controls",
)

if __name__ == "__main__":
    demo.launch(
        theme=theme,
        css=APP_CSS,
    )
