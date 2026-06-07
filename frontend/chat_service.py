import os

from openai import APIError, OpenAI

from config import API_BASE_URL, MODEL
from messages import build_message, extract_text, get_file_path, normalize_history
from rag import SessionRAG, chunk_text, extract_text_from_file
from responses import extract_reasoning, render_response, split_thinking_tags

client = OpenAI(
    base_url=API_BASE_URL,
    api_key="dummy",
)

rag_engine = SessionRAG()


SUMMARY_STYLE_PROMPTS = {
    "Executive summary": (
        "Write a concise executive summary with the main purpose, key points, "
        "important findings, and decisions or implications."
    ),
    "Detailed summary": (
        "Write a detailed structured summary. Preserve important technical "
        "details, examples, constraints, and conclusions."
    ),
    "Study notes": (
        "Create study notes with headings, key concepts, definitions, and "
        "useful bullet points for review."
    ),
    "Action items": (
        "Extract concrete action items, owners if mentioned, dependencies, "
        "risks, and follow-up questions."
    ),
}


def chat(message, history, max_tokens, enable_thinking, enable_rag, rag_files):
    text = message.get("text", "")
    files = message.get("files", [])

    try:
        messages = normalize_history(history)
        current_message = build_message(text, files)
    except Exception as error:
        yield f"Could not prepare request: {error}"
        return

    if enable_rag:
        current_message = rag_engine.apply_context(
            current_message,
            text,
            rag_files,
        )

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
            },
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


def summarize_document(rag_files, selected_document, summary_style, max_tokens):
    if not rag_files:
        yield "Upload a PDF, TXT, or DOCX document first."
        return

    file_path = find_selected_file(rag_files, selected_document)

    if file_path is None:
        yield "Select a document to summarize."
        return

    try:
        document_text = extract_text_from_file(file_path)
    except Exception as error:
        yield f"Could not read document: {error}"
        return

    chunks = chunk_text(document_text, chunk_size=900, overlap=80)

    if not chunks:
        yield "I could not extract readable text from this document."
        return

    file_name = os.path.basename(file_path)
    style_prompt = SUMMARY_STYLE_PROMPTS.get(
        summary_style,
        SUMMARY_STYLE_PROMPTS["Detailed summary"],
    )
    partial_summaries = []

    yield f"Summarizing `{file_name}` in {len(chunks)} pass(es)..."

    for index, chunk in enumerate(chunks, start=1):
        prompt = (
            f"{style_prompt}\n\n"
            "Summarize this document section. Keep details that may be needed "
            "for the final document-level summary.\n\n"
            f"Document: {file_name}\n"
            f"Section {index} of {len(chunks)}:\n{chunk}"
        )

        try:
            section_summary = run_non_streaming_completion(
                prompt,
                max_tokens=min(int(max_tokens), 2048),
            )
        except Exception as error:
            yield f"Summary request failed on section {index}: {error}"
            return

        partial_summaries.append(section_summary)
        yield (
            f"Summarized {index}/{len(chunks)} section(s)...\n\n"
            f"Latest section summary:\n\n{section_summary}"
        )

    final_prompt = (
        f"{style_prompt}\n\n"
        "Create one final document-level summary from these section summaries. "
        "Remove repetition, keep the structure clear, and mention important "
        "limitations or missing information if the summaries show any.\n\n"
        f"Document: {file_name}\n\n"
        "Section summaries:\n\n"
        + "\n\n".join(
            f"Section {index}:\n{summary}"
            for index, summary in enumerate(partial_summaries, start=1)
        )
    )

    try:
        final_summary = run_non_streaming_completion(
            final_prompt,
            max_tokens=int(max_tokens),
        )
    except Exception as error:
        yield f"Final summary request failed: {error}"
        return

    yield f"## Summary: {file_name}\n\n{final_summary}"


def find_selected_file(files, selected_document):
    if not selected_document:
        return None

    for file in files or []:
        file_path = get_file_path(file)

        if os.path.basename(file_path) == selected_document:
            return file_path

    return None


def run_non_streaming_completion(prompt, max_tokens):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        extra_body={
            "top_k": 1,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    )

    return extract_text(response.choices[0].message.content)
