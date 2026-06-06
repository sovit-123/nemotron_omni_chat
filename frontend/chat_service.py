from openai import APIError, OpenAI

from config import API_BASE_URL, MODEL
from messages import build_message, extract_text, normalize_history
from rag import SessionRAG
from responses import extract_reasoning, render_response, split_thinking_tags

client = OpenAI(
    base_url=API_BASE_URL,
    api_key="dummy",
)

rag_engine = SessionRAG()


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
