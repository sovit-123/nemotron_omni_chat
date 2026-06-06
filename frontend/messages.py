import ast
import base64
from pathlib import Path


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
