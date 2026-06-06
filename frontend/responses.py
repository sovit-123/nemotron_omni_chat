import html

from messages import extract_text


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
