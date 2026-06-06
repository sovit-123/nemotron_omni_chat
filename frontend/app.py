import gradio as gr

from chat_service import chat
from theme import APP_CSS, theme


demo = gr.ChatInterface(
    fn=chat,
    multimodal=True,
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
        placeholder="Ask anything...",
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
