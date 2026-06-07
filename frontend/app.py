import gradio as gr

from chat_service import chat, summarize_document
from messages import get_file_path
from theme import APP_CSS, theme


def update_document_choices(files):
    choices = []

    for file in files or []:
        file_path = get_file_path(file)

        if file_path:
            choices.append(file_path.name if hasattr(file_path, "name") else file_path)

    choices = [
        str(choice).split("/")[-1]
        for choice in choices
    ]

    return gr.update(
        choices=choices,
        value=choices[0] if choices else None,
    )


with gr.Blocks(
    title="Nemotron Omni",
) as demo:
    gr.Markdown("# Nemotron Omni")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            max_tokens = gr.Slider(
                minimum=2048,
                maximum=16000,
                value=2048,
                step=256,
                label="Max tokens",
            )
            show_reasoning = gr.Checkbox(
                value=False,
                label="Show reasoning",
            )

        with gr.Column(scale=4):
            with gr.Accordion("Document Workspace", open=True):
                use_documents = gr.Checkbox(
                    value=False,
                    label="Use uploaded documents in chat answers",
                )
                rag_files = gr.File(
                    file_types=[".pdf", ".txt", ".docx"],
                    file_count="multiple",
                    label="Documents",
                )
                selected_document = gr.Dropdown(
                    choices=[],
                    label="Document action target",
                    interactive=True,
                )

                with gr.Row():
                    summary_style = gr.Dropdown(
                        choices=[
                            "Executive summary",
                            "Detailed summary",
                            "Study notes",
                            "Action items",
                        ],
                        value="Detailed summary",
                        label="Summary type",
                    )
                    summarize_button = gr.Button(
                        "Generate Summary",
                        variant="primary",
                    )

                summary_output = gr.Markdown(height=300, label="Summary output")

    rag_files.change(
        fn=update_document_choices,
        inputs=rag_files,
        outputs=selected_document,
    )

    summarize_button.click(
        fn=summarize_document,
        inputs=[
            rag_files,
            selected_document,
            summary_style,
            max_tokens,
        ],
        outputs=summary_output,
    )

    gr.ChatInterface(
        fn=chat,
        multimodal=True,
        title=None,
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
            max_tokens,
            show_reasoning,
            use_documents,
            rag_files,
        ],
        additional_inputs_accordion=None,
    )


if __name__ == "__main__":
    demo.launch(
        theme=theme,
        css=APP_CSS,
    )
