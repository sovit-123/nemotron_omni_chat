import gradio as gr

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

theme = gr.themes.Soft(
    primary_hue=ORANGE,
    secondary_hue=ORANGE,
)
