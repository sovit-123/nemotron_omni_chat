# Nemotron Omni Chat

A multimodal conversational AI system built with:
- **Backend**: Modal vLLM API running Nemotron model
- **Frontend**: Gradio UI for local chat interface
- **Inference**: GPU-accelerated with L40S

## Architecture

```
Gradio UI (local)
    ↓
Modal vLLM API
    ↓
Nemotron NVFP4
```

## Project Structure

```
nemotron-gradio-modal/
├── backend
│   ├── app.py
│   └── requirements.txt
├── frontend
│   ├── app.py
│   │   └── app.cpython-312.pyc
│   └── requirements.txt
├── README.md
└── requirements.txt
```

## Setup Instructions

### Backend Setup

1. Create virtual environment:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup Modal:
```bash
modal setup
```

4. Create Hugging Face token at [HF Tokens](https://huggingface.co/settings/tokens)

5. Save HF token in Modal:
```bash
modal secret create hf-secret \
HF_TOKEN=YOUR_HF_TOKEN
```

6. Deploy backend:
```bash
modal deploy app.py
```

7. Save the endpoint URL provided (format: `https://YOUR-NAME--nemotron-omni-serve.modal.run`)

8. Test backend:
```bash
curl https://YOUR-ENDPOINT/v1/models
```

### Frontend Setup

1. Create virtual environment:
```bash
cd frontend
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Update `.env` with your Modal endpoint:
```env
API_BASE_URL=https://YOUR-ENDPOINT/v1
```

4. Run frontend:
```bash
python app.py
```

5. Open browser to the provided URL (typically http://localhost:7860)

## Features

- **Multimodal Support**: Images, videos, audio, and text
- **Conversation History**: In-memory history per session
- **Local UI**: Gradio interface running locally
- **GPU Inference**: Cloud GPU acceleration via Modal
- **OpenAI-compatible API**: Uses OpenAI client library

## Supported File Types

- Images: `.png`, `.jpg`, `.jpeg`
- Videos: `.mp4`
- Audio: `.mp3`, `.wav`
- Text: Plain text input
