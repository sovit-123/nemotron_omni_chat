import modal
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
APP_NAME = os.getenv("APP_NAME")

app = modal.App(APP_NAME)

VLLM_PORT = 8000

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "clang",
        "build-essential",
        "ffmpeg",
        "libsndfile1",
    )
    .run_commands(
        "pip install --upgrade pip wheel setuptools"
    )
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "vllm[audio]==0.20.0",
        "fastapi",
        "openai",
        "transformers",
        "huggingface_hub[hf_xet]",
        "hf_transfer",
    )
    .run_commands(
        "pip install "
        "git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 "
        "--no-build-isolation"
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

MODEL_DIR = "/models"

volume = modal.Volume.from_name(
    "nemotron-cache",
    create_if_missing=True,
)

GPU = "L40S"


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hf-secret")],
    volumes={
        MODEL_DIR: volume,
    },
)
@modal.concurrent(max_inputs=10)
@modal.web_server(
    port=VLLM_PORT,
    startup_timeout=60 * 45,
)
def serve():
    model_path = (
        f"{MODEL_DIR}/"
        "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
    )

    if not os.path.exists(model_path):

        subprocess.run(
            [
                "huggingface-cli",
                "download",
                MODEL_NAME,
                "--local-dir",
                model_path,
            ],
            check=True,
        )

        volume.commit()

    cmd = [
        "vllm",
        "serve",
        model_path,

        "--served-model-name",
        MODEL_NAME,

        "--host",
        "0.0.0.0",

        "--port",
        str(VLLM_PORT),

        "--trust-remote-code",

        "--tensor-parallel-size",
        "1",

        "--dtype",
        "auto",

        "--kv-cache-dtype",
        "fp8",

        "--gpu-memory-utilization",
        "0.80",

        "--max-model-len",
        "32768",

        "--max-num-seqs",
        "1",

        "--max-num-batched-tokens",
        "32768",

        "--limit-mm-per-prompt",
        '{"video":1,"image":1,"audio":1}',

        "--allowed-local-media-path",
        "/",

        "--enforce-eager",

        "--reasoning-parser",
        "nemotron_v3",

        "--enable-auto-tool-choice",

        "--tool-call-parser",
        "qwen3_coder",

        "--media-io-kwargs",
        '{"video":{"fps":1,"num_frames":512}}',

        "--video-pruning-rate",
        "0.25",

        "--uvicorn-log-level",
        "info",
    ]

    print("Launching vLLM...")
    print(" ".join(cmd))

    subprocess.Popen(
        cmd,
    )
