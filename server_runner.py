import os
import sys
import subprocess
import time
import json
import urllib.request
from pathlib import Path

# Dynamic environment resolution (zero hardcoded user paths)
USER_HOME = Path.home()
LOCAL_APP_DATA = Path(os.environ.get("LOCALAPPDATA", USER_HOME / "AppData" / "Local"))

OLLAMA_LIB = Path(os.environ.get("OLLAMA_LIB", LOCAL_APP_DATA / "Programs" / "Ollama" / "lib" / "ollama"))
CUDA_DIR = Path(os.environ.get("CUDA_DIR", OLLAMA_LIB / "cuda_v13"))
LLAMA_SERVER = OLLAMA_LIB / "llama-server.exe"

OLLAMA_MODELS = Path(os.environ.get("OLLAMA_MODELS", USER_HOME / ".ollama" / "models"))
BLOBS = OLLAMA_MODELS / "blobs"

MODEL_32B = BLOBS / "sha256-ac3d1ba8aa77755dab3806d9024e9c385ea0d5b412d6bdf9157f8a4a7e9fc0d9"
MODEL_DRAFT_1_5B = BLOBS / "sha256-29d8c98fa6b098e200069bfb88b9508dc3e85586d20cba59f8dda9a808165104"

PORT = 8089
BASE_DIR = Path(__file__).resolve().parent

def server_env():
    env = os.environ.copy()
    env["GGML_BACKEND_PATH"] = str(CUDA_DIR / "ggml-cuda.dll")
    env["PATH"] = f"{OLLAMA_LIB};{CUDA_DIR};" + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env

def is_running():
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False

def start():
    if is_running():
        print(f"[HARNESS] Server is already running and healthy on http://127.0.0.1:{PORT}")
        return

    cmd = [
        str(LLAMA_SERVER),
        "--model", str(MODEL_32B),
        "--load-mode", "none",
        "-np", "1",
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "--no-webui",
        "-c", "2048",
        "-t", "6",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-fa", "on",
        "-b", "256",
        "-ub", "128",
        "-ngl", "17",
        "--model-draft", str(MODEL_DRAFT_1_5B),
        "-ngld", "99",
        "--spec-draft-n-max", "4",
        "--spec-type", "draft-simple"
    ]

    log_path = BASE_DIR / "server.log"
    log_fd = open(log_path, "w", encoding="utf-8", errors="replace")
    
    # DETACHED_PROCESS = 0x00000008, CREATE_NEW_PROCESS_GROUP = 0x00000200
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(cmd, env=server_env(), stdout=log_fd, stderr=subprocess.STDOUT, creationflags=flags)
    
    print(f"[HARNESS] Launched Qwen2.5-Coder-32B + 1.5B Drafter server (PID {proc.pid})")
    print(f"[HARNESS] Logging to: {log_path}")
    print("[HARNESS] Waiting for server to initialize weights into VRAM/DDR4...")

    t0 = time.time()
    while time.time() - t0 < 120:
        if is_running():
            print(f"[HARNESS] Server is READY in {time.time()-t0:.1f}s at http://127.0.0.1:{PORT}")
            print(f"[HARNESS] OpenAI-Compatible API: http://127.0.0.1:{PORT}/v1/chat/completions")
            return
        time.sleep(2)
    print(f"[HARNESS] Server started, still loading weights in background. Check {log_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print("Running" if is_running() else "Stopped")
    else:
        start()
