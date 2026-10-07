import subprocess
import os
import sys
import time
import urllib.request
import json

USER_HOME = os.path.expanduser("~")
LOCAL_APP_DATA = os.environ.get("LOCALAPPDATA", os.path.join(USER_HOME, "AppData", "Local"))
OLLAMA_LIB = os.environ.get("OLLAMA_LIB", os.path.join(LOCAL_APP_DATA, "Programs", "Ollama", "lib", "ollama"))
CUDA_DIR = os.path.join(OLLAMA_LIB, "cuda_v13")
LLAMA_SERVER = os.path.join(OLLAMA_LIB, "llama-server.exe")
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", os.path.join(USER_HOME, ".ollama", "models"))
BLOBS = os.path.join(OLLAMA_MODELS, "blobs")
MODEL_70B = os.path.join(BLOBS, "sha256-ba1103315c449ad06c9f5fd94230bde5bcf977f794af70afb107d29153c3cd53")
MODEL_1B = os.path.join(BLOBS, "sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_run.log")
PORT = int(os.environ.get("HARNESS_PORT", "8089"))

def get_env():
    env = os.environ.copy()
    env["GGML_BACKEND_PATH"] = os.path.join(CUDA_DIR, "ggml-cuda.dll")
    env["PATH"] = f"{OLLAMA_LIB};{CUDA_DIR};" + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env

def wait_for_server(timeout=180):
    start = time.time()
    print("Waiting for dual-engine 70B+1B initialization...", flush=True)
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if data.get("status") == "ok":
                        print(f"Dual-engine server is ready! (took {time.time()-start:.1f}s)", flush=True)
                        return True
        except Exception:
            pass
        time.sleep(2)
    return False

def get_nvidia_smi():
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def execute_prompt(prompt, n_predict=50):
    url = f"http://127.0.0.1:{PORT}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
    wall_duration = time.time() - t0
    return res_data, wall_duration

def main():
    prompt = "Write an optimized Rust function to compute Fibonacci numbers using dynamic programming, with docstrings and type annotations."
    n_predict = 50
    
    print("=======================================================", flush=True)
    print("STARTING ADVANCED HYBRID SPECULATIVE 70B EXPERIMENT", flush=True)
    print(f"Initial VRAM: {get_nvidia_smi()}", flush=True)
    print("=======================================================", flush=True)

    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
        except Exception:
            pass

    cmd = [
        LLAMA_SERVER,
        "--model", MODEL_70B,
        "-ngl", "16",
        "--model-draft", MODEL_1B,
        "-ngld", "99",
        "--spec-type", "draft-simple,ngram-simple",
        "--spec-draft-n-max", "6",
        "--load-mode", "none",
        "-np", "1",
        "-c", "2048",
        "-b", "512",
        "-ub", "512",
        "--flash-attn", "auto",
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "--no-webui"
    ]

    log_fd = open(LOG_FILE, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, env=get_env(), stdout=log_fd, stderr=subprocess.STDOUT)

    try:
        ready = wait_for_server(timeout=180)
        if not ready:
            print("FAILED: Server failed to start within timeout.", flush=True)
            return

        time.sleep(2)
        vram_loaded = get_nvidia_smi()
        print(f"Post-load VRAM (Used, Free, Total MiB): {vram_loaded}", flush=True)

        print(f"\nDispatching benchmark prompt (n_predict={n_predict})...", flush=True)
        res, wall_duration = execute_prompt(prompt, n_predict=n_predict)

        timings = res.get("timings", {})
        predicted_n = timings.get("predicted_n", 0)
        predicted_ms = timings.get("predicted_ms", 0.0)
        predicted_tok_s = timings.get("predicted_per_second", 0.0)
        prompt_n = timings.get("prompt_n", 0)
        prompt_ms = timings.get("prompt_ms", 0.0)
        draft_n = timings.get("draft_n", None)
        draft_n_accepted = timings.get("draft_n_accepted", None)
        content = res.get("content", "").strip()

        print(f"\n--- ADVANCED HYBRID SPECULATIVE RESULTS ---", flush=True)
        print(f"Prompt Tokens: {prompt_n} ({prompt_ms:.1f} ms)", flush=True)
        print(f"Predicted Tokens: {predicted_n} tokens", flush=True)
        print(f"Predicted Duration (Engine): {predicted_ms:.1f} ms", flush=True)
        print(f"Measured Engine Throughput: {predicted_tok_s:.3f} tok/s", flush=True)
        print(f"Wall-Clock Total Duration: {wall_duration:.2f} s ({predicted_n / wall_duration:.3f} tok/s)", flush=True)
        if draft_n is not None:
            acc_rate = (draft_n_accepted / draft_n * 100.0) if draft_n > 0 else 0.0
            print(f"Draft Tokens Generated: {draft_n}", flush=True)
            print(f"Draft Tokens Accepted: {draft_n_accepted} ({acc_rate:.1f}% acceptance)", flush=True)
        print(f"\nFull Generated Output:\n{content}\n", flush=True)

    finally:
        print("\nStopping server process...", flush=True)
        p.terminate()
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
        log_fd.close()
        time.sleep(3)
        print("Server stopped cleanly. VRAM restored.", flush=True)

if __name__ == "__main__":
    main()
