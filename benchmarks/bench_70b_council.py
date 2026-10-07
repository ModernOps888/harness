import subprocess
import os
import sys
import time
import urllib.request
import json
import shutil

USER_HOME = os.path.expanduser("~")
LOCAL_APP_DATA = os.environ.get("LOCALAPPDATA", os.path.join(USER_HOME, "AppData", "Local"))
OLLAMA_LIB = os.environ.get("OLLAMA_LIB", os.path.join(LOCAL_APP_DATA, "Programs", "Ollama", "lib", "ollama"))
CUDA_DIR = os.path.join(OLLAMA_LIB, "cuda_v13")
LLAMA_SERVER = os.path.join(OLLAMA_LIB, "llama-server.exe")
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", os.path.join(USER_HOME, ".ollama", "models"))
BLOBS = os.path.join(OLLAMA_MODELS, "blobs")
MODEL_70B = os.path.join(BLOBS, "sha256-ba1103315c449ad06c9f5fd94230bde5bcf977f794af70afb107d29153c3cd53")
MODEL_1B = os.path.join(BLOBS, "sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_run.log")

def get_env():
    env = os.environ.copy()
    env["GGML_BACKEND_PATH"] = os.path.join(CUDA_DIR, "ggml-cuda.dll")
    env["PATH"] = f"{OLLAMA_LIB};{CUDA_DIR};" + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env

def wait_for_server(timeout=180):
    start = time.time()
    print("Waiting for server to initialize 70B weights...", flush=True)
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request("http://127.0.0.1:8080/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if data.get("status") == "ok":
                        print(f"Server is ready! (took {time.time()-start:.1f}s)", flush=True)
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

def execute_prompt(prompt, n_predict=30):
    url = "http://127.0.0.1:8080/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0, # Greedy for deterministic scientific comparison
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

def run_experiment(name, extra_args, prompt, n_predict=30):
    print(f"\n=======================================================", flush=True)
    print(f"STARTING EXPERIMENT: {name}", flush=True)
    print(f"Extra args: {' '.join(extra_args)}", flush=True)
    print(f"Initial VRAM: {get_nvidia_smi()}", flush=True)
    print(f"=======================================================", flush=True)

    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
        except Exception:
            pass

    cmd = [
        LLAMA_SERVER,
        "--model", MODEL_70B,
        "--load-mode", "none",
        "-np", "1",
        "-c", "2048",
        "-b", "512",
        "-ub", "512",
        "--flash-attn", "auto",
        "--no-warmup",
        "--port", "8080",
        "--host", "127.0.0.1",
        "--no-webui"
    ] + extra_args

    log_fd = open(LOG_FILE, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, env=get_env(), stdout=log_fd, stderr=subprocess.STDOUT)

    try:
        ready = wait_for_server(timeout=180)
        if not ready:
            print("FAILED: Server failed to start within timeout.", flush=True)
            return None

        # Give 2 seconds for telemetry to settle
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

        print(f"\n--- RESULTS FOR {name} ---", flush=True)
        print(f"Generated text preview: {repr(content[:120])}...", flush=True)
        print(f"Prompt Tokens: {prompt_n} ({prompt_ms:.1f} ms)", flush=True)
        print(f"Predicted Tokens: {predicted_n} tokens", flush=True)
        print(f"Predicted Duration (Engine): {predicted_ms:.1f} ms", flush=True)
        print(f"Measured Engine Throughput: {predicted_tok_s:.3f} tok/s", flush=True)
        print(f"Wall-Clock Total Duration: {wall_duration:.2f} s ({predicted_n / wall_duration:.3f} tok/s)", flush=True)
        if draft_n is not None:
            acc_rate = (draft_n_accepted / draft_n * 100.0) if draft_n > 0 else 0.0
            print(f"Draft Tokens Generated: {draft_n}", flush=True)
            print(f"Draft Tokens Accepted: {draft_n_accepted} ({acc_rate:.1f}% acceptance)", flush=True)

        return {
            "name": name,
            "predicted_n": predicted_n,
            "predicted_ms": predicted_ms,
            "tok_s": predicted_tok_s,
            "wall_duration": wall_duration,
            "wall_tok_s": predicted_n / wall_duration,
            "draft_n": draft_n,
            "draft_n_accepted": draft_n_accepted,
            "vram": vram_loaded,
            "content": content
        }

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
    test_mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    prompt = "Explain in detail the mathematical foundation of gradient descent optimization in deep learning, including learning rate convergence conditions."
    
    if test_mode == "baseline":
        res = run_experiment("Baseline Full 70B (Autoregressive, -ngl 22)", ["-ngl", "22"], prompt, n_predict=30)
    elif test_mode == "ngram":
        res = run_experiment("Speculative Full 70B (N-gram Lookahead, -ngl 22)", ["-ngl", "22", "--spec-type", "ngram-simple"], prompt, n_predict=30)
    elif test_mode == "neural":
        # 1B draft model fully in VRAM, 70B with 15 layers in VRAM
        res = run_experiment("Speculative Full 70B (Neural 1B Draft + 70B Verifier)", [
            "-ngl", "15",
            "--model-draft", MODEL_1B,
            "-ngld", "99",
            "--spec-type", "draft-simple",
            "--spec-draft-n-max", "4"
        ], prompt, n_predict=30)
    else:
        print(f"Unknown test mode: {test_mode}")
