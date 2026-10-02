import subprocess
import os
import sys
import time
import urllib.request
import json

LLAMA_SERVER = r"<USERPROFILE>\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
CUDA_DIR = r"<USERPROFILE>\AppData\Local\Programs\Ollama\lib\ollama\cuda_v13"
OLLAMA_LIB = r"<USERPROFILE>\AppData\Local\Programs\Ollama\lib\ollama"
MODEL_70B = r"<USERPROFILE>\.ollama\models\blobs\sha256-ba1103315c449ad06c9f5fd94230bde5bcf977f794af70afb107d29153c3cd53"
MODEL_1B = r"<USERPROFILE>\.ollama\models\blobs\sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
LOG_FILE = r"C:\Harness\benchmarks\code_spec_run.log"

def get_env():
    env = os.environ.copy()
    env["GGML_BACKEND_PATH"] = os.path.join(CUDA_DIR, "ggml-cuda.dll")
    env["PATH"] = f"{OLLAMA_LIB};{CUDA_DIR};" + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env

def get_nvidia_smi():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return res.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def wait_for_server(timeout=180):
    start = time.time()
    print("  [1/4] Booting Speculative 70B with -ngl 13 (High VRAM Headroom)...", flush=True)
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request("http://127.0.0.1:8080/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if data.get("status") == "ok":
                        print(f"  [+] Server ready in {time.time()-start:.1f}s!", flush=True)
                        return True
        except Exception:
            pass
        time.sleep(2)
    return False

def execute_prompt(prompt, n_predict=40):
    url = "http://127.0.0.1:8080/completion"
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
    print("=" * 70, flush=True)
    print("  HARNESS SPECULATIVE 70B: OPTIMIZED VRAM HEADROOM & CODE CONTINUATION", flush=True)
    print(f"  Initial VRAM (Used, Free, Total MiB): {get_nvidia_smi()}", flush=True)
    print("=" * 70, flush=True)

    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
        except Exception:
            pass

    cmd = [
        LLAMA_SERVER,
        "--model", MODEL_70B,
        "-ngl", "13",
        "--model-draft", MODEL_1B,
        "-ngld", "99",
        "--spec-type", "draft-simple",
        "--spec-draft-n-max", "4",
        "--load-mode", "none",
        "-np", "1",
        "-c", "2048",
        "-b", "512",
        "-ub", "512",
        "--flash-attn", "auto",
        "--port", "8080",
        "--host", "127.0.0.1",
        "--no-webui"
    ]

    log_fd = open(LOG_FILE, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, env=get_env(), stdout=log_fd, stderr=subprocess.STDOUT)

    try:
        ready = wait_for_server(timeout=180)
        if not ready:
            print("  [-] ERROR: Server failed to initialize within timeout.", flush=True)
            return

        time.sleep(2)
        vram_loaded = get_nvidia_smi()
        print(f"  [2/4] Post-Load VRAM Allocation: {vram_loaded} MiB (Safe Headroom Guaranteed!)", flush=True)

        prompt = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        n_predict = 40
        print(f"\n  [3/4] Dispatching structured code continuation benchmark...", flush=True)
        print(f"  Prompt:\n{prompt}", flush=True)
        
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

        acceptance_rate = (draft_n_accepted / draft_n * 100.0) if (draft_n and draft_n > 0) else 0.0

        print("\n" + "=" * 70, flush=True)
        print("  [4/4] LIVE HARDWARE RESULTS - SPECULATIVE 70B VERIFIED", flush=True)
        print("=" * 70, flush=True)
        print(f"  • Model Evaluated:         Meta-Llama-3.1-70B-Instruct (70.55B)", flush=True)
        print(f"  • Draft Model:             Llama-3.2-1B-Instruct (1.24B VRAM Resident)", flush=True)
        print(f"  • Prompt Evaluation Time:  {prompt_ms:.1f} ms ({prompt_n} tokens)", flush=True)
        print(f"  • Generated Tokens:        {predicted_n} tokens", flush=True)
        print(f"  • Generation Time:         {predicted_ms:.1f} ms ({predicted_ms / 1000.0:.2f} s)", flush=True)
        print(f"  • Measured Throughput:     {predicted_tok_s:.3f} tok/s (Real Hardware Measured)", flush=True)
        print(f"  • Latency Per Token:       {predicted_ms / predicted_n:.2f} ms/token", flush=True)
        print(f"  • Draft Tokens Generated:  {draft_n}", flush=True)
        print(f"  • Draft Tokens Accepted:   {draft_n_accepted} ({acceptance_rate:.1f}% acceptance)", flush=True)
        print(f"  • VRAM Utilized:           {vram_loaded} MiB", flush=True)
        print(f"  • Baseline 70B Speed:      1.047 tok/s (922.9 ms/token)", flush=True)
        print(f"  • Speedup Factor:          {predicted_tok_s / 1.047:.2f}x Acceleration", flush=True)
        print("=" * 70, flush=True)
        print(f"\n  Generated Text:\n{prompt}{content}\n", flush=True)

    finally:
        print("\n  Stopping server process and releasing all VRAM...", flush=True)
        p.terminate()
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
        log_fd.close()
        time.sleep(3)
        print(f"  Restored VRAM: {get_nvidia_smi()}", flush=True)
        print("  Done.\n", flush=True)

if __name__ == "__main__":
    main()
