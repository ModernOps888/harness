"""
HARNESS raw-capture benchmark driver.

Principle: nothing in a result is typed by hand.
  * server stdout/stderr is redirected to a file UNMODIFIED
  * every /completion HTTP response is stored VERBATIM (raw bytes)
  * hardware snapshots (nvidia-smi, free RAM) are stored raw
  * summarize_runs.py computes every number from those files

Usage:
    python benchmarks/raw_capture.py run  <name> [--prompts all] -- <llama-server args...>
    python benchmarks/raw_capture.py matrix
"""
import ctypes
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

USER_HOME = os.path.expanduser("~")
LOCAL_APP_DATA = os.environ.get("LOCALAPPDATA", os.path.join(USER_HOME, "AppData", "Local"))
OLLAMA_LIB = os.environ.get("OLLAMA_LIB", os.path.join(LOCAL_APP_DATA, "Programs", "Ollama", "lib", "ollama"))
CUDA_DIR = os.path.join(OLLAMA_LIB, "cuda_v13")
LLAMA_SERVER = os.path.join(OLLAMA_LIB, "llama-server.exe")
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", os.path.join(USER_HOME, ".ollama", "models"))
BLOBS = os.path.join(OLLAMA_MODELS, "blobs")
# llama3.1:70b-instruct-q2_K  (26,375,113,056 bytes) and llama3.2:1b Q8_0 (1,321,082,688 bytes)
MODEL_70B = os.path.join(BLOBS, "sha256-ba1103315c449ad06c9f5fd94230bde5bcf977f794af70afb107d29153c3cd53")
MODEL_1B = os.path.join(BLOBS, "sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
PORT = 8089
N_PREDICT = 48

# NOTE: no explicit <|begin_of_text|>; llama-server prepends BOS itself (explicit one => double BOS warning).
TEMPLATE = (
    "<|start_header_id|>user<|end_header_id|>\n\n{p}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
PROMPTS = {
    "prose": "Explain in detail the mathematical foundation of gradient descent optimization in deep "
             "learning, including learning rate convergence conditions.",
    "code": "Write a Python function that merges two sorted lists into one sorted list, with a docstring "
            "and three assert-based tests.",
    "quote": "Here is a passage:\n\n'The memory wall is the growing gap between processor speed and memory "
             "bandwidth. In autoregressive decoding every generated token requires reading all model weights "
             "once, so throughput is bounded by memory bandwidth divided by model size.'\n\n"
             "Repeat the passage word for word, then add one sentence of commentary.",
}


class MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_ram_gb():
    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return round(m.ullAvailPhys / 2**30, 2), round(m.ullTotalPhys / 2**30, 2)


def smi():
    q = "memory.used,memory.free,memory.total,pcie.link.gen.current,pcie.link.width.current,clocks.current.memory"
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv"], capture_output=True, text=True)
    return {"time": datetime.now().isoformat(timespec="seconds"), "raw": r.stdout.strip()}


def server_env():
    env = os.environ.copy()
    env["GGML_BACKEND_PATH"] = os.path.join(CUDA_DIR, "ggml-cuda.dll")
    env["PATH"] = f"{OLLAMA_LIB};{CUDA_DIR};" + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def http_json(path, payload=None, timeout=600):
    url = f"http://127.0.0.1:{PORT}{path}"
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def wait_ready(proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, f"server exited early with code {proc.returncode}"
        try:
            body = http_json("/health", timeout=3)
            if json.loads(body).get("status") == "ok":
                return True, f"ready in {time.time() - t0:.1f}s"
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(2)
    return False, "timeout waiting for /health"


def kill_tree(proc):
    if proc.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def run_config(name, server_args, prompt_names=("prose", "code", "quote"), model=None, template=None):
    model = model or MODEL_70B
    template = template or TEMPLATE
    out = os.path.join(RUNS_DIR, name)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))

    cmd = [LLAMA_SERVER, "--model", model, "--load-mode", "none", "-np", "1",
           "--port", str(PORT), "--host", "127.0.0.1", "--no-webui"] + server_args
    meta = {
        "name": name,
        "cmd": cmd,
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                   cwd=HERE).stdout.strip(),
        "free_ram_gb_before_launch": free_ram_gb()[0],
        "total_ram_gb": free_ram_gb()[1],
        "n_predict": N_PREDICT,
        "smi_before_launch": smi(),
    }
    print(f"[{name}] launching: {' '.join(server_args)}", flush=True)
    log_fd = open(os.path.join(out, "server_stdout.txt"), "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, env=server_env(), stdout=log_fd, stderr=subprocess.STDOUT)
    try:
        ok, msg = wait_ready(proc)
        meta["ready"] = {"ok": ok, "msg": msg}
        print(f"[{name}] {msg}", flush=True)
        if not ok:
            return meta
        meta["smi_after_load"] = smi()
        meta["free_ram_gb_after_load"] = free_ram_gb()[0]

        # warm-up (page-in, graph build); stored but excluded from results
        warm = http_json("/completion", {"prompt": template.format(p="Say hello."), "n_predict": 8,
                                         "temperature": 0.0, "top_k": 1, "seed": 1, "cache_prompt": False})
        with open(os.path.join(out, "response_warmup.json"), "wb") as f:
            f.write(warm)

        for pn in prompt_names:
            payload = {"prompt": template.format(p=PROMPTS[pn]), "n_predict": N_PREDICT, "temperature": 0.0,
                       "top_k": 1, "seed": 1, "cache_prompt": False, "stream": False}
            t0 = time.time()
            raw = http_json("/completion", payload)
            wall = time.time() - t0
            with open(os.path.join(out, f"response_{pn}.json"), "wb") as f:
                f.write(raw)
            t = json.loads(raw).get("timings", {})
            print(f"[{name}] {pn}: wall={wall:.1f}s predicted_per_second={t.get('predicted_per_second')} "
                  f"draft={t.get('draft_n')}/{t.get('draft_n_accepted')}", flush=True)
        meta["smi_after_run"] = smi()
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
    finally:
        kill_tree(proc)
        log_fd.close()
        with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return meta


def offloaded_layers(name):
    """Parse the REAL server log for how many layers llama.cpp actually put on the GPU."""
    p = os.path.join(RUNS_DIR, name, "server_stdout.txt")
    if not os.path.exists(p):
        return None
    m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", open(p, encoding="utf-8", errors="replace").read())
    return (int(m.group(1)), int(m.group(2))) if m else None


COMMON = ["-c", "2048", "-t", "6"]
TUNED = ["-fitt", "384", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-b", "256", "-ub", "128"]


# Qwen2.5-Coder-7B-Instruct Q4_K_M blob (4,683,074,048 bytes) from the local ollama manifest, ChatML template
MODEL_7B = os.path.join(BLOBS, "sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463")
CHATML = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"


def m7b():
    run_config("m7b_qwen25coder_resident", ["-c", "2048", "-t", "6", "-ngl", "99"], model=MODEL_7B, template=CHATML)


def matrix():
    run_config("c0_default_fit", COMMON)
    run_config("c1_tuned_fit", COMMON + TUNED)
    off = offloaded_layers("c1_tuned_fit")
    base_ngl = off[0] if off else 15
    # leave room for the 1.3 GB draft model (~4 target layers worth of VRAM)
    spec_ngl = str(max(base_ngl - 4, 0))
    draft = ["--model-draft", MODEL_1B, "-ngld", "99", "--spec-draft-n-max", "4"]
    run_config("c2_ngram_simple", COMMON + TUNED + ["--spec-type", "ngram-simple",
               "--spec-ngram-simple-size-n", "4", "--spec-ngram-simple-size-m", "8"])
    run_config("c3_draft1b_k4", COMMON + TUNED[2:] + ["-ngl", spec_ngl] + draft + ["--spec-type", "draft-simple"])
    run_config("c4_draft1b_plus_ngram", COMMON + TUNED[2:] + ["-ngl", spec_ngl] + draft +
               ["--spec-type", "ngram-simple,draft-simple", "--spec-ngram-simple-size-n", "4",
                "--spec-ngram-simple-size-m", "8"])
    run_config("c5_tuned_threads4", ["-c", "2048", "-t", "4"] + TUNED)


if __name__ == "__main__":
    os.makedirs(RUNS_DIR, exist_ok=True)
    if len(sys.argv) >= 2 and sys.argv[1] == "matrix":
        matrix()
    elif len(sys.argv) >= 2 and sys.argv[1] == "m7b":
        m7b()
    elif len(sys.argv) >= 4 and sys.argv[1] == "run":
        rest = sys.argv[3:]
        rest = rest[rest.index("--") + 1:] if "--" in rest else rest
        run_config(sys.argv[2], rest)
    else:
        print(__doc__)
