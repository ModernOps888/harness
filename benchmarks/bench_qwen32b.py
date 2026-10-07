"""
Benchmark driver for Qwen2.5-Coder-32B on bare-metal PC (RTX 5060 8GB + 32GB DDR4-2133).
Evaluates raw baseline vs. speculative decoding with Qwen2.5-Coder-1.5B drafter on complex coding tasks.

Stores raw captures:
  - server_stdout.txt
  - response_<prompt>.json
  - meta.json
  - QWEN32B_RESULTS.md (summarized directly from raw JSON timings)
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
MANIFEST_32B = os.path.join(OLLAMA_MODELS, "manifests", "registry.ollama.ai", "library", "qwen2.5-coder", "32b")

# Drafter: Qwen2.5-Coder-1.5B
MODEL_DRAFT_1_5B = os.path.join(BLOBS, "sha256-29d8c98fa6b098e200069bfb88b9508dc3e85586d20cba59f8dda9a808165104")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "runs", "qwen32b_coder")
PORT = 8089
N_PREDICT = 192  # Deep enough for complete implementations and tests

CHATML_TEMPLATE = (
    "<|im_start|>system\n"
    "You are an expert programming assistant and systems software engineer. "
    "Write clean, optimal, and correct code with zero unnecessary filler.<|im_end|>\n"
    "<|im_start|>user\n{p}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

CODING_PROMPTS = {
    "humaneval_merge_intervals": (
        "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges all overlapping intervals. "
        "Include an `if __name__ == '__main__':` block that runs these exact assertions:\n"
        "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
        "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
        "assert merge_intervals([]) == []\n"
        "assert merge_intervals([[1,4],[0,4]]) == [[0,4]]\n"
        "assert merge_intervals([[1,4],[2,3]]) == [[1,4]]\n"
        "print('ALL_TESTS_PASSED')\n"
        "Output ONLY the complete executable python code block."
    ),
    "lru_cache_rust": (
        "Write a production-quality, thread-safe LRU Cache in Rust using standard library synchronization. "
        "Include O(1) get/put operations, evictions, capacity checks, and complete unit tests."
    ),
    "async_deadlock_audit": (
        "Analyze this asynchronous Rust pattern for deadlocks and race conditions:\n"
        "```rust\n"
        "use std::sync::Arc;\n"
        "use tokio::sync::Mutex;\n"
        "struct State { count: u32, waiters: Vec<tokio::sync::oneshot::Sender<()>> }\n"
        "async fn notify(state: Arc<Mutex<State>>) {\n"
        "    let mut lock = state.lock().await;\n"
        "    lock.count += 1;\n"
        "    for tx in lock.waiters.drain(..) { let _ = tx.send(()); }\n"
        "}\n"
        "```\n"
        "Identify potential deadlocks or unbounded allocations and provide the clean idiomatic fix."
    ),
}


class MEMSTAT(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)
    ]


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


def resolve_32b_model():
    if not os.path.exists(MANIFEST_32B):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_32B}. Ensure model is pulled.")
    with open(MANIFEST_32B, "r", encoding="utf-8") as f:
        data = json.load(f)
    for layer in data.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer.get("digest", "").replace("sha256:", "sha256-")
            blob_path = os.path.join(BLOBS, digest)
            if os.path.exists(blob_path):
                return blob_path
    raise FileNotFoundError("Could not find model blob in manifest.")


def http_json(path, payload=None, timeout=600):
    url = f"http://127.0.0.1:{PORT}{path}"
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
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


def run_test_config(name, model_path, server_args):
    cfg_dir = os.path.join(OUT_DIR, name)
    os.makedirs(cfg_dir, exist_ok=True)

    meta_path = os.path.join(cfg_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            all_done = all(os.path.exists(os.path.join(cfg_dir, f"response_{p}.json")) for p in CODING_PROMPTS.keys())
            if saved.get("finished") and all_done:
                print(f"\n=======================================================", flush=True)
                print(f"[{name}] ALREADY COMPLETED AT {saved.get('finished')}. SKIPPING.", flush=True)
                print(f"=======================================================", flush=True)
                return saved
        except Exception:
            pass

    cmd = [
        LLAMA_SERVER, "--model", model_path, "--load-mode", "none", "-np", "1",
        "--port", str(PORT), "--host", "127.0.0.1", "--no-webui", "-lv", "4"
    ] + server_args

    meta = {
        "name": name,
        "model": model_path,
        "cmd": cmd,
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
        "free_ram_gb_before_launch": free_ram_gb()[0],
        "total_ram_gb": free_ram_gb()[1],
        "n_predict": N_PREDICT,
        "smi_before_launch": smi(),
    }
    print(f"\n=======================================================", flush=True)
    print(f"[{name}] LAUNCHING: {' '.join(server_args)}", flush=True)
    print(f"=======================================================", flush=True)

    log_fd = open(os.path.join(cfg_dir, "server_stdout.txt"), "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, env=server_env(), stdout=log_fd, stderr=subprocess.STDOUT)
    try:
        ok, msg = wait_ready(proc)
        meta["ready"] = {"ok": ok, "msg": msg}
        print(f"[{name}] {msg}", flush=True)
        if not ok:
            return meta

        meta["smi_after_load"] = smi()
        meta["free_ram_gb_after_load"] = free_ram_gb()[0]

        # Warm-up pass
        warm = http_json("/completion", {
            "prompt": CHATML_TEMPLATE.format(p="fn test() {}"),
            "n_predict": 8, "temperature": 0.0, "top_k": 1, "seed": 1, "cache_prompt": False
        })
        with open(os.path.join(cfg_dir, "response_warmup.json"), "wb") as f:
            f.write(warm)

        # Complex coding prompts evaluation
        for p_key, p_text in CODING_PROMPTS.items():
            payload = {
                "prompt": CHATML_TEMPLATE.format(p=p_text),
                "n_predict": N_PREDICT,
                "temperature": 0.0,
                "top_k": 1,
                "seed": 1,
                "cache_prompt": False,
                "stream": False
            }
            t0 = time.time()
            raw = http_json("/completion", payload)
            wall = time.time() - t0
            with open(os.path.join(cfg_dir, f"response_{p_key}.json"), "wb") as f:
                f.write(raw)

            data = json.loads(raw)
            t = data.get("timings", {})
            tok_s = t.get("predicted_per_second", 0.0)
            dn = t.get("draft_n")
            da = t.get("draft_n_accepted")
            draft_str = f"draft={da}/{dn} ({100*da/dn:.0f}%)" if dn else "draft=none"

            exec_status = ""
            if p_key == "humaneval_merge_intervals":
                code_text = data.get("content", "").strip()
                if code_text.startswith("```python"):
                    code_text = code_text[9:].strip()
                elif code_text.startswith("```"):
                    code_text = code_text[3:].strip()
                if code_text.endswith("```"):
                    code_text = code_text[:-3].strip()

                lines = code_text.splitlines()
                valid_lines = [l for l in lines if not (l.strip().startswith("assert") and not (l.strip().endswith("]") or l.strip().endswith(")")))]
                test_harness = (
                    "\nif 'ALL_TESTS_PASSED' not in locals():\n"
                    "    assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
                    "    assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
                    "    assert merge_intervals([]) == []\n"
                    "    assert merge_intervals([[1,4],[0,4]]) == [[0,4]]\n"
                    "    assert merge_intervals([[1,4],[2,3]]) == [[1,4]]\n"
                    "    print('ALL_TESTS_PASSED')\n"
                )
                py_code = "\n".join(valid_lines) + test_harness
                try:
                    res = subprocess.run([sys.executable, "-c", py_code], capture_output=True, text=True, timeout=10)
                    if res.returncode == 0 and "ALL_TESTS_PASSED" in res.stdout:
                        exec_status = " | TEST EXECUTION: PASSED (100% asserts passed)"
                    else:
                        exec_status = f" | TEST EXECUTION: FAILED ({res.stderr.strip()[:60]})"
                except Exception as ex:
                    exec_status = f" | TEST EXECUTION: ERROR ({ex})"

            print(f"[{name}] {p_key}: wall={wall:.1f}s | {tok_s:.2f} tok/s | {draft_str}{exec_status}", flush=True)

        meta["smi_after_run"] = smi()
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
    finally:
        kill_tree(proc)
        log_fd.close()
        with open(os.path.join(cfg_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return meta


def summarize():
    cfgs = sorted(c for c in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, c)))
    lines = [
        "# Qwen2.5-Coder-32B Factual Benchmark & Speculative Optimization",
        "",
        "Evaluated on bare metal: NVIDIA RTX 5060 8GB GDDR7 + 32GB DDR4-2133 Host RAM.",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Configuration | Offloaded Layers | Tokens | Total Gen Time | **Pooled tok/s** | Per-Prompt tok/s (LRU / Async / VecLog) | Draft Acceptance |",
        "|---|---|---|---|---|---|---|"
    ]

    for c in cfgs:
        d = os.path.join(OUT_DIR, c)
        meta_p = os.path.join(d, "meta.json")
        log_p = os.path.join(d, "server_stdout.txt")
        if not os.path.exists(meta_p) or not os.path.exists(log_p):
            continue

        log_txt = open(log_p, encoding="utf-8", errors="replace").read()
        m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log_txt)
        gl = f"{m.group(1)}/{m.group(2)}" if m else "n/a"

        res = {}
        for p in CODING_PROMPTS.keys():
            rf = os.path.join(d, f"response_{p}.json")
            if os.path.exists(rf):
                res[p] = json.loads(open(rf, "rb").read())

        if not res:
            continue

        tot_n = sum(r["timings"]["predicted_n"] for r in res.values())
        tot_ms = sum(r["timings"]["predicted_ms"] for r in res.values())
        pooled_speed = (1000.0 * tot_n / tot_ms) if tot_ms > 0 else 0.0

        per = " / ".join(f"{res[p]['timings']['predicted_per_second']:.2f}" if p in res else "-" for p in CODING_PROMPTS.keys())
        dn = sum(r["timings"].get("draft_n", 0) or 0 for r in res.values())
        da = sum(r["timings"].get("draft_n_accepted", 0) or 0 for r in res.values())
        acc = f"{da}/{dn} ({100 * da / dn:.0f}%)" if dn else "n/a"

        lines.append(f"| {c} | {gl} | {tot_n} | {tot_ms/1000.0:.1f}s | **{pooled_speed:.2f}** | {per} | {acc} |")

    lines.extend([
        "",
        "## Prompt Quality & Code Assessment",
        ""
    ])

    for c in cfgs:
        d = os.path.join(OUT_DIR, c)
        lines.append(f"### Configuration: `{c}`")
        for p in CODING_PROMPTS.keys():
            rf = os.path.join(d, f"response_{p}.json")
            if os.path.exists(rf):
                data = json.loads(open(rf, "rb").read())
                snippet = data.get("content", "").strip()
                lines.append(f"#### Prompt `{p}` Sample:")
                lines.append("```rust\n" + snippet[:400] + ("..." if len(snippet) > 400 else "") + "\n```\n")

    summary_file = os.path.join(OUT_DIR, "QWEN32B_RESULTS.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSummary successfully written to {summary_file}")


def get_offloaded_layers(name):
    p = os.path.join(OUT_DIR, name, "server_stdout.txt")
    if not os.path.exists(p):
        return None
    text = open(p, encoding="utf-8", errors="replace").read()
    m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
    if m:
        return int(m.group(1))
    m = re.search(r"CUDA\d+.*?(\d+)\s+layers", text)
    if m:
        return int(m.group(1))
    return None


def main():
    model_32b = resolve_32b_model()
    print(f"Target Qwen2.5-Coder-32B blob: {model_32b}")
    print(f"Drafter Qwen2.5-Coder-1.5B blob: {MODEL_DRAFT_1_5B}")

    # Config 1: Raw Baseline (auto-fit layer offload, no speculative)
    run_test_config("c0_raw_baseline", model_32b, ["-c", "2048", "-t", "6"])

    # Config 2: Tuned KV cache + flash attention
    run_test_config("c1_tuned_kv", model_32b, [
        "-c", "2048", "-t", "6", "-fitt", "384", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-b", "256", "-ub", "128"
    ])

    off = get_offloaded_layers("c1_tuned_kv") or get_offloaded_layers("c0_raw_baseline") or 14
    spec_ngl = max(0, off - 4)
    print(f"\n[DRAFT CONFIG] Base offloaded: {off} layers -> Allocating {spec_ngl} layers to 32B verifier + all layers to 1.5B drafter\n", flush=True)

    # Config 3: Speculative Decoding with Qwen2.5-Coder-1.5B Drafter
    # Offload 1.5B fully into GPU VRAM (-ngld 99), K=4 draft window
    run_test_config("c2_spec_draft_1_5b", model_32b, [
        "-c", "2048", "-t", "6", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-b", "256", "-ub", "128",
        "-ngl", str(spec_ngl),
        "--model-draft", MODEL_DRAFT_1_5B,
        "-ngld", "99",
        "--spec-draft-n-max", "4",
        "--spec-type", "draft-simple"
    ])

    # Config 4: N-gram Lookup Speculative Decoding (No extra VRAM needed)
    run_test_config("c3_spec_ngram", model_32b, [
        "-c", "2048", "-t", "6", "-fitt", "384", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-b", "256", "-ub", "128",
        "--spec-type", "ngram-simple",
        "--spec-ngram-simple-size-n", "4",
        "--spec-ngram-simple-size-m", "8"
    ])

    summarize()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        main()
