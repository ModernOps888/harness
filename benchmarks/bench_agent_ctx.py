"""Agent-context benchmark for the Qwen3-Coder-30B-A3B MoE engine on an 8 GB GPU.

VS Code native agent mode sends 10k+ token prompts (36 tool schemas + instructions + files), so a
2048-token context is unusable. This script measures, per llama-server configuration:
  * VRAM after load (nvidia-smi) and whether decode collapses (WDDM spill cliff),
  * prefill speed on a ~8k-token code prompt (cache_prompt=false => true prefill, no cache hit),
  * decode speed at that filled context,
  * decode speed on a short prompt (reference).
Every number is taken verbatim from the llama-server `timings` JSON. Output: runs/agent_ctx/<name>.json
"""
import json, os, subprocess, sys, time, urllib.request, glob

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "runs", "agent_ctx")
os.makedirs(OUT, exist_ok=True)

LOCAL = os.environ.get("LOCALAPPDATA", os.path.expanduser("~/AppData/Local"))
OLLAMA_LIB = os.environ.get("OLLAMA_LIB", os.path.join(LOCAL, "Programs", "Ollama", "lib", "ollama"))
CUDA_DIR = os.path.join(OLLAMA_LIB, "cuda_v13")
LLAMA = os.path.join(OLLAMA_LIB, "llama-server.exe")
BLOBS = os.path.join(os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models")), "blobs")
MOE30B = os.path.join(BLOBS, "sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a")
PORT = 8099

COMMON = ["-ngl", "99", "-np", "1", "--host", "127.0.0.1", "--port", str(PORT), "--no-webui",
          "-t", "6", "-tb", "10", "-fa", "on",
          "--cache-reuse", "64", "--prio", "2", "--prio-batch", "3", "--poll", "100", "--load-mode", "none"]

NOP = ["--no-op-offload", "-b", "512", "-ub", "256"]
GPUP = ["-b", "2048", "-ub", "1024"]
GPUP2K = ["-b", "4096", "-ub", "2048"]
KV_Q8Q4 = ["-ctk", "q8_0", "-ctv", "q4_0"]
KV_Q8Q8 = ["-ctk", "q8_0", "-ctv", "q8_0"]
KV_F16 = ["-ctk", "f16", "-ctv", "f16"]

# name -> (n_cpu_moe, ctx, kv args, batching args)
CONFIGS = {
    "a_ctx2k_ncmoe32_baseline": (32, 2048, KV_Q8Q4, NOP),
    "g_ctx16k_q8q8_ncmoe34": (34, 16384, KV_Q8Q8, NOP),
    "h_ctx16k_q8q4_ncmoe34_control": (34, 16384, KV_Q8Q4, NOP),
    "i_ctx16k_q8q8_ncmoe34_gpuprefill": (34, 16384, KV_Q8Q8, GPUP),
    "j_ctx16k_f16_ncmoe35": (35, 16384, KV_F16, NOP),
    "o_ctx16k_f16_ncmoe34_gpuprefill": (34, 16384, KV_F16, GPUP),
    "r_ctx32k_q8q8_ncmoe36_gpuprefill": (36, 32768, KV_Q8Q8, GPUP),
    "s_ctx32k_q8q8_ncmoe37_gpuprefill": (37, 32768, KV_Q8Q8, GPUP),
    "t_ctx64k_q8q8_ncmoe42_gpuprefill": (42, 65536, KV_Q8Q8, GPUP),
    "u_ctx64k_q8q8_ncmoe44_gpuprefill": (44, 65536, KV_Q8Q8, GPUP),
    "p_ctx32k_q8q8_ncmoe38_gpuprefill": (38, 32768, KV_Q8Q8, GPUP),
    "q_ctx32k_q8q8_ncmoe37_ub512": (37, 32768, KV_Q8Q8, ["-b", "1024", "-ub", "512"]),
    "k_ctx32k_q8q8_ncmoe36_gpuprefill": (36, 32768, KV_Q8Q8, GPUP),
    "l_ctx16k_q8q8_ncmoe34_ub2048": (34, 16384, KV_Q8Q8, GPUP2K),
    "m_ctx32k_q8q8_ncmoe35_gpuprefill": (35, 32768, KV_Q8Q8, GPUP),
    "n_ctx32k_q8q8_ncmoe34_gpuprefill": (34, 32768, KV_Q8Q8, GPUP),
}


def build_prompt(target_chars=int(os.environ.get('PROMPT_CHARS', '13000'))):
    parts = []
    for f in sorted(glob.glob(os.path.join(REPO, "crates", "harness-server", "src", "**", "*.rs"), recursive=True)) + \
            sorted(glob.glob(os.path.join(REPO, "crates", "harness-quant", "src", "*.rs"))) + sorted(glob.glob(os.path.join(REPO, "crates", "harness-core", "src", "**", "*.rs"), recursive=True)) + sorted(glob.glob(os.path.join(REPO, "crates", "harness-cli", "src", "*.rs"))):
        parts.append(f"// ===== {os.path.relpath(f, REPO)} =====\n" + open(f, encoding="utf-8", errors="replace").read())
        if sum(map(len, parts)) > target_chars:
            break
    code = "\n".join(parts)[:target_chars]
    return code + "\n\nQuestion: In one sentence, what does the first file in the listing above do?"


def post(path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def smi():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,clocks.current.memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def run(name):
    n_cpu_moe, ctx, kv, extra = CONFIGS[name]
    args = [LLAMA, "--model", MOE30B, "--n-cpu-moe", str(n_cpu_moe), "-c", str(ctx), "-lv", "4"] + COMMON + kv + extra
    env = os.environ.copy()
    env["GGML_BACKEND_PATH"] = os.path.join(CUDA_DIR, "ggml-cuda.dll")
    env["PATH"] = f"{OLLAMA_LIB};{CUDA_DIR};" + env["PATH"]
    env["CUDA_VISIBLE_DEVICES"] = "0"
    logf = open(os.path.join(OUT, f"{name}.server.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT)
    res = {"name": name, "args": args[1:], "kv_args": kv, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    assert "-ctk" in args and "-ctv" in args, "KV cache flags missing from server args"
    try:
        t0 = time.time()
        while time.time() - t0 < 400:
            try:
                if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()).get("status") == "ok":
                    break
            except Exception:
                time.sleep(2)
        else:
            res["error"] = "server did not become healthy"
            return res
        res["load_s"] = round(time.time() - t0, 1)
        res["smi_after_load"] = smi()

        short = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Write a Rust function that reverses a string slice."}],
                                              "max_tokens": 96, "temperature": 0, "seed": 1, "cache_prompt": False})
        res["short"] = {k: short["timings"][k] for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second")}

        long_p = build_prompt()
        lg = post("/v1/chat/completions", {"messages": [{"role": "user", "content": long_p}], "max_tokens": 64, "temperature": 0,
                                           "seed": 1, "cache_prompt": False}, timeout=1500)
        res["long"] = {k: lg["timings"][k] for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_per_second")}
        res["long_answer"] = lg["choices"][0]["message"]["content"][:200]
        res["smi_after_long"] = smi()
    except Exception as e:  # noqa
        res["error"] = repr(e)
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except Exception:
            proc.kill()
        logf.close()
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8"), indent=2)
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    for n in names:
        r = run(n)
        print(n, json.dumps({k: r.get(k) for k in ("load_s", "smi_after_load", "short", "long", "error")}), flush=True)
        time.sleep(3)
