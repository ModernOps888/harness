"""
Compute the results table ONLY from raw captured files in benchmarks/runs/<config>/:
  server_stdout.txt   - unmodified llama-server output
  response_<p>.json   - verbatim /completion HTTP responses
  meta.json           - exact command line + raw nvidia-smi / RAM snapshots

Writes benchmarks/runs/RESULTS.md. Run:  python benchmarks/summarize_runs.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
PROMPTS = ["prose", "code", "quote"]
BASELINE = "c0_default_fit"


def load(cfg):
    d = os.path.join(RUNS, cfg)
    res = {}
    for p in PROMPTS:
        fp = os.path.join(d, f"response_{p}.json")
        if os.path.exists(fp):
            res[p] = json.loads(open(fp, "rb").read())
    meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
    log = open(os.path.join(d, "server_stdout.txt"), encoding="utf-8", errors="replace").read() \
        if os.path.exists(os.path.join(d, "server_stdout.txt")) else ""
    return res, meta, log


MODEL_BYTES = 26375113056  # size of the llama3.1:70b-instruct-q2_K blob on disk


def gpu_info(cfg, meta, log):
    """GPU layer split: (1) llama-server log, (2) verbose probe log layer_probe_<cfg>.txt,
    (3) explicit -ngl N in the recorded command line. Returns None if none of them has it."""
    pat = r"offloaded (\d+)/(\d+) layers to GPU"
    for src, text in (("server log", log), ("verbose probe log", None)):
        if text is None:
            fp = os.path.join(RUNS, f"layer_probe_{cfg}.txt")
            text = open(fp, encoding="utf-8", errors="replace").read() if os.path.exists(fp) else ""
        m = re.search(pat, text)
        if m:
            b = re.search(r"CUDA0 model buffer size =\s+([\d.]+) MiB", text)
            return {"offloaded": int(m.group(1)), "total": int(m.group(2)), "source": src,
                    "gpu_weights_mib": float(b.group(1)) if b else None}
    cmd = meta.get("cmd") or []
    if "-ngl" in cmd:
        return {"offloaded": min(int(cmd[cmd.index("-ngl") + 1]), 81), "total": 81, "source": "explicit -ngl",
                "gpu_weights_mib": None}
    return None


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    cfgs = sorted(c for c in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, c)))
    base_res = load(BASELINE)[0] if BASELINE in cfgs else {}
    lines = ["# Raw-captured results (computed from files in `benchmarks/runs/`)", "",
             "Every number below is derived by `summarize_runs.py` from the verbatim llama-server `timings` JSON.",
             "Model: `llama3.1:70b-instruct-q2_K` (all 70.55B parameters, 26,375,113,056 bytes), greedy decoding "
             "(temperature 0, top_k 1, seed 1).", ""]
    lines += ["| config | GPU layers | tokens | gen time (s) | **tok/s (pooled)** | per-prompt tok/s (prose / code / quote) "
              "| draft accepted | lossless vs baseline |",
              "|---|---|---|---|---|---|---|---|"]
    for c in cfgs:
        res, meta, log = load(c)
        if not res:
            err = "no responses (see server_stdout.txt)"
            lines.append(f"| {c} | - | - | - | - | {err} | - | - |")
            continue
        gi = gpu_info(c, meta, log)
        gl = f"{gi['offloaded']}/{gi['total']} ({gi['source']})" if gi else "n/a"
        tot_n = sum(r["timings"]["predicted_n"] for r in res.values())
        tot_ms = sum(r["timings"]["predicted_ms"] for r in res.values())
        per = " / ".join(f"{res[p]['timings']['predicted_per_second']:.2f}" if p in res else "-" for p in PROMPTS)
        dn = sum(r["timings"].get("draft_n", 0) or 0 for r in res.values())
        da = sum(r["timings"].get("draft_n_accepted", 0) or 0 for r in res.values())
        acc = f"{da}/{dn} ({100 * da / dn:.0f}%)" if dn else "n/a"
        if c.startswith("m7b"):
            ll = "n/a (different model: Qwen2.5-Coder-7B Q4_K_M)"
        elif c == BASELINE or not base_res:
            ll = "reference"
        else:
            parts = []
            for p in PROMPTS:
                if p in res and p in base_res:
                    a, b = res[p]["content"], base_res[p]["content"]
                    parts.append("identical" if a == b else f"diverges@char {common_prefix(a, b)}/{len(b)}")
            ll = "; ".join(parts)
        lines.append(f"| {c} | {gl} | {tot_n} | {tot_ms / 1000:.1f} | **{1000 * tot_n / tot_ms:.3f}** | {per} | {acc} | {ll} |")

    lines += ["", "## Exact command lines and hardware state (raw)", ""]
    for c in cfgs:
        _, meta, _ = load(c)
        if not meta:
            continue
        lines += [f"### {c}", "```", " ".join(f'"{x}"' if " " in x else x for x in meta.get("cmd", [])), "```",
                  f"- git HEAD: `{meta.get('git_head')}`  started: `{meta.get('started')}`  "
                  f"free RAM before launch: {meta.get('free_ram_gb_before_launch')} GB of {meta.get('total_ram_gb')} GB",
                  "- nvidia-smi before launch / after load:", "```",
                  meta.get("smi_before_launch", {}).get("raw", ""),
                  meta.get("smi_after_load", {}).get("raw", ""), "```", ""]
    open(os.path.join(RUNS, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:30]))
    write_json(cfgs, base_res)


KEY_LOG = re.compile(r"offloaded|model buffer|KV buffer|compute buffer|load time|n_threads|speculative|draft|"
                     r"fit|n_gpu_layers|flash|n_ctx\b", re.I)


def write_json(cfgs, base_res):
    """results.json: computed values + verbatim raw excerpts only. Nothing hand-typed."""
    out = {"generated_by": "benchmarks/summarize_runs.py", "model": "llama3.1:70b-instruct-q2_K",
           "baseline_config": BASELINE, "configs": []}
    hw = os.path.join(RUNS, "hardware.txt")
    out["hardware_raw"] = open(hw, encoding="utf-8", errors="replace").read() if os.path.exists(hw) else None
    for c in cfgs:
        res, meta, log = load(c)
        entry = {"config": c, "cmd": meta.get("cmd"), "started": meta.get("started"),
                 "git_head": meta.get("git_head"), "free_ram_gb_before_launch": meta.get("free_ram_gb_before_launch"),
                 "smi_after_load_raw": meta.get("smi_after_load", {}).get("raw"),
                 "log_excerpt": [ln for ln in log.splitlines() if KEY_LOG.search(ln)][:40],
                 "prompts": {}}
        gi = gpu_info(c, meta, log)
        entry["gpu_layers"] = gi
        tn = tm = dn = da = 0
        for p in PROMPTS:
            if p not in res:
                continue
            t = res[p]["timings"]
            tn += t["predicted_n"]
            tm += t["predicted_ms"]
            dn += t.get("draft_n", 0) or 0
            da += t.get("draft_n_accepted", 0) or 0
            same = None
            if c != BASELINE and not c.startswith("m7b") and p in base_res:
                same = res[p]["content"] == base_res[p]["content"]
            entry["prompts"][p] = {"timings_verbatim": t, "content": res[p]["content"], "identical_to_baseline": same}
        if tm:
            entry["pooled_tok_s"] = 1000 * tn / tm
            entry["tokens"] = tn
            entry["gen_seconds"] = tm / 1000
        entry["draft_n"], entry["draft_accepted"] = dn, da
        fp = os.path.join(RUNS, f"layer_probe_{c}.txt")
        if os.path.exists(fp):
            entry["layer_probe_excerpt"] = [ln.strip() for ln in open(fp, encoding="utf-8", errors="replace")
                                            if re.search(r"offload|model buffer|KV buffer|compute buffer|layers,", ln)][:30]
        # Computed (not measured) cross-check, only for plain autoregressive configs with a known GPU weight size:
        # every CPU-resident weight byte is read once per token => bytes_on_CPU * tok/s = implied read rate.
        if gi and gi.get("gpu_weights_mib") and "pooled_tok_s" in entry and dn == 0:
            cpu_gb = (MODEL_BYTES - gi["gpu_weights_mib"] * 2**20) / 1e9
            entry["cpu_resident_weights_gb"] = cpu_gb
            entry["implied_cpu_read_gb_s"] = cpu_gb * entry["pooled_tok_s"]
        out["configs"].append(entry)
    with open(os.path.join(RUNS, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    extra = ["", "## Computed cross-check: implied CPU-side weight read rate (plain autoregressive configs only)", "",
             "`(26,375,113,056 B - GPU model buffer from the verbose probe log) x pooled tok/s`. This is arithmetic on logged "
             "values, not a bandwidth measurement; `hardware.txt` gives the calculated DDR4 peak for comparison.", "",
             "| config | GPU layers | CPU-resident weights (GB) | pooled tok/s | implied read rate (GB/s) |", "|---|---|---|---|---|"]
    for e in out["configs"]:
        if "implied_cpu_read_gb_s" in e:
            g = e["gpu_layers"]
            extra.append(f"| {e['config']} | {g['offloaded']}/{g['total']} | {e['cpu_resident_weights_gb']:.2f} | "
                         f"{e['pooled_tok_s']:.3f} | {e['implied_cpu_read_gb_s']:.1f} |")
    with open(os.path.join(RUNS, "RESULTS.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(extra) + "\n")




if __name__ == "__main__":
    main()
