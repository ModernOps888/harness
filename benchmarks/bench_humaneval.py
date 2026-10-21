"""HumanEval / HumanEval+ pass@1 (greedy) through the HARNESS OpenAI endpoint, scored by EvalPlus.

  python bench_humaneval.py [--endpoint http://127.0.0.1:8090/v1] [--limit N] [--tag NAME]

Generation: one chat request per problem (temperature 0, top_k 1), prompt format as recommended by EvalPlus for
instruct models. Scoring: `evalplus.evaluate` on the extracted code (base tests = original HumanEval; plus = 80x more
EvalPlus tests). Generations, timings and the EvalPlus result are written to benchmarks/runs/humaneval/<tag>/.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def chat(endpoint, model, prompt, max_tokens):
    body = {"model": model, "temperature": 0, "top_k": 1, "seed": 1, "max_tokens": max_tokens, "cache_prompt": True,
            "messages": [
                {"role": "system", "content": "You are an expert Python programmer."},
                {"role": "user", "content": "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\n" + prompt.strip() + "\n```"},
            ]}
    req = urllib.request.Request(endpoint + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def build_solution(problem, text):
    code = extract_code(text)
    if f"def {problem['entry_point']}" not in code:
        return problem["prompt"] + code
    imports = [l for l in problem["prompt"].splitlines() if re.match(r"\s*(import|from)\s+\S+", l)]
    missing = [l for l in imports if l.strip() not in code]
    return ("\n".join(missing) + "\n" if missing else "") + code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    a = ap.parse_args()

    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    ids = sorted(problems, key=lambda k: int(k.split("/")[1]))
    if a.limit:
        ids = ids[: a.limit]
    out_dir = os.path.join(HERE, "runs", "humaneval", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    gen_path = os.path.join(out_dir, "generations.jsonl")
    done = {}
    if os.path.exists(gen_path):
        for line in open(gen_path, encoding="utf-8"):
            r = json.loads(line)
            done[r["task_id"]] = r
    t_start = time.time()
    with open(gen_path, "a", encoding="utf-8") as f:
        for i, tid in enumerate(ids):
            if tid in done:
                continue
            p = problems[tid]
            t0 = time.time()
            r = chat(a.endpoint, a.model, p["prompt"], a.max_tokens)
            text = r["choices"][0]["message"]["content"] or ""
            rec = {"task_id": tid, "solution": build_solution(p, text), "raw": text,
                   "completion_tokens": r.get("usage", {}).get("completion_tokens", 0),
                   "prompt_tokens": r.get("usage", {}).get("prompt_tokens", 0),
                   "decode_tps": (r.get("timings") or {}).get("predicted_per_second"),
                   "finish_reason": r["choices"][0].get("finish_reason"), "wall_s": round(time.time() - t0, 2)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[tid] = rec
            print(f"[{i+1}/{len(ids)}] {tid} tokens={rec['completion_tokens']} tps={rec['decode_tps']} {rec['wall_s']}s", flush=True)
    samples = os.path.join(out_dir, "samples.jsonl")
    with open(samples, "w", encoding="utf-8") as f:
        for tid in ids:
            f.write(json.dumps({"task_id": tid, "solution": done[tid]["solution"]}) + "\n")
    toks = sum(done[t]["completion_tokens"] for t in ids)
    tps_vals = [done[t]["decode_tps"] for t in ids if done[t].get("decode_tps")]
    mean_tps = round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None
    summary = {"problems": len(ids), "completion_tokens": toks, "mean_decode_tps": mean_tps,
               "generation_wall_s": round(sum(done[t].get("wall_s", 0.0) for t in ids), 1),
               "truncated": sum(1 for t in ids if done[t]["finish_reason"] == "length")}
    if a.limit and a.limit <= len(problems):
        import numpy as np
        from evalplus.data import get_human_eval_plus_hash
        from evalplus.eval import _poly, is_floats
        from evalplus.evaluate import get_groundtruth

        sub_problems = {tid: problems[tid] for tid in ids}
        dataset_hash = get_human_eval_plus_hash()
        expected_output = get_groundtruth(sub_problems, dataset_hash, [])

        def eval_split(prob, code, inputs, expected, atol):
            ns = {}
            try:
                exec(code, ns)
                fn = ns[prob["entry_point"]]
                for inp, exp in zip(inputs, expected):
                    out = fn(*inp)
                    if prob["entry_point"] == "find_zero":
                        assert abs(_poly(*inp, out)) <= (atol or 1e-6)
                        continue
                    eff_atol = 1e-6 if (atol == 0 and is_floats(exp)) else atol
                    if out != exp:
                        if eff_atol != 0:
                            assert type(out) == type(exp)
                            if isinstance(exp, (list, tuple)):
                                assert len(out) == len(exp)
                            assert np.allclose(out, exp, rtol=1e-07, atol=eff_atol)
                        else:
                            return False
                return True
            except BaseException:
                return False

        base_pass = 0
        plus_pass = 0
        per_task = {}
        for tid in ids:
            p = problems[tid]
            gt = expected_output[tid]
            atol = p.get("atol", 0)
            b_ok = eval_split(p, done[tid]["solution"], p["base_input"], gt["base"], atol)
            p_ok = b_ok and eval_split(p, done[tid]["solution"], p["plus_input"], gt["plus"], atol)
            base_pass += int(b_ok)
            plus_pass += int(p_ok)
            per_task[tid] = {"base_pass": b_ok, "plus_pass": p_ok}
        summary["base_pass"] = base_pass
        summary["plus_pass"] = plus_pass
        summary["base_pass_rate"] = round(base_pass / len(ids), 4)
        summary["plus_pass_rate"] = round(plus_pass / len(ids), 4)
        summary["per_task"] = per_task
        out_txt = (
            f"EvalPlus evaluation ({len(ids)} problems):\n"
            f"humaneval (base tests): {base_pass}/{len(ids)} = {100*base_pass/len(ids):.1f}%\n"
            f"humaneval+ (base + 80x EvalPlus extra tests): {plus_pass}/{len(ids)} = {100*plus_pass/len(ids):.1f}%\n"
        )
        open(os.path.join(out_dir, "evalplus_stdout.txt"), "w", encoding="utf-8").write(out_txt)
        json.dump(summary, open(os.path.join(out_dir, "generation_summary.json"), "w"), indent=1)
        print("generation summary:", summary, flush=True)
        print(out_txt, flush=True)
    else:
        json.dump(summary, open(os.path.join(out_dir, "generation_summary.json"), "w"), indent=1)
        print("generation summary:", summary, flush=True)
        r = subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval", "--samples", samples],
                           capture_output=True, text=True, cwd=out_dir)
        open(os.path.join(out_dir, "evalplus_stdout.txt"), "w", encoding="utf-8").write(r.stdout + "\n" + r.stderr)
        print(r.stdout[-1500:])


if __name__ == "__main__":
    main()
