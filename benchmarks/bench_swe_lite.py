"""HARNESS-SWE-Lite runner: a tool-using agent loop through the HARNESS OpenAI endpoint.

  python bench_swe_lite.py --validate                 # prove every task fails before / passes after the reference fix
  python bench_swe_lite.py --endpoint http://127.0.0.1:8090/v1 [--tasks t01,t02] [--max-turns 14]

The agent only sees the issue + visible tests. It edits files with native OpenAI tool calls
(list_dir / read_file / write_file / replace_in_file / run_tests). Scoring happens AFTER the loop:
the hidden test file is copied in and pytest must pass on visible + hidden tests.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "swe_lite"))
from tasks import TASKS  # noqa: E402

OUT_ROOT = os.path.join(HERE, "runs", "swe_lite")

TOOLS = [
    {"type": "function", "function": {"name": "list_dir", "description": "List files in a directory of the project (use '.' for the root).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file of the project.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file with the full given content.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "replace_in_file", "description": "Replace exactly one occurrence of `old` with `new` in a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the project's visible pytest tests and return the output.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM = ("You are an expert software engineer working in a small Python project. Fix the bug described in the issue by "
          "using the provided tools: inspect the code, edit it, and run the tests. Keep going until the visible tests pass, "
          "and make the fix general (handle edge cases the issue implies), because hidden tests will also be run. "
          "When finished, reply with a one-sentence summary and no tool call.")


def materialize(task, root, solution=False, hidden=False):
    files = dict(task["files"])
    if solution:
        files.update(task["solution"])
    if hidden:
        files.update(task["hidden"])
    for rel, content in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(content.lstrip("\n"))


def pytest(root, names=None, timeout=120):
    args = [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"] + (names or [])
    try:
        r = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)[-3500:]
    except subprocess.TimeoutExpired:
        return 124, "pytest timed out"


def visible_tests(task):
    return [f for f in task["files"] if os.path.basename(f).startswith("test_")]


def safe(root, rel):
    p = os.path.abspath(os.path.join(root, rel))
    if not p.startswith(os.path.abspath(root) + os.sep) and p != os.path.abspath(root):
        raise ValueError("path escapes the project")
    return p


def run_tool(task, root, name, args):
    try:
        if name == "list_dir":
            p = safe(root, args.get("path", "."))
            return "\n".join(sorted(x for x in os.listdir(p) if not x.startswith((".", "__pycache__"))))
        if name == "read_file":
            return open(safe(root, args["path"]), encoding="utf-8").read()
        if name == "write_file":
            p = safe(root, args["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w", encoding="utf-8", newline="\n").write(args["content"])
            return f"wrote {len(args['content'])} chars to {args['path']}"
        if name == "replace_in_file":
            p = safe(root, args["path"])
            s = open(p, encoding="utf-8").read()
            n = s.count(args["old"])
            if n != 1:
                return f"ERROR: `old` occurs {n} times (must be exactly 1)"
            open(p, "w", encoding="utf-8", newline="\n").write(s.replace(args["old"], args["new"]))
            return "replaced 1 occurrence"
        if name == "run_tests":
            rc, out = pytest(root, visible_tests(task))
            return f"exit code {rc}\n{out}"
        return f"ERROR: unknown tool {name}"
    except Exception as e:  # noqa
        return f"ERROR: {type(e).__name__}: {e}"


def chat(endpoint, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "temperature": 0.2, "max_tokens": max_tokens}
    req = urllib.request.Request(endpoint + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def run_agent(task, endpoint, model, max_turns, max_tokens):
    root = tempfile.mkdtemp(prefix=f"swe_{task['id']}_")
    materialize(task, root)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Issue:\n{task['issue']}\n\nProject files: {', '.join(sorted(task['files']))}"}]
    rec = {"id": task["id"], "turns": 0, "tool_calls": [], "prompt_tokens": 0, "completion_tokens": 0, "decode_tps": [], "error": None}
    t0 = time.time()
    for turn in range(max_turns):
        try:
            r = chat(endpoint, model, messages, max_tokens)
        except urllib.error.HTTPError as e:
            rec["error"] = f"HTTP {e.code}: {e.read().decode()[:300]}"
            break
        except Exception as e:  # noqa
            rec["error"] = repr(e)
            break
        rec["turns"] += 1
        u = r.get("usage", {})
        rec["prompt_tokens"] += u.get("prompt_tokens", 0)
        rec["completion_tokens"] += u.get("completion_tokens", 0)
        if r.get("timings"):
            rec["decode_tps"].append(r["timings"].get("predicted_per_second", 0))
        elif u.get("tokens_per_second"):
            rec["decode_tps"].append(u["tokens_per_second"])
        msg = r["choices"][0]["message"]
        if r["choices"][0].get("finish_reason") == "length":
            rec["truncated_turns"] = rec.get("truncated_turns", 0) + 1
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")} | ({"content": msg.get("content") or ""} if not msg.get("tool_calls") else {}))
        calls = msg.get("tool_calls") or []
        if not calls:
            rec["final"] = (msg.get("content") or "")[:300]
            break
        for c in calls:
            fn = c["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
                out = "ERROR: tool arguments were not valid JSON"
            else:
                out = run_tool(task, root, fn["name"], args)
            rec["tool_calls"].append(fn["name"])
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": out})
    rec["wall_s"] = round(time.time() - t0, 1)
    # Scoring (tamper-proof): copy the agent's final tree, RESTORE the original visible tests, delete
    # pytest configuration files the agent could have used to skip tests, then add the hidden tests.
    score_root = tempfile.mkdtemp(prefix=f"score_{task['id']}_")
    shutil.copytree(root, score_root, dirs_exist_ok=True)
    for junk in ("conftest.py", "pytest.ini", "setup.cfg", "tox.ini", "pyproject.toml", "sitecustomize.py", "usercustomize.py"):
        jp = os.path.join(score_root, junk)
        if os.path.exists(jp):
            os.remove(jp)
            rec.setdefault("removed_config_files", []).append(junk)
    for rel, content in task["files"].items():
        if os.path.basename(rel).startswith("test_"):
            with open(os.path.join(score_root, rel), "w", encoding="utf-8", newline="\n") as f:
                f.write(content.lstrip("\n"))
    materialize({"files": {}, "solution": {}, "hidden": task["hidden"]}, score_root, hidden=True)
    vrc, vout = pytest(score_root, visible_tests(task))
    hrc, hout = pytest(score_root, list(task["hidden"]))
    rec.update(visible_pass=vrc == 0, hidden_pass=hrc == 0, resolved=(vrc == 0 and hrc == 0),
               pytest_tail=(hout if hrc else vout)[-600:])
    shutil.rmtree(score_root, ignore_errors=True)
    rec["messages"] = messages
    shutil.rmtree(root, ignore_errors=True)
    return rec


def validate(repeats=3):
    ok = True
    for t in TASKS:
        a = tempfile.mkdtemp()
        materialize(t, a, hidden=True)
        brc, _ = pytest(a, visible_tests(t))
        hbrc, _ = pytest(a, list(t["hidden"]))
        shutil.rmtree(a, ignore_errors=True)
        rcs = []
        for _ in range(repeats):  # flakiness check: the reference solution must pass every time
            b = tempfile.mkdtemp()
            materialize(t, b, solution=True, hidden=True)
            src, sout = pytest(b, visible_tests(t) + list(t["hidden"]))
            shutil.rmtree(b, ignore_errors=True)
            rcs.append(src)
        good = brc != 0 and hbrc != 0 and all(r == 0 for r in rcs)
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {t['id']}: buggy visible rc={brc} hidden rc={hbrc} (both must be !=0); "
              f"reference solution x{repeats} rc={rcs} (all must be 0)")
        if any(rcs):
            print(sout[-400:])
    print("ALL TASKS VALID" if ok else "VALIDATION FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    a = ap.parse_args()
    if a.validate:
        sys.exit(validate())
    wanted = set(filter(None, a.tasks.split(",")))
    sel = [t for t in TASKS if not wanted or any(t["id"].startswith(w) for w in wanted)]
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []
    for t in sel:
        print(f"[{t['id']}] running ...", flush=True)
        r = run_agent(t, a.endpoint, a.model, a.max_turns, a.max_tokens)
        results.append(r)
        print(f"[{t['id']}] resolved={r['resolved']} visible={r['visible_pass']} hidden={r['hidden_pass']} turns={r['turns']} "
              f"tools={len(r['tool_calls'])} wall={r['wall_s']}s err={r['error']}", flush=True)
        json.dump(results, open(os.path.join(OUT_ROOT, f"{a.tag}.json"), "w", encoding="utf-8"), indent=1)
    n = len(results)
    res = sum(r["resolved"] for r in results)
    print(f"\nRESOLVED {res}/{n} = {100*res/max(n,1):.1f}%")


if __name__ == "__main__":
    main()
