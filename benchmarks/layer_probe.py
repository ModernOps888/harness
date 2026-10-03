"""
Layer-split probe: relaunch llama-server with the SAME args as a matrix config plus -lv 4 (verbose),
wait until /health is ok, store the unmodified stdout, then kill it. No generation is done.
Output: benchmarks/runs/layer_probe_<config>.txt  (raw server stdout, verbatim)

Usage: python benchmarks/layer_probe.py <config_name> <llama-server args...>
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raw_capture as rc  # noqa: E402


def main():
    name, args = sys.argv[1], sys.argv[2:]
    cmd = [rc.LLAMA_SERVER, "--model", rc.MODEL_70B, "--load-mode", "none", "-np", "1", "--port", str(rc.PORT),
           "--host", "127.0.0.1", "--no-webui", "-lv", "4"] + args
    out = os.path.join(rc.RUNS_DIR, f"layer_probe_{name}.txt")
    with open(out, "w", encoding="utf-8", errors="replace") as fd:
        fd.write("# cmd: " + " ".join(cmd) + "\n")
        fd.flush()
        proc = subprocess.Popen(cmd, env=rc.server_env(), stdout=fd, stderr=subprocess.STDOUT)
        try:
            ok, msg = rc.wait_ready(proc)
            print(name, ok, msg, flush=True)
        finally:
            rc.kill_tree(proc)


if __name__ == "__main__":
    main()
