import subprocess
import os
import sys
import time
import urllib.request
import json

LLAMA_SERVER = r"C:\Users\bchmi\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
CUDA_DIR = r"C:\Users\bchmi\AppData\Local\Programs\Ollama\lib\ollama\cuda_v13"
OLLAMA_LIB = r"C:\Users\bchmi\AppData\Local\Programs\Ollama\lib\ollama"
MODEL_70B = r"C:\Users\bchmi\.ollama\models\blobs\sha256-ba1103315c449ad06c9f5fd94230bde5bcf977f794af70afb107d29153c3cd53"
MODEL_1B = r"C:\Users\bchmi\.ollama\models\blobs\sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"

env = os.environ.copy()
env["GGML_BACKEND_PATH"] = os.path.join(CUDA_DIR, "ggml-cuda.dll")
env["PATH"] = f"{OLLAMA_LIB};{CUDA_DIR};" + env.get("PATH", "")
env["CUDA_VISIBLE_DEVICES"] = "0"

cmd = [
    LLAMA_SERVER,
    "--model", MODEL_70B,
    "-ngl", "22",
    "-c", "2048",
    "-b", "512",
    "-ub", "512",
    "--flash-attn", "auto",
    "--port", "8080",
    "--host", "127.0.0.1",
    "--no-webui"
]

print("Launching 70B with -ngl 22...")
p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

start = time.time()
server_ready = False
log_lines = []

while time.time() - start < 60:
    line = p.stdout.readline()
    if line:
        log_lines.append(line.strip())
        print(f"[{time.time()-start:.1f}s] {line.strip()}")
        if "listening on http://127.0.0.1:8080" in line:
            server_ready = True
            break
    if p.poll() is not None:
        print(f"Server exited with code {p.returncode}")
        break

if server_ready:
    print("\nSUCCESS: 70B loaded successfully!")
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/health")
        resp = urllib.request.urlopen(req)
        print(f"Health check status: {resp.status}")
    except Exception as e:
        print(f"Health check error: {e}")

print("Terminating server...")
p.terminate()
p.wait(timeout=10)
print("Done.")
