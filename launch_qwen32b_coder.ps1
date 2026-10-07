# Qwen2.5-Coder-32B Bare-Metal Hardware-Accelerated Server
# Hardware: NVIDIA RTX 5060 8GB GDDR7 + 32GB DDR4-2133 Host RAM
# Performance: ~3.0 tok/s pooled on complex coding tasks (+155% speedup via Speculative Decoding)

param(
    [int]$Port = 8089,
    [int]$Threads = 6,
    [int]$Ctx = 2048,
    [switch]$Background
)

$ErrorActionPreference = "Stop"

$OLLAMA_LIB = "<USERPROFILE>\AppData\Local\Programs\Ollama\lib\ollama"
$CUDA_DIR = "$OLLAMA_LIB\cuda_v13"
$LLAMA_SERVER = "$OLLAMA_LIB\llama-server.exe"

$MODEL_32B = "<USERPROFILE>\.ollama\models\blobs\sha256-ac3d1ba8aa77755dab3806d9024e9c385ea0d5b412d6bdf9157f8a4a7e9fc0d9"
$MODEL_DRAFT_1_5B = "<USERPROFILE>\.ollama\models\blobs\sha256-29d8c98fa6b098e200069bfb88b9508dc3e85586d20cba59f8dda9a808165104"

if (-not (Test-Path $MODEL_32B)) {
    Write-Error "Qwen2.5-Coder-32B model blob not found at: $MODEL_32B"
}
if (-not (Test-Path $MODEL_DRAFT_1_5B)) {
    Write-Error "Qwen2.5-Coder-1.5B drafter blob not found at: $MODEL_DRAFT_1_5B"
}

# Set environment
$env:GGML_BACKEND_PATH = "$CUDA_DIR\ggml-cuda.dll"
$env:PATH = "$OLLAMA_LIB;$CUDA_DIR;" + $env:PATH
$env:CUDA_VISIBLE_DEVICES = "0"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " 🚀 Harness Bare-Metal Model Engine: Qwen2.5-Coder-32B" -ForegroundColor Cyan
Write-Host " Speculative Drafter: Qwen2.5-Coder-1.5B (100% in GDDR7)" -ForegroundColor Cyan
Write-Host " Endpoint: http://127.0.0.1:$Port/v1/chat/completions" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

$serverArgs = @(
    "--model", $MODEL_32B,
    "--load-mode", "none",
    "-np", "1",
    "--port", "$Port",
    "--host", "127.0.0.1",
    "--no-webui",
    "-c", "$Ctx",
    "-t", "$Threads",
    "-ctk", "q8_0",
    "-ctv", "q8_0",
    "-fa", "on",
    "-b", "256",
    "-ub", "128",
    "-ngl", "17",
    "--model-draft", $MODEL_DRAFT_1_5B,
    "-ngld", "99",
    "--spec-draft-n-max", "4",
    "--spec-type", "draft-simple"
)

if ($Background) {
    $logOut = "C:\Harness\server.log"
    $logErr = "C:\Harness\server_err.log"
    Start-Process -FilePath $LLAMA_SERVER -ArgumentList $serverArgs -RedirectStandardOutput $logOut -RedirectStandardError $logErr -WindowStyle Minimized
    Write-Host "[LAUNCHED] Server starting in background on port $Port (Logs: $logOut)" -ForegroundColor Green
} else {
    & $LLAMA_SERVER @serverArgs
}
