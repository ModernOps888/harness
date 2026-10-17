# HARNESS Elite 30B-32B Coder Bare-Metal Hardware-Accelerated Server
# Hardware: NVIDIA RTX 5060 8,150 MiB GDDR7 + Intel Core i5-10400F + 32 GB DDR4-2133
# Verified Modes:
#   -Mode MoE30B   (Default): Qwen3-Coder-30B-A3B Sparse MoE (-ngl 99 --n-cpu-moe 32) -> 16.37 tok/s pooled / 16.53 tok/s peak, 78.0 tok/s prefill
#   -Mode Dense32B          : Qwen2.5-Coder-32B (-ngl 20) + 0.5B Drafter (-ngld 99)   -> 3.90 tok/s pooled / 4.80 tok/s peak
#   -Mode Dense14B          : Qwen2.5-Coder-14B-Q4_K_S (-ngl 38) + 0.5B Drafter       -> 15.89 tok/s pooled / 19.17 tok/s peak

param(
    [ValidateSet("MoE30B", "Dense32B", "Dense14B")]
    [string]$Mode = "MoE30B",
    [int]$Port = 8089,
    [int]$Threads = 6,
    [int]$Ctx = 2048,
    [switch]$Background
)

$ErrorActionPreference = "Stop"

# Dynamic environment resolution (zero hardcoded user paths)
$USER_HOME = $env:USERPROFILE
$LOCAL_APP_DATA = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { "$USER_HOME\AppData\Local" }
$OLLAMA_LIB = if ($env:OLLAMA_LIB) { $env:OLLAMA_LIB } else { "$LOCAL_APP_DATA\Programs\Ollama\lib\ollama" }
$CUDA_DIR = "$OLLAMA_LIB\cuda_v13"
$LLAMA_SERVER = "$OLLAMA_LIB\llama-server.exe"

$OLLAMA_MODELS = if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS } else { "$USER_HOME\.ollama\models" }
$BLOBS = "$OLLAMA_MODELS\blobs"

$BLOB_MOE_30B    = "$BLOBS\sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a"
$BLOB_DENSE_32B  = "$BLOBS\sha256-ac3d1ba8aa77755dab3806d9024e9c385ea0d5b412d6bdf9157f8a4a7e9fc0d9"
$BLOB_DENSE_14B  = "$BLOBS\sha256-0119ca9464dd3f5f759846e0fd5e0f149a7a5f1df77cad572ff41756f8e2d3ca"
$BLOB_DRAFT_0_5B = "$BLOBS\sha256-20693aeb02c63304e263a72453b6ab89e1c700a87c6948cac523ac1e6f7cade0"

# Set environment
$env:GGML_BACKEND_PATH = "$CUDA_DIR\ggml-cuda.dll"
$env:PATH = "$OLLAMA_LIB;$CUDA_DIR;" + $env:PATH
$env:CUDA_VISIBLE_DEVICES = "0"

$commonArgs = @(
    "--load-mode", "none",
    "-np", "1",
    "--port", "$Port",
    "--host", "127.0.0.1",
    "--no-webui",
    "-c", "$Ctx",
    "-t", "$Threads",
    "-tb", "10",
    "--no-op-offload",
    "-ctk", "q8_0",
    "-ctv", "q4_0",
    "-fa", "on",
    "-b", "512",
    "-ub", "256",
    "--cache-reuse", "64",
    "--prio", "2",
    "--prio-batch", "3",
    "--poll", "100"
)

if ($Mode -eq "MoE30B") {
    if (-not (Test-Path $BLOB_MOE_30B)) { Write-Error "Qwen3-Coder-30B-A3B blob not found at: $BLOB_MOE_30B" }
    Write-Host "========================================================================" -ForegroundColor Cyan
    Write-Host " ⚡ HARNESS Bare-Metal Engine: Qwen3-Coder-30B-A3B Sparse MoE (30.5B)" -ForegroundColor Cyan
    Write-Host " Offload: 100% Attention/Router/KV + 16 Full MoE Layers in 8GB GDDR7" -ForegroundColor Cyan
    Write-Host " Verified Speed: 16.37 tok/s pooled | 16.53 tok/s peak | 78.0 tok/s prefill" -ForegroundColor Green
    Write-Host " Endpoint: http://127.0.0.1:$Port/v1/chat/completions" -ForegroundColor Yellow
    Write-Host "========================================================================" -ForegroundColor Cyan
    $serverArgs = @("--model", $BLOB_MOE_30B, "-ngl", "99", "--n-cpu-moe", "32") + $commonArgs
} elseif ($Mode -eq "Dense32B") {
    if (-not (Test-Path $BLOB_DENSE_32B)) { Write-Error "Qwen2.5-Coder-32B blob not found at: $BLOB_DENSE_32B" }
    if (-not (Test-Path $BLOB_DRAFT_0_5B)) { Write-Error "Qwen2.5-Coder-0.5B drafter blob not found at: $BLOB_DRAFT_0_5B" }
    Write-Host "========================================================================" -ForegroundColor Cyan
    Write-Host " ⚡ HARNESS Bare-Metal Engine: Qwen2.5-Coder-32B Dense + 0.5B Drafter" -ForegroundColor Cyan
    Write-Host " Offload: 20 Dense 32B Layers + 100% 0.5B Drafter in 8GB GDDR7" -ForegroundColor Cyan
    Write-Host " Verified Speed: 3.90 tok/s pooled | 4.80 tok/s peak (96.2% draft acc)" -ForegroundColor Green
    Write-Host " Endpoint: http://127.0.0.1:$Port/v1/chat/completions" -ForegroundColor Yellow
    Write-Host "========================================================================" -ForegroundColor Cyan
    $serverArgs = @(
        "--model", $BLOB_DENSE_32B,
        "-ngl", "20",
        "--model-draft", $BLOB_DRAFT_0_5B,
        "-ngld", "99",
        "-ctkd", "q8_0",
        "-ctvd", "q4_0",
        "--spec-draft-n-max", "7",
        "--spec-draft-n-min", "1",
        "--spec-draft-p-min", "0.68",
        "--spec-type", "draft-simple"
    ) + $commonArgs
} else {
    if (-not (Test-Path $BLOB_DENSE_14B)) { Write-Error "Qwen2.5-Coder-14B blob not found at: $BLOB_DENSE_14B" }
    if (-not (Test-Path $BLOB_DRAFT_0_5B)) { Write-Error "Qwen2.5-Coder-0.5B drafter blob not found at: $BLOB_DRAFT_0_5B" }
    Write-Host "========================================================================" -ForegroundColor Cyan
    Write-Host " ⚡ HARNESS Bare-Metal Engine: Qwen2.5-Coder-14B-Q4_K_S + 0.5B Drafter" -ForegroundColor Cyan
    Write-Host " Offload: 38/49 Layers + 100% 0.5B Drafter in 8GB GDDR7" -ForegroundColor Cyan
    Write-Host " Verified Speed: 15.89 tok/s pooled | 19.17 tok/s peak (98.7% draft acc)" -ForegroundColor Green
    Write-Host " Endpoint: http://127.0.0.1:$Port/v1/chat/completions" -ForegroundColor Yellow
    Write-Host "========================================================================" -ForegroundColor Cyan
    $serverArgs = @(
        "--model", $BLOB_DENSE_14B,
        "-ngl", "38",
        "--model-draft", $BLOB_DRAFT_0_5B,
        "-ngld", "99",
        "-ctkd", "q8_0",
        "-ctvd", "q4_0",
        "--spec-draft-n-max", "7",
        "--spec-draft-n-min", "1",
        "--spec-draft-p-min", "0.68",
        "--spec-type", "draft-simple"
    ) + $commonArgs
}

$baseDir = if ($PSScriptRoot) { $PSScriptRoot } else { "." }

if ($Background) {
    $logOut = "$baseDir\server.log"
    $logErr = "$baseDir\server_err.log"
    Start-Process -FilePath $LLAMA_SERVER -ArgumentList $serverArgs -RedirectStandardOutput $logOut -RedirectStandardError $logErr -WindowStyle Minimized
    Write-Host "[LAUNCHED] Server ($Mode) starting in background on port $Port (Logs: $logOut)" -ForegroundColor Green
} else {
    & $LLAMA_SERVER @serverArgs
}
