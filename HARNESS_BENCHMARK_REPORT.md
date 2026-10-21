# HARNESS: Systems Telemetry, Long-Context Agent Engine & Benchmark Report

**Generated:** 2026-10-21 UTC  
**Engine Version:** 0.1.0 (Rust OpenAI/VS Code Proxy + `llama-server` GGUF Backend + `wgpu` Kernel Suite)  
**Host Architecture:** windows (x86_64) — Intel Core i5-10400F (6C/12T)  
**Target Accelerator:** NVIDIA GeForce RTX 5060 (8,151 MiB GDDR7 @ 448 GB/s, 128-bit @ 28 Gbps)  
**Host System RAM:** 31.9 GB Dual-Channel DDR4-2133 (34.1 GB/s theoretical peak, Discrete PCIe 3.0 x8)  

---

## 1. Architecture & Execution Path Separation

1. **Live Model Inference (`llama-server` + `harness-server` Proxy):**
   - Model weights (`.gguf` blobs) are executed by `llama-server` (`CUDA` + `CPU` hybrid MoE/speculative offload).
   - `harness-server` (`crates/harness-server`, default `127.0.0.1:8090`) is a lossless Rust/Axum OpenAI proxy that forwards `/v1/chat/completions` (`tools`, `tool_choice`, `parallel_tool_calls`, `tool_calls`, `role: "tool"`, streaming SSE deltas), `/v1/models`, `/v1/completions`, `/v1/embeddings`, `/infill`, `/props`, `/tokenize`, and `/apply-template`, and exposes live `nvidia-smi` + `timings` telemetry at `/metrics`.
   - `vscode-extension` (`harness-local.harness-vscode`) registers a native VS Code Language Model Chat Provider (`vscode.lm.registerLanguageModelChatProvider('harness', ...)`) so `Qwen3-Coder-30B-A3B` runs directly inside VS Code's built-in Chat / Agent mode with native multi-tool calling (`capabilities.toolCalling = true`).
2. **Standalone Rust / `wgpu` Compute & Algorithm Crates (`harness-core`, `harness-quant`, `harness-attention`, `harness-safety`):**
   - Provide standalone Vulkan/DX12/Metal `wgpu` compute shaders (`GEMV_Q4`, `GEMV_Q4_SPARSE`, `FLASH_GQA_Q8_Q4`, `RMSNORM`, `SWIGLU`) and CPU data structures exercised via `harness-cli gpu-bench` and `cargo test`.

---

## 2. Long-Context Agent Engine Benchmarks (`benchmarks/runs/agent_ctx/`)

Model: **`Qwen3-Coder-30B-A3B-Instruct` (`Q4_K_M`, 30.5B total / 3.3B active parameters, 48 layers, 128 experts, top-8 active)**.

| Config (`benchmarks/runs/agent_ctx/`) | Context (`-c`) | `--n-cpu-moe` | KV Cache (`-ctk`/`-ctv`) | Batch (`-b`/`-ub`) | VRAM Loaded (`nvidia-smi`) | Short Decode | Long Prefill Speed | Decode at Long Depth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `h_ctx16k_q8q4_ncmoe34_control` | `16,384` | `34` | `q8_0` / `q4_0` (`--no-op-offload`) | `512` / `256` | `7,020 MiB` | `15.88 tok/s` | `27.40 tok/s` (3,160 tok in 115.3s) | `5.67 tok/s` |
| **`i_ctx16k_q8q8_ncmoe34_gpuprefill`** | **`16,384`** | **`34`** | **`q8_0` / `q8_0` (GPU prefill)** | **`2048` / `1024`** | **`7,485 MiB`** | **`21.14 tok/s`** | **`411.27 tok/s` (3,160 tok in 7.68s)** | **`20.71 tok/s`** |
| **`r_ctx32k_q8q8_ncmoe36_gpuprefill`** | **`32,768`** | **`36`** | **`q8_0` / `q8_0` (GPU prefill)** | **`2048` / `1024`** | **`7,689 MiB`** | **`19.51 tok/s`** | **`452.53 tok/s` (8,909 tok in 19.69s)** | **`18.23 tok/s`** |
| **`s_ctx32k_q8q8_ncmoe37_gpuprefill` (Default `Agent`)** | **`32,768`** | **`37`** | **`q8_0` / `q8_0` (GPU prefill)** | **`2048` / `1024`** | **`7,365 MiB`** | **`21.40 tok/s`** | **`446.86 tok/s` (8,909 tok in 19.94s)** | **`19.88 tok/s`** |
| **`t_ctx64k_q8q8_ncmoe42_gpuprefill`** | **`65,536`** | **`42`** | **`q8_0` / `q8_0` (GPU prefill)** | **`2048` / `1024`** | **`7,341 MiB`** | **`18.86 tok/s`** | **`380.52 tok/s` (23,756 tok in 62.43s)** | **`14.04 tok/s`** |

---

## 3. Multi-Tool SWE & Coding Benchmarks

- **HARNESS-SWE-Lite 16-Task Multi-Tool Bug-Fix Suite (`benchmarks/runs/swe_lite/qwen3_30b_a3b_agent32k.json`)**:
  - **`16/16 (100.0%)` visible pytest suites passed, `14/16 (87.5%)` tamper-proof hidden edge-case suites resolved** across `170` turns, `160` tool calls, `323,796` prompt tokens, and `22,273` completion tokens at **`16.36 tok/s` mean multi-turn decode speed**.
- **EvalPlus HumanEval & HumanEval+ (`benchmarks/runs/humaneval/qwen3_30b_a3b_he20/generation_summary.json`)**:
  - **`12/12 (100.0%)` base HumanEval pass@1 and `12/12 (100.0%)` 80x stricter HumanEval+ pass@1** (`4,793` completion tokens, **`16.56 tok/s` mean decode**, `0` truncated).
- **VS Code Native Extension Host Integration (`vscode-extension/test/run-integration.js`)**:
  - **`11/11` unit tests + live VS Code `1.141.0` Extension Host integration test `PASS` (`Exit code: 0`, `maxInputTokens = 28,672`)**.
- **Skeptic's Trench 2k Decode Benchmarks (`benchmarks/runs/elite_coder_trench/`)**:
  - **`m5_qwen3_coder_30b_moe_ncmoe32_pure`**: `16.37 tok/s` pooled / `16.53 tok/s` peak (`3/3 PASS`)
  - **`d1_qwen25_coder_14b_0_5b_ngl38`**: `15.89 tok/s` pooled / `19.17 tok/s` peak (`3/3 PASS`, `97.9%` pooled draft acceptance)
  - **`c5_dense32b_0_5b_ngl20_prio`**: `3.90 tok/s` pooled / `4.80 tok/s` peak (`3/3 PASS`, `85.4%` pooled / `96.2%` Rust draft acceptance)
