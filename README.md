# 🚀 HARNESS: Pure-Rust 27B–35B Coder Inference Orchestration, GPU Parallelism & Safety Engine

[![Multi-Platform CI](https://github.com/ModernOps888/harness/actions/workflows/ci.yml/badge.svg)](https://github.com/ModernOps888/harness)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20(Metal%20UMA)%20%7C%20Windows%20(CUDA%2FVulkan)-blueviolet.svg)](#multi-os-hardware-auto-detection)
[![Enterprise-Advisory](https://img.shields.io/badge/Enterprise-Sovereign_AI_Advisory-00f0ff.svg)](https://infinitytechstack.uk/consulting)

> 🚀 **Enterprise Sovereign AI Deployment & Audits**: Scaling on-prem or air-gapped LLM inference on consumer hardware or sovereign cloud? We build **turnkey on-prem AI architecture blueprints** (£7,500) and perform **production agent security audits** (£4,500 with remediation PRs). [**Explore Packages & Book a Call →**](https://infinitytechstack.uk/consulting)

**HARNESS** is a high-performance, pure-Rust inference orchestration, GPU compute kernel, quantization, and safety middleware engineered to maximize **27B–35B Coder models** (`Qwen2.5-Coder-32B-Instruct`, `Qwen3-30B-A3B` MoE, `Qwen2.5-Coder-14B`, `DeepSeek-Coder-33B`, `Gemma-2-27B-IT`) on constrained consumer GPUs (e.g., 8 GB GDDR7 VRAM + 32 GB DDR4 RAM). By combining **GPU-Resident Speculative Drafting ($M = K+1$ Batched Verification)**, **Asymmetric INT8/INT4 KV Cache Compression**, **Native `wgpu` (Vulkan/DX12) WGSL Compute Shaders**, **Pushdown/DFA Constrained Code Decoding**, and **Shannon Entropy Hallucination Gating**, HARNESS delivers verified, lossless speedups and real-time hardware telemetry.

---

## 📸 Verified Platform Telemetry

![HARNESS Platform Screenshot](docs/screenshots/harness_platform.png)

*Figure: Real-time telemetry dashboard running on `http://localhost:3000` monitoring active local LLM inference with quantized KV cache blocks, speculative draft acceptance telemetry, and microsecond DFA/pushdown code syntax verification.*

---

## 🏆 27B–35B Coder Bare-Metal Benchmark Ledger (`Qwen2.5-Coder-32B-Instruct` `Q4_K_M`)

All measurements below were captured on bare metal (**NVIDIA GeForce RTX 5060 8,151 MiB GDDR7 @ 504 GB/s + Intel Core i5-10400F 6C/12T + 32 GB DDR4-2133 dual-channel, PCIe Gen 3 x8**) by [`benchmarks/bench_qwen32b.py`](benchmarks/bench_qwen32b.py). Every raw `llama-server` log (`server_stdout.txt`), command line, and verbatim JSON response is committed under [`benchmarks/runs/qwen32b_coder/`](benchmarks/runs/qwen32b_coder/QWEN32B_RESULTS.md).

| Configuration | GPU Layers (`32B` + `1.5B` Draft) | KV Cache Format (`2048` ctx) | PCIe Graph Splits (`bs=128`) | Total Gen Time (`576 tok`) | **Pooled tok/s** | Per-Prompt tok/s (`HumanEval` / `LRU` / `Deadlock`) | Draft Acceptance ($\alpha$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`c0_raw_baseline`** | `18/65` | FP16 (`512.00 MiB`) | `606` | `498.3s` | **1.16 tok/s** (`1.00x`) | `1.22` / `1.29` / `0.99` | n/a |
| **`c1_tuned_kv`** | `21/65` | `q8_0` K / `q4_0` V (`208.00 MiB`) | `682` | `365.2s` | **1.58 tok/s** (`1.36x`) | `1.56` / `1.59` / `1.56` | n/a |
| **`c3_spec_ngram`** | `21/65` | `q8_0` K / `q4_0` V (`208.00 MiB`) | `682` | `362.6s` | **1.59 tok/s** (`1.37x`) | `1.72` / `1.55` / `1.49` | `89/197` (`45.2%`) |
| **`c2_spec_draft_1_5b`** | `17/65` + `29/29` | `q8_0` K / `q4_0` V (`208 MiB` + `56 MiB` FP16 draft) | `674` | `192.5s` | **2.99 tok/s** (`2.58x`) | `3.65` / `3.24` / `2.35` | `419/610` (`68.7%`) |
| **`c4_ultra_parallel_spec`** | **`18/65` + `29/29`** | **`q8_0` K / `q4_0` V (`208 MiB` + `22.75 MiB` draft)** | **`36` (-94.7%)** | **`171.1s`** | **3.37 tok/s (`2.91x`)** | **`4.44` / `3.65` / `2.52`** *(3s peak: **`5.17 tok/s`**)* | **`391/434` (`90.1%`)** *(`95.7%` on HumanEval)* |

### Key Hardware Engineering Breakthroughs in `c4_ultra_parallel_spec`
1. **94.7% Reduction in PCIe Graph Splits (`--no-op-offload`)**:
   Without `--no-op-offload`, `ggml-cuda` offloaded temporary host matrix operations across PCIe Gen 3 x8 during batched $M = K+1$ speculative verification, causing **674 graph splits** (`sched_reserve: graph splits = 674`) and consuming `186.58 MiB` of CUDA compute buffer. Enabling `--no-op-offload` kept CPU-layer ops on the CPU threadpool and GPU-layer ops on CUDA0, cutting `graph splits` to **`36`** and shrinking `CUDA0 compute buffer size` to **`45.58 MiB`** ([`c4_ultra_parallel_spec/server_stdout.txt`](benchmarks/runs/qwen32b_coder/c4_ultra_parallel_spec/server_stdout.txt)).
2. **Asymmetric KV Cache Quantization (`-ctk q8_0 -ctv q4_0 -ctkd q8_0 -ctvd q4_0`)**:
   Qwen2.5-Coder-32B uses 5:1 Grouped-Query Attention (40 query heads, 8 KV heads, `head_dim = 128`). Quantizing Keys to `Q8_0` (preserving RoPE phase precision) and Values to `Q4_0` compressed the 32B KV cache from `512.00 MiB` to **`208.00 MiB`** (`136.00 MiB` K + `72.00 MiB` V) and the 1.5B drafter KV cache from `56.00 MiB` to **`22.75 MiB`**, freeing enough GDDR7 VRAM to offload an extra 32B layer (`18/65`, `5,453.82 MiB`) alongside 100% (`29/29` layers, `934.70 MiB`) of `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M`.
3. **Batched $M = K+1$ Speculative Verification & Asymmetric Threading (`-t 6 -tb 10`)**:
   During speculative verification of $K$ draft tokens, the 32B target model reads its `13.47 GiB` of DDR4-resident weights **strictly once** in a single $M = K+1$ micro-GEMM batch using 10 SMT threads (`-tb 10`), while single-token steps use 6 physical cores (`-t 6`) to avoid cache thrashing. With `--spec-draft-n-max 6 --spec-draft-p-min 0.68`, draft acceptance reached **95.7% (`157/164`, mean burst length `6.23` tokens, 100% unit tests passed)** on `humaneval_merge_intervals`.

---

## 📊 Audited Systems & Middleware Telemetry Scorecard

| Evaluation Task / Primitive | Verified Measurement | Hardware Grounding / Verification Mechanism |
| :--- | :--- | :--- |
| **32B Coder Ultra-Parallel Speculative (`c4`)** | **3.37 tok/s pooled** (`4.44 tok/s` HumanEval, `5.17 tok/s` 3s peak, `90.1%` draft acc) | Real 32.76B weights (`qwen2.5-coder:32b`) + 1.5B Q4 drafter in RTX 5060 GDDR7; see [`QWEN32B_RESULTS.md`](benchmarks/runs/qwen32b_coder/QWEN32B_RESULTS.md) |
| **Native GPU In-Register INT4 GEMV (`gemv_q4.wgsl`)** | **80.03 μs** (50-run avg, **1,220.2 GB/s** eff BW) / **58.46 μs** (10-run warm, **1,670.4 GB/s** eff BW) | Measured on RTX 5060 via `harness-cli gpu-bench` ($M=1, K=5120, N=5120$); see [`gpu_bench_32b_coder.txt`](benchmarks/runs/gpu_bench_32b_coder.txt) |
| **Native GPU 32B FP32 Token GEMV (`gemv.wgsl`)** | **71.32 μs** (**735.16 GFLOP/s**, $M=1, K=5120, N=5120$) | Measured on RTX 5060 via `harness-cli gpu-bench`; see [`gpu_bench_32b_coder.txt`](benchmarks/runs/gpu_bench_32b_coder.txt) |
| **Native GPU 2D Tiled GEMM (`gemm.wgsl`)** | **0.44 ms** (**4.84 TFLOP/s**, $1024 \times 1024 \times 1024$ FP32) | Bank-conflict-free `16x17` padded shared memory via `wgpu` Vulkan; see [`gpu_bench_32b_coder.txt`](benchmarks/runs/gpu_bench_32b_coder.txt) |
| **Native GPU RMSNorm & SwiGLU** | **257.09 μs** (**3.98M tok/s**) & **640.50 μs** (**5,525 M-elem/s**) | Measured on 32B Coder hidden/FFN geometry ($1024 \times 5120$ & $128 \times 27648$); see [`gpu_bench_32b_coder.txt`](benchmarks/runs/gpu_bench_32b_coder.txt) |
| **DFA & Pushdown Constrained Decoding** | tens of ns per mask (`harness-cli stress`) | Precompiled bitmasks + pushdown delimiter state machine (`crates/harness-safety/src/constrained.rs`); see [`microbench_stress.txt`](benchmarks/runs/microbench_stress.txt) |
| **PagedAttention & Quantized KV Cache** | ~10^7 blocks/s allocator + real `Int8`/`Int4` storage | Allocator churn test + symmetric group-quantized KV storage (`crates/harness-attention/src/kv_cache.rs`) |
| **Pure-Rust AVX2 CPU GEMM Kernel** | ~21 GFLOP/s ($1024 \times 1024$ FP32, `stress` step 1) | Rayon + AVX2 FMA microbenchmark with precision audit; see [`microbench_stress.txt`](benchmarks/runs/microbench_stress.txt) |
| **70B Raw Hardware Execution (8GB VRAM)** | **0.65 tok/s** (`c3_spec_1b` on `llama3.1:70b-instruct-q2_K`) | Historical 70.55B baseline ledger; raw logs committed in [`RESULTS.md`](benchmarks/runs/RESULTS.md) |

---

## ⚡ Tri-Core Principal Architecture (Pure-Rust, Zero-Wrapper Execution)

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  HARNESS TRI-CORE ARCHITECTURE                                         │
├────────────────────────────────────┬───────────────────────────────────┬───────────────────────────────┤
│ CORE 1: NATIVE GPU LATENCY & TOK/S │ CORE 2: CODING QUALITY & DFA AST  │ CORE 3: TEST-TIME COMPUTE     │
├────────────────────────────────────┼───────────────────────────────────┼───────────────────────────────┤
│ • Direct wgpu (Vulkan / DX12) WGSL │ • Microsecond DFA Grammar Masking │ • Best-of-N Trajectory Search │
│ • Coalesced INT4 GEMV (1220+ GB/s) │ • Pushdown bracket/quote automaton│ • Shannon Entropy H(X) score  │
│ • Batched M=K+1 Q4 Spec Verifier   │ • Rust lifetime ('a) aware repair │ • Cortical Lateral Inhibition │
│ • 5:1 GQA FlashAttention + RoPE    │ • JSON / Rust / Python validation │ • Stigmergic dead-end pruning │
└────────────────────────────────────┴───────────────────────────────────┴───────────────────────────────┘
```

### Core 1: Native GPU Compute Subsystem (`crates/harness-core/src/gpu/`)
Executed directly on the **NVIDIA GeForce RTX 5060** via `wgpu` (Vulkan/DirectX 12) with zero Python or external daemon dependencies ([`benchmarks/runs/gpu_bench_32b_coder.txt`](benchmarks/runs/gpu_bench_32b_coder.txt)):
- **2D Tiled Shared-Memory GEMM ([`gemm.wgsl`](crates/harness-core/src/gpu/kernels/gemm.wgsl))**: Uses `array<array<f32, 17>, 16>` bank-conflict-free shared memory padding to achieve **4.84 TFLOP/s** ($1024^3$ FP32 in **0.44 ms**).
- **32B Coder Single-Token GEMV ([`gemv.wgsl`](crates/harness-core/src/gpu/kernels/gemv.wgsl))**: Cooperative `shared_x` tile caching across 8 warps per block executes a full $5120 \times 5120$ projection in **71.32 μs** (**735.16 GFLOP/s**).
- **In-Register Quantized GEMV Q4 ([`gemv_q4.wgsl`](crates/harness-core/src/gpu/kernels/gemv_q4.wgsl))**: 1-warp-per-row coalesced `u32` nibble loads and branchless SIMT shifts dequantize 4-bit weights in registers, executing $5120 \times 5120$ Q4 projections in **80.03 μs** (50-run avg, **1,220.2 GB/s** effective bandwidth) and **58.46 μs** (10-run warm, **1,670.4 GB/s** effective bandwidth / **313.20 GB/s** physical VRAM read).
- **Batched Speculative Q4 GEMM ([`gemm_q4_batched.wgsl`](crates/harness-core/src/gpu/kernels/gemm_q4_batched.wgsl))**: Verifies $M = K+1$ speculative draft tokens (`M <= 8`) simultaneously while reading packed INT4 weights from VRAM only once.
- **5:1 GQA FlashAttention ([`flash_attn_gqa.wgsl`](crates/harness-core/src/gpu/kernels/flash_attn_gqa.wgsl))**: Native online-softmax Grouped-Query Attention supporting Qwen2.5-Coder-32B's `40:8` head ratio (`head_dim = 128`) and speculative batch causal masking.
- **NeoX Half-Split RoPE ([`rope.wgsl`](crates/harness-core/src/gpu/kernels/rope.wgsl)) & Fused Add+RMSNorm ([`fused_add_rmsnorm.wgsl`](crates/harness-core/src/gpu/kernels/fused_add_rmsnorm.wgsl))**: Exact half-dimension rotary pairings `(i, i + head_dim/2)` with $\theta = 1,000,000$ for Qwen2.5/Qwen3/Llama-3, plus single-dispatch residual addition and RMS normalization (**3.98M tokens/sec**).

Run the bare-metal GPU compute benchmark:
```bash
cargo run --release -p harness-cli -- gpu-bench --matrix-dim 1024 --iterations 50
```

### Core 2: In-Loop Code Quality & Syntax Verification Engine (`harness-cli code-check`)
- Lexical AST and delimiter verification across **JSON, Rust, and Python** ([`crates/harness-pipeline/src/code_engine.rs`](crates/harness-pipeline/src/code_engine.rs)).
- Correctly distinguishes Rust lifetime annotations (`&'a T`, `'static`) and line/block comments (`//`, `/* */`, `#`) from string literals before performing deterministic delimiter auto-repair.
- Coupled with [`ConstrainedDecoder`](crates/harness-safety/src/constrained.rs), which tracks live brace/bracket/string depth via a pushdown state machine to mask invalid closing tokens or prematurely terminated JSON tool calls.

Test code verification and repair:
```bash
cargo run --release -p harness-cli -- code-check --code "{\"model\": \"qwen2.5-coder:32b\", \"ctx\": 2048" --lang json
```

### Core 3: Test-Time Compute (TTC) Reasoning Engine (`harness-cli reason`)
- Evaluates Best-of-$N$ candidate code and reasoning trajectories ([`crates/harness-pipeline/src/ttc_engine.rs`](crates/harness-pipeline/src/ttc_engine.rs)).
- Scores candidates using Shannon entropy $H(X) = -\sum p(x) \ln p(x)$ and cortical lateral inhibition contrast, pruning syntax-invalid or high-entropy branches.

---

## 🧮 Mathematical Quantization, Attention & Model Architecture Primitives

1. **Bit-Exact OCP FP8 E4M3 (`E4M3FN`) Quantization ([`crates/harness-quant/src/fp8.rs`](crates/harness-quant/src/fp8.rs))**:
   Implements the Open Compute Project 8-bit floating-point format (1 sign bit, 4 exponent bits with bias $b=7$, 3 mantissa bits, max finite value $448.0$, subnormal scaling $2^{-9}$) with 32-element group absmax scaling.
2. **Group-Scaled NormalFloat4 (`NF4`) with Branchless Binary Search ([`crates/harness-quant/src/nf4.rs`](crates/harness-quant/src/nf4.rs))**:
   Uses 64-element group scales and a 4-step branchless binary search over the 15 exact Voronoi midpoints of the standard normal distribution $\mathcal{N}(0,1)$ quantiles.
3. **Activation-Aware `imatrix` INT4 & Fused MatVec ([`crates/harness-quant/src/q4.rs`](crates/harness-quant/src/q4.rs))**:
   Supports importance-matrix weighted scale calibration (`from_f32_tensor_imatrix`), outlier crest-factor profiling ([`isq.rs`](crates/harness-quant/src/isq.rs)), and in-register fused matrix-vector multiplication (`matvec_fused`) without intermediate full-matrix FP32 allocation.
4. **Quantized KV Cache & Decode-Aware FlashAttention ([`crates/harness-attention/`](crates/harness-attention/src/kv_cache.rs))**:
   - [`KVCache`](crates/harness-attention/src/kv_cache.rs) stores compressed `Int8` (`Q8_0`) or `Int4` (`Q4_0`) pages with 32-element group scales and reconstructs full context via `get_active_kv()`.
   - [`FlashAttention`](crates/harness-attention/src/flash_attn.rs) enforces offset-aware causal masking `max_k_pos = (seq_len_kv - seq_len_q) + qi + 1` so both prefill ($S_q = S_{kv}$) and autoregressive/speculative decode ($S_q \ll S_{kv}$) attend over full cached history.
   - [`RadixPrefixCache`](crates/harness-attention/src/prefix_cache.rs) preserves proportional `block_ids` across prefix-tree edge splits for zero-loss KV block reuse.
5. **GGUF v2/v3 Binary Header & Tensor Inspector ([`crates/harness-loader/src/gguf.rs`](crates/harness-loader/src/gguf.rs))**:
   Parses GGUF v2/v3 metadata key-value tables (`general.architecture`, `*.block_count`, `*.attention.head_count_kv`, `*.rope.freq_base`) and tensor descriptors (`GGMLType::Q4_K`, `Q6_K`, `Q8_0`, etc.) directly from disk or memory-mapped files.
6. **Dense & Sparse MoE 27B–35B Coder Architectures ([`crates/harness-models/`](crates/harness-models/src/qwen.rs))**:
   Includes first-class model definitions and hardware offload planners (`HardwareProfile::plan_coder_offload`) for:
   - `Qwen2.5-Coder-32B-Instruct` (64 layers, 40Q/8KV GQA, QKV bias enabled, `intermediate_size = 27648`, `vocab = 152064`)
   - `Qwen2.5-Coder-1.5B-Instruct` (28 layers, 12Q/2KV GQA, 100% VRAM-resident speculative drafter)
   - `Qwen3-30B-A3B` Sparse MoE (48 layers, 128 total experts, top-8 routing via [`MoELayer::forward`](crates/harness-models/src/moe.rs), QK-Norm enabled, 3.3B active parameters per token)
   - `DeepSeek-Coder-33B-Instruct` & `Gemma-2-27B-IT`

---

## 🔒 Enterprise Security Hardening

1. **Unsafe Slice Pointer Alignment Checks**:
   In [`crates/harness-core/src/tensor.rs`](crates/harness-core/src/tensor.rs), raw byte-to-float pointer casts enforce 4-byte pointer alignment (`(ptr as usize) % std::mem::align_of::<f32>() == 0`) and overflow-checked buffer boundary validation (`checked_mul`).
2. **Axum HTTP DoS Mitigation**:
   Enforces `DefaultBodyLimit::max(16 * 1024 * 1024)` across API endpoints to block unbounded memory buffering attacks.
3. **Directory Traversal Protection**:
   In [`crates/harness-loader/src/mmap_loader.rs`](crates/harness-loader/src/mmap_loader.rs), model paths are verified for regular file status and resolved with `canonicalize()`, preventing relative path traversal (`../`).

---

## ⚡ Quickstart

### Prerequisites
- **Rust Toolchain**: `rustc 1.85+` (`cargo`)
- **Node.js**: `v20+` (for the frontend GUI)

### Build, Verify, and Launch
```bash
# 1. Clone repository
git clone https://github.com/ModernOps888/harness.git
cd harness

# 2. Run the full verification test suite (52 unit and integration tests across 11 crates)
cargo test --workspace

# 3. Inspect the 27B-35B Coder hardware offload plan, VRAM budget, and speculative speedup
cargo run --release -p harness-cli -- coder32b

# 4. Run bare-metal wgpu/Vulkan GPU kernel benchmarks (Qwen2.5-Coder-32B geometry)
cargo run --release -p harness-cli -- gpu-bench --matrix-dim 1024 --iterations 50

# 5. Launch the bare-metal Qwen2.5-Coder-32B + 1.5B Drafter server on port 8089 (Windows PowerShell)
powershell -ExecutionPolicy Bypass -File .\launch_qwen32b_coder.ps1

# 6. Launch the HARNESS orchestration API server (auto-detects port 8089 first)
cargo run --release -p harness-cli -- serve --port 8080

# 7. Launch the React/TypeScript telemetry dashboard
cd frontend
npm install
npm run dev
```

Visit **`http://localhost:3000`** in your browser to monitor live 32B Coder inference, KV cache compression, and speculative draft acceptance.

---

## 📜 Crates Hierarchy

```text
crates/
├── harness-core        # Tensor abstractions, wgpu/Vulkan WGSL kernels (GEMM, GEMV, Q4, Batched Q4, GQA FlashAttn, RoPE, RMSNorm, SwiGLU), coder configs
├── harness-loader      # GGUF v2/v3 metadata & tensor parser, zero-copy mmap SafeTensors/GGUF loader with path canonicalization
├── harness-quant       # Bit-exact OCP FP8 E4M3, 64-group NF4 with binary search, imatrix Q4_0 & fused in-register matvec
├── harness-attention   # Decode-aware FlashAttention-3, Quantized Int8/Int4 KV Cache, LIF Spiking Attention, MLA, RadixPrefixCache
├── harness-models      # Qwen2.5-Coder-32B/1.5B (QKV bias), Qwen3-30B-A3B MoE (QK-Norm + top-k MoELayer), DeepSeek, LLaMA
├── harness-pipeline    # Speculative decoding (batch cost gamma + vocab alignment), Rust-lifetime-aware CodeQualityEngine, TTC engine
├── harness-safety      # Pushdown/DFA constrained decoding, Cortical lateral inhibition, Shannon entropy gating
├── harness-rag         # Hippocampal fast-slow dual memory, Cortical engram vector store
├── harness-server      # Axum HTTP/SSE OpenAI-compatible API server prioritizing port 8089 bare-metal 32B Coder runner
├── harness-cli         # Unified CLI (`coder32b`, `gpu-bench`, `code-check`, `reason`, `stress`, `tune`, `serve`)
└── harness-mcp         # Model Context Protocol (MCP) server for IDE pair-programming
```

---

## 📄 License

Licensed under the **MIT License** ([LICENSE](LICENSE)).
