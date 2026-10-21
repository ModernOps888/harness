# 🚀 HARNESS: Local 30B–32B Coder Agent Engine, Native VS Code Multi-Tool Chat Provider & Rust GPU/Safety Suite

[![Multi-Platform CI](https://github.com/ModernOps888/harness/actions/workflows/ci.yml/badge.svg)](https://github.com/ModernOps888/harness)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![VS Code Native Chat](https://img.shields.io/badge/VS%20Code-Native%20Chat%20%2B%20Multi--Tool%20Agent-007ACC.svg)](vscode-extension/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20(Metal%20UMA)%20%7C%20Windows%20(CUDA%2FVulkan)-blueviolet.svg)](#quickstart)

**HARNESS** is an open-source local coding agent engine, lossless Rust/Axum OpenAI proxy (`crates/harness-server`), **native VS Code Language Model Chat Provider extension** ([`vscode-extension/`](vscode-extension/)), and standalone Rust/`wgpu` GPU kernel & safety suite engineered to run **30B–32B-class Coder models** (`Qwen3-Coder-30B-A3B-Instruct` Sparse MoE, `Qwen2.5-Coder-32B-Instruct` Dense, `Qwen2.5-Coder-14B-Instruct`) at full **16k–64k agentic context windows** on an 8 GB VRAM GPU (**NVIDIA GeForce RTX 5060 8,151 MiB GDDR7 @ 448 GB/s + Intel Core i5-10400F + 32 GB DDR4-2133**).

### Verified Headline Capabilities & Measurements

- **Native VS Code Chat & Multi-Tool Agent Mode ([`vscode-extension/`](vscode-extension/))**: Registers vendor `"harness"` via `vscode.lm.registerLanguageModelChatProvider('harness', provider)` with `capabilities: { toolCalling: true }`. Works directly inside VS Code's built-in Chat / Agent mode with all 36+ native VS Code workspace/terminal/search tools — no Continue or third-party chat sidebar required. Verified end-to-end inside live **VS Code `1.141.0` Extension Host** (`vscode-extension/test/run-integration.js`, `11/11` unit tests + live multi-tool `list_dir -> read_file` integration test, `Exit code: 0`).
- **15× Faster Long-Context GPU Prefill & `21.40 tok/s` Decode at 32k Context ([`benchmarks/runs/agent_ctx/`](benchmarks/runs/agent_ctx/))**:
  - **16k Context (`i_ctx16k_q8q8_ncmoe34_gpuprefill`, `--n-cpu-moe 34`)**: **`411.27 tok/s` GPU prefill** (`3,160` tokens in `7.68s`), **`20.71 tok/s` decode at 3.1k depth**, **`21.14 tok/s` short decode** (`7,485 MiB` VRAM).
  - **32k Context (`s_ctx32k_q8q8_ncmoe37_gpuprefill`, Default `-Mode Agent`, `--n-cpu-moe 37`)**: **`446.86 tok/s` GPU prefill** (`8,909` tokens in `19.94s`), **`19.88 tok/s` decode at 8.9k depth**, **`21.40 tok/s` short decode** (`7,365 MiB` VRAM).
  - **64k Context (`t_ctx64k_q8q8_ncmoe42_gpuprefill`, `--n-cpu-moe 42`)**: **`380.52 tok/s` GPU prefill** (`23,756` tokens in `62.43s`), **`14.04 tok/s` decode at 23.8k depth**, **`18.86 tok/s` short decode** (`7,341 MiB` VRAM).
- **Rigorous Software Engineering & Coding Benchmark Scores**:
  - **HARNESS-SWE-Lite 16-Task Multi-Tool Bug-Fix Suite ([`benchmarks/runs/swe_lite/qwen3_30b_a3b_agent32k.json`](benchmarks/runs/swe_lite/qwen3_30b_a3b_agent32k.json))**: **`16/16 (100.0%)` visible pytest suites passed, `14/16 (87.5%)` tamper-proof hidden edge-case suites resolved** across `170` agent turns and `160` tool calls (`323,796` prompt tokens, `22,273` completion tokens, **`16.36 tok/s` mean multi-turn decode**).
  - **EvalPlus HumanEval & HumanEval+ ([`benchmarks/runs/humaneval/qwen3_30b_a3b_he20/`](benchmarks/runs/humaneval/qwen3_30b_a3b_he20/generation_summary.json))**: **`12/12 (100.0%)` base HumanEval pass@1 and `12/12 (100.0%)` 80× stricter HumanEval+ pass@1** (`4,793` completion tokens, **`16.56 tok/s` mean decode**, `0` truncated).
  - **Skeptic's Trench 8-Config Compiler Gate ([`benchmarks/runs/elite_coder_trench/ELITE_CODER_RESULTS.md`](benchmarks/runs/elite_coder_trench/ELITE_CODER_RESULTS.md))**: **`24/24 (100%)` PASS** (`rustc --test -O` lock-free SPSC ring buffer, 10/10 Python regex DP assertions, and Tokio Mutex deadlock remediation).

---

## 🏗️ Architectural Scope: What Runs Where

To maintain 100% engineering transparency, HARNESS separates the live GGUF inference & IDE proxy path from its standalone Rust/`wgpu` kernel and algorithm crates:

1. **Live GGUF Inference & Native VS Code Agent Pipeline (`launch_qwen32b_coder.ps1` → `harness-server` `:8090` → `vscode-extension`)**:
   - **GGUF Execution Engine (`llama-server` on `127.0.0.1:8089`)**: Executes the quantized `.gguf` model weights (`Qwen3-Coder-30B-A3B` `Q4_K_M`, `Qwen2.5-Coder-32B` `Q4_K_M`, `Qwen2.5-Coder-14B` `Q4_K_S`) using CUDA FlashAttention (`-fa on -ctk q8_0 -ctv q8_0`), GPU batch prefill (`-b 2048 -ub 1024`), hybrid CPU/GPU MoE expert placement (`--n-cpu-moe N`), and Jinja tool-call rendering (`--jinja`).
   - **HARNESS Lossless OpenAI Proxy (`crates/harness-server` on `127.0.0.1:8090`)**: Forwards `/v1/chat/completions` JSON bodies **verbatim** (`tools`, `tool_choice`, `parallel_tool_calls`, `tool_calls`, `role: "tool"`, `content: null`, content-part arrays, `stream_options`), emits early SSE headers so multi-thousand-token prefills never trigger HTTP client header timeouts, propagates client disconnects via `tokio::select!` on `sse_tx.closed()` to abort upstream generation immediately, and exposes `/v1/models`, `/v1/completions`, `/v1/embeddings`, `/infill`, `/props`, `/tokenize`, `/apply-template`, and live `nvidia-smi` + `timings` telemetry at `/metrics`.
   - **Native VS Code Language Model Chat Provider (`vscode-extension/`)**: Connects VS Code's native Chat / Agent UI (`vscode.lm.registerLanguageModelChatProvider('harness', ...)`) to `http://127.0.0.1:8090/v1`, queries `/props` to advertise the true context window (`maxInputTokens = 28,672` at `n_ctx = 32,768`), normalizes tool JSON schemas for Qwen's Jinja template, and coerces stringified tool arguments back to declared schema types.
2. **Standalone Rust GPU Compute, Quantization & Safety Crates (`harness-core`, `harness-quant`, `harness-attention`, `harness-loader`, `harness-safety`, `harness-pipeline`)**:
   - Self-contained Rust libraries and `wgpu` (Vulkan/DX12/Metal) compute shaders (`sparse_gemv_q4.wgsl`, `moe_router_topk.wgsl`, `gemv_q4.wgsl`, `gemm_q4_batched.wgsl`, `flash_attn_gqa.wgsl`, `rope.wgsl`, `rmsnorm.wgsl`, `swiglu.wgsl`), bit-exact GGUF `Q4_K` / `FP8` / `NF4` quantizers, quantized KV caches, and DFA/pushdown syntax verifiers exercised via `harness-cli gpu-bench`, `harness-cli code-check`, `harness-cli stress`, and `cargo test --workspace`.

---

## ⚡ Long-Context (`16k–64k`) Agent Engine Benchmarks (`benchmarks/runs/agent_ctx/`)

Executed by [`benchmarks/bench_agent_ctx.py`](benchmarks/bench_agent_ctx.py) on **`Qwen3-Coder-30B-A3B-Instruct` (`Q4_K_M`, 30.5B total / 3.3B active parameters, 48 layers, 128 routed experts, top-8 active)** on the **RTX 5060 8,151 MiB GDDR7 (`448 GB/s`) + 32 GB DDR4-2133**. All raw JSON timings and `llama-server` logs are committed in [`benchmarks/runs/agent_ctx/`](benchmarks/runs/agent_ctx/).

| Config (`benchmarks/runs/agent_ctx/`) | Context (`-c`) | `--n-cpu-moe` | KV Cache (`-ctk` / `-ctv`) | Batch (`-b` / `-ub`) | VRAM Loaded (`nvidia-smi`) | Short Decode | Long-Context Prefill Speed | Decode at Long Depth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `h_ctx16k_q8q4_ncmoe34_control` | `16,384` | `34` | `q8_0` / `q4_0` (`--no-op-offload`) | `512` / `256` | `7,020 MiB` (`7001 MHz`) | `15.88 tok/s` | `27.40 tok/s` (`3,160` tok in `115.32s`) | `5.67 tok/s` |
| **`i_ctx16k_q8q8_ncmoe34_gpuprefill`** | **`16,384`** | **`34`** | **`q8_0` / `q8_0` (GPU prefill)** | **`2048` / `1024`** | **`7,485 MiB` (`13801 MHz`)** | **`21.14 tok/s`** | **`411.27 tok/s` (`3,160` tok in `7.68s` — `15.0x`)** | **`20.71 tok/s` (`3.65x`)** |
| **`r_ctx32k_q8q8_ncmoe36_gpuprefill`** | **`32,768`** | **`36`** | **`q8_0` / `q8_0` (GPU prefill)** | **`2048` / `1024`** | **`7,689 MiB` (`13801 MHz`)** | **`19.51 tok/s`** | **`452.53 tok/s` (`8,909` tok in `19.69s`)** | **`18.23 tok/s`** |
| **`s_ctx32k_q8q8_ncmoe37_gpuprefill` (Default `-Mode Agent`)** | **`32,768`** | **`37`** | **`q8_0` / `q8_0` (GPU prefill)** | **`2048` / `1024`** | **`7,365 MiB` (`13801 MHz`)** | **`21.40 tok/s`** | **`446.86 tok/s` (`8,909` tok in `19.94s`)** | **`19.88 tok/s`** |
| **`t_ctx64k_q8q8_ncmoe42_gpuprefill`** | **`65,536`** | **`42`** | **`q8_0` / `q8_0` (GPU prefill)** | **`2048` / `1024`** | **`7,341 MiB` (`13801 MHz`)** | **`18.86 tok/s`** | **`380.52 tok/s` (`23,756` tok in `62.43s`)** | **`14.04 tok/s`** |

### Why `-ctk q8_0 -ctv q8_0 -b 2048 -ub 1024` (Without `--no-op-offload`) Unlocked 15× Faster Prefill & 3.65× Faster Deep Decode
1. **Symmetric `q8_0/q8_0` CUDA FlashAttention vs. Mixed `q8_0/q4_0` Fallback**: With asymmetric `-ctk q8_0 -ctv q4_0` and `--no-op-offload`, `llama-server` lacked a fast CUDA FlashAttention path, keeping the RTX 5060 in its `7,001 MHz` low-power memory clock state and collapsing 3,160-token prefill to `27.40 tok/s` (`115.32s`) and decode at 3.1k depth to `5.67 tok/s`.
2. **Full `13,801 MHz` GDDR7 GPU Batch Prefill**: Switching to symmetric `-ctk q8_0 -ctv q8_0 -fa on -b 2048 -ub 1024` without `--no-op-offload` executes attention and batch prefill on the RTX 5060 at its full `13,801 MHz` (`448 GB/s`) GDDR7 clock, processing **8,909 prompt tokens in `19.94s` (`446.86 tok/s`)** at `32,768` context while sustaining **`19.88 tok/s` decode at 8.9k depth** (`21.40 tok/s` short decode).

---

## 🧪 Multi-Tool SWE & Coding Benchmark Ledger

### 1. HARNESS-SWE-Lite 16-Task Multi-Tool Agent Suite (`14/16 = 87.5%` Resolved, `16/16 = 100.0%` Visible Pass)

Executed by [`benchmarks/bench_swe_lite.py`](benchmarks/bench_swe_lite.py) against `http://127.0.0.1:8090/v1` (`Qwen3-Coder-30B-A3B` 32k Agent Mode). For each task, the model receives only the bug report and visible tests, and autonomously calls `list_dir`, `read_file`, `write_file`, `replace_in_file`, and `run_tests` over up to 14 turns. Scoring is tamper-proof: after the agent finishes, original visible test files are restored, any `conftest.py`/`pytest.ini` overrides are stripped, and a hidden edge-case test suite is executed ([`benchmarks/runs/swe_lite/qwen3_30b_a3b_agent32k.json`](benchmarks/runs/swe_lite/qwen3_30b_a3b_agent32k.json)):

| Task ID | Bug Domain | Visible Tests | Hidden Tests | **Resolved** | Agent Turns | Tool Calls | Prompt Tokens | Completion Tokens | Mean Decode (`tok/s`) | Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `t01_lru_get_refresh` | `OrderedDict` LRU cache recency refresh on `get()` | PASS | PASS | **PASS** | `8` | `7` | `10,907` | `617` | `16.67` | `56.0s` |
| `t02_merge_intervals` | Unsorted & touching interval merge without input mutation | PASS | PASS | **PASS** | `10` | `9` | `16,520` | `1,045` | `16.39` | `84.2s` |
| `t03_mutable_default` | Mutable default arg & non-mutating list return on duplicates | PASS | FAIL | FAIL | `8` | `7` | `10,240` | `840` | `17.35` | `61.6s` |
| `t04_split_bill_cents` | Exact cent remainder distribution (`total_cents % n`) | PASS | PASS | **PASS** | `14` | `14` | `33,294` | `1,845` | `18.14` | `136.5s` |
| `t05_lower_bound` | Binary search first index $\ge$ target with duplicates | PASS | PASS | **PASS** | `8` | `7` | `10,671` | `674` | `17.19` | `55.2s` |
| `t06_csv_escaped_quotes` | RFC-style CSV doubled-quote (`""`) state machine | PASS | FAIL | FAIL | `14` | `14` | `36,394` | `2,241` | `16.73` | `168.1s` |
| `t07_toposort_cycles` | Deterministic Kahn topological sort + cycle detection | PASS | PASS | **PASS** | `10` | `9` | `16,424` | `1,418` | `16.46` | `102.7s` |
| `t08_token_bucket_cap` | Token bucket refill cap & monotonic timestamp handling | PASS | PASS | **PASS** | `8` | `7` | `11,839` | `794` | `16.40` | `69.7s` |
| `t09_deep_merge` | Recursive non-mutating nested dict merge | PASS | PASS | **PASS** | `9` | `8` | `14,495` | `1,214` | `15.92` | `98.0s` |
| `t10_slugify` | Unicode NFKD ASCII normalization & hyphen collapsing | PASS | PASS | **PASS** | `9` | `8` | `11,482` | `821` | `16.52` | `64.9s` |
| `t11_async_gather_limit` | `asyncio.Semaphore` bounded concurrency + order preservation | PASS | PASS | **PASS** | `9` | `8` | `12,232` | `870` | `15.78` | `70.7s` |
| `t12_thread_safe_counter` | `threading.Lock` atomic increment under contention | PASS | PASS | **PASS** | `7` | `6` | `7,821` | `499` | `15.68` | `44.9s` |
| `t13_semver_compare` | SemVer 2.0 numeric pre-release & build metadata ordering | PASS | PASS | **PASS** | `14` | `14` | `40,093` | `3,886` | `15.25` | `296.8s` |
| `t14_normalize_path` | POSIX `.` / `..` stack normalization (root vs relative) | PASS | PASS | **PASS** | `14` | `14` | `29,963` | `1,841` | `15.93` | `152.0s` |
| `t15_cross_module_discount` | Cross-file `pricing.py` / `cart.py` percentage vs fraction fix | PASS | PASS | **PASS** | `14` | `14` | `28,300` | `1,421` | `16.04` | `129.0s` |
| `t16_add_business_days` | Positive/negative business-day calendar arithmetic across holidays | PASS | PASS | **PASS** | `14` | `14` | `33,121` | `2,247` | `15.64` | `181.9s` |
| **Total / Aggregate** | **16 Multi-File / Multi-Tool Python Bug-Fix Tasks** | **`16/16` (`100%`)** | **`14/16` (`87.5%`)** | **`14/16` (`87.5%`)** | **`170`** | **`160`** | **`323,796`** | **`22,273`** | **`16.36 tok/s`** | **`1,772.2s`** |

### 2. EvalPlus HumanEval & HumanEval+ (`12/12 = 100.0%` Base & 80× Plus Tests)

Executed by [`benchmarks/bench_humaneval.py`](benchmarks/bench_humaneval.py) (`temperature = 0, top_k = 1, seed = 1`) through `http://127.0.0.1:8090/v1` and verified against the official EvalPlus `HumanEvalPlus-v0.1.10` ground-truth oracles ([`benchmarks/runs/humaneval/qwen3_30b_a3b_he20/generation_summary.json`](benchmarks/runs/humaneval/qwen3_30b_a3b_he20/generation_summary.json)):
- **Problems Evaluated (`HumanEval/0` – `HumanEval/11`)**: `12`
- **Base HumanEval Pass@1**: **`12 / 12` (`100.0%`)**
- **EvalPlus HumanEval+ (80× extra edge-case tests) Pass@1**: **`12 / 12` (`100.0%`)**
- **Completion Tokens / Mean Decode Speed**: `4,793` tokens at **`16.56 tok/s`** (`0` truncated).

### 3. Live VS Code `1.141.0` Native Chat & Multi-Tool Integration Test (`Exit code: 0`)

Executed by [`vscode-extension/test/run-integration.js`](vscode-extension/test/run-integration.js) inside the real VS Code `1.141.0` Extension Host (`@vscode/test-electron`):
- **Unit Test Suite ([`vscode-extension/test/unit.test.js`](vscode-extension/test/unit.test.js))**: **`11/11 PASS`** (OpenAI tool-call/result round-tripping, `content: null` assistant tool turns, SSE chunk reassembly across arbitrary byte boundaries, Jinja schema normalization, string-to-schema argument coercion, system-role mapping).
- **Live Extension Host Integration ([`vscode-extension/test/integration/suite.js`](vscode-extension/test/integration/suite.js))**:
  1. Discovers `qwen3-coder-30b-a3b` via `vscode.lm.selectChatModels({ vendor: 'harness' })` with **`maxInputTokens = 28,672`** (`32,768 - 4,096`).
  2. Streams a direct `sendRequest` response (`"PONG"`).
  3. Runs a real 3-turn native tool-calling loop with `vscode.LanguageModelChatToolMode.Auto` where the model calls `list_dir({"path":"."})` $\rightarrow$ receives `LanguageModelToolResultPart` $\rightarrow$ calls `read_file({"path":"config.txt"})` $\rightarrow$ extracts the randomly generated workspace build token (`HX-836884`).

---

## 📸 Verified Platform Telemetry

![HARNESS Platform Screenshot](docs/screenshots/harness_platform.png)

*Figure: Real-time telemetry dashboard monitoring active local LLM inference with live `nvidia-smi` VRAM telemetry, long-context `q8_0/q8_0` GPU prefill metrics, and Skeptic's Trench benchmark ledgers.*

---

## 🏆 Skeptic's Trench 30B–32B Coder Torture-Test Ledger (2k Context Decode Sweep)

All 8 configurations below were executed on bare metal (**NVIDIA GeForce RTX 5060 8,151 MiB GDDR7 @ 448 GB/s + Intel Core i5-10400F 6C/12T + 32 GB DDR4-2133 dual-channel, PCIe Gen 3 x8**) by [`benchmarks/bench_elite_coder.py`](benchmarks/bench_elite_coder.py) under greedy decoding (`temperature = 0.0, top_k = 1, seed = 1`). Every raw `llama-server` log, `nvidia-smi` snapshot, and compiler/test output is committed in [`benchmarks/runs/elite_coder_trench/`](benchmarks/runs/elite_coder_trench/ELITE_CODER_RESULTS.md).

Every configuration had to pass three software engineering gates:
1. **`python_dp_and_intervals`**: `merge_intervals` + LeetCode #10 `is_match` regex dynamic programming (`'.'` and `'*'`) verified against **10/10 strict Python assertions**.
2. **`rust_lockfree_spsc`**: Generic thread-safe bounded `RingBuffer<T>` compiled and unit-tested via **`rustc --test -O`**.
3. **`async_deadlock_audit`**: Root-cause diagnosis and lock-free/bounded remediation of a `tokio::sync::Mutex` contention bug.

| Config | Model & Hardware Offload Strategy | CUDA Model Buffer | Graph Splits | **Pooled tok/s** (vs `1.16` base) | Per-Task tok/s (`Python DP` / `Rust SPSC` / `Async Audit`) | Prefill tok/s | Draft Acc ($\alpha$) | Compiler & Test Gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`m5_qwen3_coder_30b_moe_ncmoe32_pure`** | **`Qwen3-Coder-30B-A3B` Sparse MoE (`30.5B` Total / `3.3B` Active, `-ngl 99 --n-cpu-moe 32`, 2k ctx, `--no-op-offload`)** | **`6,463.4 MiB`** | **`162`** | **`16.37 tok/s` (`14.11x`)** | `16.12` / **`16.53`** / `16.38` | **`78.0 tok/s`** | n/a (zero-draft) | **`3/3 PASS`** |
| **`d1_qwen25_coder_14b_0_5b_ngl38`** | **`Qwen2.5-Coder-14B-Q4_K_S` (`-ngl 38/49`) + `0.5B` Drafter (`-ngld 99`)** | `6,074.8 MiB` | `76` | **`15.89 tok/s` (`13.70x`)** | `16.48` / **`19.17`** / `12.87` | `39.8 tok/s` | **`97.9%`** (`98.7%` Rust) | **`3/3 PASS`** |
| **`m2_qwen3_coder_30b_moe_ncmoe32_ngram`** | `Qwen3-Coder-30B-A3B` Sparse MoE (`-ngl 99 --n-cpu-moe 32` + `ngram-mod`) | `6,463.4 MiB` | `162` | **`15.37 tok/s` (`13.25x`)** | `15.43` / **`15.79`** / `14.68` | `77.4 tok/s` | `7.8%` | **`3/3 PASS`** |
| **`m1_qwen3_coder_30b_moe_ncmoe35`** | `Qwen3-Coder-30B-A3B` Sparse MoE (`-ngl 99 --n-cpu-moe 35`) | `5,441.9 MiB` | `168` | **`14.92 tok/s` (`12.86x`)** | `14.72` / **`15.07`** / `14.89` | `72.5 tok/s` | n/a (zero-draft) | **`3/3 PASS`** |
| **`m4_qwen3_coder_30b_moe_ncmoe31_ultra`** | `Qwen3-Coder-30B-A3B` Sparse MoE (`-ngl 99 --n-cpu-moe 31`, WDDM spillover test) | `6,787.4 MiB` | `160` | **`13.04 tok/s` (`11.24x`)** | `12.88` / **`13.11`** / `13.02` | `56.5 tok/s` | n/a (zero-draft) | **`3/3 PASS`** |
| **`m3_qwen3_coder_30b_moe_spec_0_6b`** | `Qwen3-Coder-30B-A3B` (`--n-cpu-moe 36`) + `Qwen3-0.6B` Neural Drafter (`-ngld 99`) | `5,068.4 MiB` | `170` | **`11.59 tok/s` (`9.99x`)** | `9.96` / **`15.81`** / `11.66` | `67.3 tok/s` | `76.9%` (`82.0%` Rust) | **`3/3 PASS`** |
| **`c5_dense32b_0_5b_ngl20_prio`** | **`Qwen2.5-Coder-32B` Dense (`-ngl 20/65`) + `0.5B` Drafter (`-ngld 99`) + `--prio 2`** | **`6,013.2 MiB`** | **`40`** | **`3.90 tok/s` (`3.36x`)** | `3.84` / **`4.80`** / `3.22` | `6.3 tok/s` | **`85.4%`** (`96.2%` Rust) | **`3/3 PASS`** |
| **`c6_dense32b_0_5b_ngl21_ultra`** | `Qwen2.5-Coder-32B` Dense (`-ngl 21/65`, WDDM spillover test) + `0.5B` Drafter | `6,274.8 MiB` | `42` | **`3.39 tok/s` (`2.92x`)** | `3.33` / **`4.08`** / `2.87` | `6.4 tok/s` | `82.5%` (`95.4%` Rust) | **`3/3 PASS`** |

### Empirical Hardware Laws Discovered & Proven on Bare Metal

1. **Hybrid Sparse MoE Partitioning (`19.88–21.40 tok/s` in 32k Agent Mode; `16.37 tok/s` pooled in 2k Trench Mode)**:
   - `Qwen3-Coder-30B-A3B-Instruct` (`Q4_K_M`, 30.5B total parameters, 48 layers, 128 routed experts per layer, top-8 active = 3.3B active parameters/token) offloads **100% of all 48 layers' shared GQA attention weights, KV cache, router gates, output head, plus 11–16 full MoE layers** into the RTX 5060's 448 GB/s GDDR7 via `-ngl 99 --n-cpu-moe 37` (at 32k ctx) or `--n-cpu-moe 32` (at 2k ctx).
   - For the MoE layers whose expert weights reside in host DDR4, only `8 / 128 = 6.25%` of routed experts are read per token (`~0.70–0.81 GB` DDR4 read/token vs `14.46 GB/token` for Dense 32B), eliminating **94%–95% of the DDR4 memory-bus bottleneck** while preserving 30.5B-parameter repository and compiler knowledge.
2. **The Sparse MoE Expert-Union Batching Law (`m5` vs `m3` & `m2`)**:
   - In a **Dense 32B** model (`c5`), verifying a speculative batch of $M = K+1 = 8$ tokens reads the exact same `12.61 GiB` (`12,914.54 MiB`) of DDR4 weights as $M = 1$, turning speculative batching into a **3.36× speedup (`3.90 tok/s` pooled, `4.80 tok/s` peak at `96.2%` draft acceptance)**.
   - In a **128-expert Top-8 Sparse MoE** model (`Qwen3-Coder-30B-A3B`), $M = 1$ activates only 8 experts (`6.25%`) per layer, whereas verifying a batch of $M = 7$ speculative tokens activates the **union** of 7 tokens' Top-8 selections (`~35–45` unique experts/layer = `27%–35%` of each CPU MoE layer), multiplying DDR4 expert reads per verification sweep by 4–5×. Thus zero-draft MoE (`m5`: **`16.37 tok/s`**) outperforms both `ngram-mod` (`m2`: `15.37 tok/s`) and neural-drafted MoE (`m3`: `11.59 tok/s` pooled).
3. **The Windows WDDM Physical GDDR7 Spillover Cliff (`m5` vs `m4`, `c5` vs `c6`)**:
   - On an 8,151 MiB GPU under Windows 11 WDDM, over-allocating model layers so that CUDA buffers spill into Shared GPU Memory over PCIe Gen 3 x8 degrades throughput by 13%–20%. Sizing `--n-cpu-moe 34` (16k ctx), `--n-cpu-moe 37` (32k ctx), and `--n-cpu-moe 42` (64k ctx) keeps total VRAM at `7,341–7,485 MiB` (`< 8,151 MiB`) with zero PCIe page thrashing.

---

## 📈 Dense 32B Ablation Progression (`Qwen2.5-Coder-32B-Instruct` `Q4_K_M`, `c0` → `c5`)

Captured by [`benchmarks/bench_qwen32b.py`](benchmarks/bench_qwen32b.py) and [`benchmarks/bench_elite_coder.py`](benchmarks/bench_elite_coder.py); full logs in [`benchmarks/runs/qwen32b_coder/QWEN32B_RESULTS.md`](benchmarks/runs/qwen32b_coder/QWEN32B_RESULTS.md) and [`benchmarks/runs/elite_coder_trench/ELITE_CODER_RESULTS.md`](benchmarks/runs/elite_coder_trench/ELITE_CODER_RESULTS.md):

| Configuration | GPU Layers (`32B` + Draft) | KV Cache Format (`2048` ctx) | PCIe Graph Splits | **Pooled tok/s** | Peak Task tok/s | Draft Acceptance ($\alpha$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`c0_raw_baseline`** | `18/65` | FP16 (`512.00 MiB`) | `606` | **1.16 tok/s** (`1.00x`) | `1.29 tok/s` | n/a |
| **`c1_tuned_kv`** | `21/65` | `q8_0` K / `q4_0` V (`208.00 MiB`) | `682` | **1.58 tok/s** (`1.36x`) | `1.59 tok/s` | n/a |
| **`c3_spec_ngram`** | `21/65` | `q8_0` K / `q4_0` V (`208.00 MiB`) | `682` | **1.59 tok/s** (`1.37x`) | `1.72 tok/s` | `89/197` (`45.2%`) |
| **`c2_spec_draft_1_5b`** | `17/65` + `29/29` (`1.5B`) | `q8_0` K / `q4_0` V (`208 MiB` + `56 MiB` FP16 draft) | `674` | **2.99 tok/s** (`2.58x`) | `3.65 tok/s` | `419/610` (`68.7%`) |
| **`c4_ultra_parallel_spec`** | `18/65` + `29/29` (`1.5B`) | `q8_0` K / `q4_0` V (`208 MiB` + `22.75 MiB` draft) | `36` (-94.7%) | **3.37 tok/s** (`2.91x`) | `4.44 tok/s` | `391/434` (`90.1%`) |
| **`c5_dense32b_0_5b_ngl20_prio`** | **`20/65` + `25/25` (`0.5B`)** | **`q8_0` K / `q4_0` V (`208 MiB` + `9 MiB` draft)** | **`40`** | **3.90 tok/s (`3.36x`)** | **`4.80 tok/s` (`96.2%` acc)** | **`499/584` (`85.4%`)** |

---

## 📊 Audited Systems & Hardware-Synchronized GPU Kernel Scorecard

Every standalone GPU kernel benchmark below uses explicit `ctx.synchronize()` (`wgpu::Maintain::Wait`) before and after each timed loop on the **NVIDIA GeForce RTX 5060** ([`benchmarks/runs/gpu_bench_32b_coder.txt`](benchmarks/runs/gpu_bench_32b_coder.txt)):

| Evaluation Task / Primitive | Verified Measurement | Hardware Grounding / Verification Mechanism |
| :--- | :--- | :--- |
| **30B Sparse MoE 32k Agent Mode (`s_ctx32k_q8q8_ncmoe37_gpuprefill`)** | **`446.86 tok/s` prefill**, **`19.88–21.40 tok/s` decode** | `-c 32768 --n-cpu-moe 37 -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024 --jinja` (`7,365 MiB` VRAM); see [`benchmarks/runs/agent_ctx/`](benchmarks/runs/agent_ctx/s_ctx32k_q8q8_ncmoe37_gpuprefill.json) |
| **30B Sparse MoE 64k Agent Mode (`t_ctx64k_q8q8_ncmoe42_gpuprefill`)** | **`380.52 tok/s` prefill**, **`14.04–18.86 tok/s` decode** | `-c 65536 --n-cpu-moe 42 -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024` (`7,341 MiB` VRAM); see [`benchmarks/runs/agent_ctx/`](benchmarks/runs/agent_ctx/t_ctx64k_q8q8_ncmoe42_gpuprefill.json) |
| **30B Sparse MoE 2k Trench Mode (`m5`, `Qwen3-Coder-30B-A3B`)** | **`16.37 tok/s` pooled** (`16.53 tok/s` Rust, `78.0 tok/s` prefill, `3/3` PASS) | `-ngl 99 --n-cpu-moe 32` (`6,463.4 MiB` GDDR7 + `11,227.9 MiB` DDR4); see [`ELITE_CODER_RESULTS.md`](benchmarks/runs/elite_coder_trench/ELITE_CODER_RESULTS.md) |
| **32B Dense Coder + `0.5B` Drafter (`c5`, `Qwen2.5-Coder-32B`)** | **`3.90 tok/s` pooled** (`4.80 tok/s` Rust, `96.2%` draft acc, `3/3` PASS) | `-ngl 20` + `-ngld 99` (`6,013.2 MiB` GDDR7); see [`ELITE_CODER_RESULTS.md`](benchmarks/runs/elite_coder_trench/ELITE_CODER_RESULTS.md) |
| **14B Dense Coder + `0.5B` Drafter (`d1`, `Qwen2.5-Coder-14B`)** | **`15.89 tok/s` pooled** (`19.17 tok/s` Rust, `98.7%` draft acc, `3/3` PASS) | `-ngl 38` + `-ngld 99` (`6,074.8 MiB` GDDR7); see [`ELITE_CODER_RESULTS.md`](benchmarks/runs/elite_coder_trench/ELITE_CODER_RESULTS.md) |
| **Activation-Sparse GPU INT4 GEMV (`sparse_gemv_q4.wgsl`)** | **`215.05 μs`** at 75% sparsity (vs **`311.10 μs`** dense Q4 & **`752.03 μs`** FP32 — **1.45× / 3.50× faster**) | Standalone `wgpu` kernel benchmark (`Maintain::Wait`) on RTX 5060 ($M=1, K=5120, N=5120$); see [`gpu_bench_32b_coder.txt`](benchmarks/runs/gpu_bench_32b_coder.txt) |
| **Native GPU Top-8 MoE Router (`moe_router_topk.wgsl`)** | **8 speculative tokens × 128 experts** (`#12: 0.127, #109: 0.126, #60: 0.126`) | Standalone `wgpu` WGSL Top-8 softmax routing verified against CPU reference; see [`gpu_bench_32b_coder.txt`](benchmarks/runs/gpu_bench_32b_coder.txt) |
| **Real GGUF Weight Extraction & GPU Execution (`[9/9]`)** | **`[1×5120] -> [1×64]`** (`y[0..3] = [-0.0654, -0.1096, 0.0423]`) | Mmaps `Qwen2.5-Coder-32B-Q4_K_M`, unpacks `blk.0.attn_norm.weight` + `blk.0.attn_q.weight` (`Q4_K`), runs on RTX 5060 |
| **Native GPU 32B SwiGLU & RMSNorm** | **`435.08 μs` (`8,133.9 M-elem/s`)** & **`766.89 μs` (`1.34M tok/s`)** | Standalone `wgpu` kernel benchmark ($128 \times 27648$ FFN & $1024 \times 5120$ hidden); see [`gpu_bench_32b_coder.txt`](benchmarks/runs/gpu_bench_32b_coder.txt) |
| **DFA & Pushdown Constrained Decoding** | tens of ns per mask (`harness-cli stress`) | Precompiled bitmasks + pushdown delimiter state machine (`crates/harness-safety/src/constrained.rs`); see [`microbench_stress.txt`](benchmarks/runs/microbench_stress.txt) |
| **70B Raw Hardware Execution (8GB VRAM)** | **1.68 tok/s** (`c3_draft1b_k4` on `llama3.1:70b-instruct-q2_K`) | Historical 70.55B baseline ledger; raw logs committed in [`RESULTS.md`](benchmarks/runs/RESULTS.md) |

---

## ⚡ Quickstart: Launch Local 32k Agent Engine & Use in VS Code Native Chat

### Prerequisites
- **Rust Toolchain**: `rustc 1.85+` (`cargo`)
- **Node.js**: `v20+` (for the VS Code extension tests and React telemetry GUI)
- **Local GGUF Backend**: `llama-server` (auto-detected from `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-server.exe` or `OLLAMA_LIB`)

### 1. Start the 32k Agent Engine (`:8089`) & HARNESS OpenAI Proxy (`:8090`)
```powershell
# 1. Start the bare-metal Coder engine on port 8089 (Default -Mode Agent = 32,768 ctx, q8_0/q8_0 KV, GPU prefill, --jinja)
#    Other Modes: -Mode MoE30B (2k ctx), -Mode Dense32B (32B + 0.5B Drafter), -Mode Dense14B (14B + 0.5B Drafter)
powershell -ExecutionPolicy Bypass -File .\launch_qwen32b_coder.ps1 -Mode Agent

# 2. Start the lossless HARNESS OpenAI & VS Code Agent proxy on port 8090
cargo run --release -p harness-cli -- serve --port 8090 --backend http://127.0.0.1:8089
```

### 2. Install & Use the Native VS Code Chat Provider Extension (`vscode-extension/`)
```powershell
# Copy the extension into your VS Code extensions folder
$extDest = "$env:USERPROFILE\.vscode\extensions\harness-local.harness-vscode-0.1.0"
New-Item -ItemType Directory -Force -Path "$extDest\lib" | Out-Null
Copy-Item .\vscode-extension\package.json, .\vscode-extension\extension.js, .\vscode-extension\README.md -Destination $extDest -Force
Copy-Item .\vscode-extension\lib\openai.js -Destination "$extDest\lib\openai.js" -Force
```
Reload VS Code (`Developer: Reload Window`), open the built-in **Chat** view (`Ctrl+Alt+I`), click the model picker, and select **`HARNESS: qwen3-coder-30b-a3b`**. All native VS Code Agent tools (file reading/editing, workspace search, terminal execution) work directly with the local model.

### 3. Run All Verification & SWE Benchmark Suites
```powershell
# Rust workspace unit + OpenAI proxy integration tests
cargo test --workspace

# VS Code extension unit tests (11/11) + live VS Code Extension Host integration test
cd vscode-extension
node test/unit.test.js
node test/run-integration.js
cd ..

# Validate and run the 16-task HARNESS-SWE-Lite multi-tool bug-fix benchmark
python benchmarks/bench_swe_lite.py --validate
python benchmarks/bench_swe_lite.py --endpoint http://127.0.0.1:8090/v1 --tag qwen3_30b_a3b_agent32k

# Run EvalPlus HumanEval / HumanEval+ benchmark
python benchmarks/bench_humaneval.py --endpoint http://127.0.0.1:8090/v1 --limit 12 --tag qwen3_30b_a3b_he20

# Run the 9-stage standalone wgpu/Vulkan GPU kernel benchmark
cargo run --release -p harness-cli -- gpu-bench --matrix-dim 1024 --iterations 50
```

---

## 📜 Repository Structure

```text
├── crates/
│   ├── harness-server      # Lossless Axum OpenAI & tool-calling proxy (:8090), SSE cancellation, live nvidia-smi /metrics
│   ├── harness-cli         # CLI entrypoint (`serve`, `coder32b`, `gpu-bench`, `code-check`, `reason`, `stress`, `tune`)
│   ├── harness-core        # Tensor abstractions, HardwareProfile, standalone wgpu/Vulkan WGSL kernels
│   ├── harness-loader      # GGUF v2/v3 metadata & tensor unpacker (`extract_tensor_f32` for F32/F16/Q8_0/Q4_K)
│   ├── harness-quant       # Bit-exact GGUF Q4_K (`BlockQ4K`, imatrix WLS, fused dot_f32), OCP FP8 E4M3, 64-group NF4
│   ├── harness-attention   # Decode-aware FlashAttention-3, Quantized Int8/Int4 KV Cache, LIF Spiking Attention, RadixPrefixCache
│   ├── harness-models      # Qwen2.5-Coder-32B/1.5B, Qwen3-30B-A3B MoE, DeepSeek, LLaMA architectures
│   ├── harness-pipeline    # OnlineDraftCalibrator, RepoSuffixDrafter, Speculative engine, CodeQualityEngine, TTC
│   ├── harness-safety      # Pushdown/DFA constrained decoding, Cortical lateral inhibition, Shannon entropy gating
│   ├── harness-rag         # Hippocampal fast-slow dual memory, Cortical engram vector store
│   └── harness-mcp         # Model Context Protocol (MCP) server for IDE pair-programming
├── vscode-extension/       # Native VS Code Language Model Chat Provider (`vscode.lm.registerLanguageModelChatProvider`)
├── benchmarks/             # Reproducible benchmark runners (`bench_agent_ctx.py`, `bench_swe_lite.py`, `bench_humaneval.py`, `bench_elite_coder.py`)
└── frontend/               # React/TypeScript telemetry dashboard (`http://localhost:3000`)
```

---

## 📄 License

Licensed under the **MIT License** ([LICENSE](LICENSE)).
