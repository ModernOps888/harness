# Skeptic's Trench 30B–32B Coder Torture-Test Ledger (Bare-Metal Verified)

Every configuration below was executed on bare metal (**NVIDIA GeForce RTX 5060 8,151 MiB GDDR7 @ 448 GB/s + Intel Core i5-10400F 6C/12T + 32 GB DDR4-2133 dual-channel, PCIe Gen 3 x8**) by [`benchmarks/bench_elite_coder.py`](../../bench_elite_coder.py) under greedy decoding (`temperature = 0.0, top_k = 1, seed = 1`).

Unlike synthetic text-only benchmarks, every generated response was automatically compiled and executed against three strict software engineering gates:
1. **`python_dp_and_intervals`**: `merge_intervals` + LeetCode #10 `is_match` regex dynamic programming verified against **10/10 Python assertions**.
2. **`rust_lockfree_spsc`**: Generic thread-safe bounded `RingBuffer<T>` compiled and unit-tested with **`rustc --test -O`**.
3. **`async_deadlock_audit`**: Root-cause diagnosis and lock-free/bounded remediation of a `tokio::sync::Mutex` contention bug.

| Config | Architecture & Strategy | CUDA Model Buffer | Graph Splits | **Pooled tok/s** (vs `1.16` base) | Per-Task tok/s (`Python DP` / `Rust SPSC` / `Async Audit`) | Prefill tok/s | Draft Acc ($\alpha$) | Compiler/Test Gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`m1_qwen3_coder_30b_moe_ncmoe35`** | Qwen3-Coder-30B-A3B Sparse MoE (30.5B Total / 3.3B Active, -ngl 99 --n-cpu-moe 35) | `5441.9 MiB` | `168` | **`14.92 tok/s` (`12.86x`)** | `14.72` / **`15.07`** / `14.89` | `72.3 tok/s` | n/a (zero-draft) | **`3/3 PASS`** |
| **`m2_qwen3_coder_30b_moe_ncmoe32_ngram`** | Qwen3-Coder-30B-A3B Sparse MoE (-ngl 99 --n-cpu-moe 32 + ngram-mod Speculative) | `6463.4 MiB` | `162` | **`15.37 tok/s` (`13.25x`)** | `15.43` / **`15.79`** / `14.68` | `76.5 tok/s` | `7.8%` | **`3/3 PASS`** |
| **`m3_qwen3_coder_30b_moe_spec_0_6b`** | Qwen3-Coder-30B-A3B Sparse MoE (--n-cpu-moe 36) + Qwen3-0.6B Speculative Drafter | `5068.4 MiB` | `170` | **`11.59 tok/s` (`9.99x`)** | `9.96` / **`15.81`** / `11.66` | `67.3 tok/s` | `76.9%` | **`3/3 PASS`** |
| **`c5_dense32b_0_5b_ngl20_prio`** | Qwen2.5-Coder-32B (-ngl 20) + 0.5B Drafter (-ngld 99) + --no-op-offload + Realtime Prio | `6013.2 MiB` | `40` | **`3.90 tok/s` (`3.36x`)** | `3.84` / **`4.80`** / `3.22` | `6.3 tok/s` | `85.4%` | **`3/3 PASS`** |
| **`d1_qwen25_coder_14b_spec_0_5b`** | Qwen2.5-Coder-14B-Q4_K_S (-ngl 38) + 0.5B Drafter (-ngld 99) + Q8_0/Q4_0 KV | `6074.8 MiB` | `76` | **`15.89 tok/s` (`13.70x`)** | `16.48` / **`19.17`** / `12.87` | `39.8 tok/s` | `88.8%` | **`3/3 PASS`** |
| **`m4_qwen3_coder_30b_moe_ncmoe31_ultra`** | Qwen3-Coder-30B-A3B Sparse MoE (-ngl 99 --n-cpu-moe 31, 17 Full MoE + 48 Attn Layers in 8GB GDDR7) | `6787.4 MiB` | `160` | **`13.04 tok/s` (`11.24x`)** | `12.88` / **`13.11`** / `13.02` | `54.3 tok/s` | n/a (zero-draft) | **`3/3 PASS`** |
| **`c6_dense32b_0_5b_ngl21_ultra`** | Qwen2.5-Coder-32B (-ngl 21, 21 Layers in GDDR7) + 0.5B Drafter (-ngld 99) + Realtime Prio | `6274.8 MiB` | `42` | **`3.39 tok/s` (`2.92x`)** | `3.33` / **`4.08`** / `2.87` | `6.4 tok/s` | `82.5%` | **`3/3 PASS`** |
| **`m5_qwen3_coder_30b_moe_ncmoe32_pure`** | Qwen3-Coder-30B-A3B Sparse MoE (-ngl 99 --n-cpu-moe 32, 16 Full MoE + 48 Attn Layers in GDDR7, Zero Draft Overhead) | `6463.4 MiB` | `162` | **`16.37 tok/s` (`14.11x`)** | `16.12` / **`16.53`** / `16.38` | `77.0 tok/s` | n/a (zero-draft) | **`3/3 PASS`** |

## Empirical Hardware Laws Discovered in the Skeptic's Trench

1. **Sparse MoE Hybrid Placement (`m5_qwen3_coder_30b_moe_ncmoe32_pure`: `16.37 tok/s` pooled, `16.53 tok/s` peak — `14.11x` over dense baseline)**:
   - `Qwen3-Coder-30B-A3B-Instruct` (`Q4_K_M`, 30.5B total parameters, 48 layers, 128 routed experts/layer, top-8 active = 3.3B active parameters/token) offloads **100% of all 48 layers' shared GQA attention projections, KV cache (`78 MiB`), router gates, output head, plus 16 full MoE layers (`6,463.42 MiB` CUDA model buffer)** into the RTX 5060's 448 GB/s GDDR7 via `-ngl 99 --n-cpu-moe 32`.
   - For the 32 MoE layers whose expert weights reside in DDR4 (`11,227.92 MiB`), only `8 / 128 = 6.25%` of routed experts are read per token (`~0.70 GB` DDR4 read/token vs `14.46 GB` for Dense 32B), eliminating 95.2% of the DDR4 bandwidth bottleneck while retaining 30.5B-parameter coding knowledge.
2. **The Sparse MoE Expert-Union Batching Law (`m5` vs `m3` and `m2`)**:
   - In Dense 32B (`c5_dense32b_0_5b_ngl20_prio`), verifying a speculative batch of $M = K+1 = 8$ tokens reads the exact same `12.61 GiB` (`12,914.54 MiB`) of CPU weights as $M = 1$, delivering a **3.36x speedup (`3.90 tok/s` pooled, `4.80 tok/s` peak at `96.2%` draft acceptance)**.
   - In 128-expert Top-8 Sparse MoE (`Qwen3-Coder-30B-A3B`), $M = 1$ activates exactly 8 experts (`6.25%`) per layer, whereas verifying a batch of $M = 7$ tokens activates the **union** of 7 tokens' Top-8 choices (`~35–45` unique experts/layer = `27%–35%` of each CPU MoE layer), multiplying DDR4 traffic per pass by 4–5x. Consequently, pure zero-draft MoE (`m5`: **`16.37 tok/s`**) outperforms both `ngram-mod` (`m2`: `15.37 tok/s`) and neural-drafted MoE (`m3`: `11.59 tok/s` pooled).
3. **The Windows WDDM `~6,650 MiB` Physical GDDR7 Spillover Cliff (`m5` vs `m4`, `c5` vs `c6`)**:
   - On an 8,151 MiB GPU under Windows 11 WDDM, pushing total CUDA allocation from `6,579 MiB` (`--n-cpu-moe 32`, `m5`: **`16.37 tok/s`**) to `6,920 MiB` (`--n-cpu-moe 31`, `m4`: `13.04 tok/s`) or from `6,565 MiB` (`-ngl 20`, `c5`: **`3.90 tok/s`**) to `6,845 MiB` (`-ngl 21`, `c6`: `3.39 tok/s`) causes WDDM to silently evict ~200–250 MiB of pages into Shared GPU Memory over PCIe Gen 3 x8, degrading throughput by 13%–20%.
4. **The Long-Context (`16k–64k`) GPU Prefill & Symmetric `q8_0/q8_0` KV Law (`benchmarks/runs/agent_ctx/`)**:
   - Mixed `-ctk q8_0 -ctv q4_0` KV quantization lacks a fast CUDA FlashAttention kernel in `llama-server` and, combined with `--no-op-offload -b 512 -ub 256`, collapses prefill to `27.40 tok/s` and decode at 3.1k depth to `5.67 tok/s` (`h_ctx16k_q8q4_ncmoe34_control`).
   - Switching to symmetric **`-ctk q8_0 -ctv q8_0 -fa on -b 2048 -ub 1024` (without `--no-op-offload`)** engages CUDA FlashAttention and GPU batch prefill at full `13,801 MHz` GDDR7 memory clock (`448 GB/s`), boosting prefill **15x** (`411.27 tok/s` at 16k ctx, **`446.86–452.53 tok/s` at 32k ctx**, `380.52 tok/s` at 64k ctx) and sustaining **`19.88–21.40 tok/s` decode** at 16k–32k context (`14.04–18.86 tok/s` at 64k context).
