# Raw-captured results (computed from files in `benchmarks/runs/`)

Every number below is derived by `summarize_runs.py` from the verbatim llama-server `timings` JSON.
Model: `llama3.1:70b-instruct-q2_K` (all 70.55B parameters, 26,375,113,056 bytes), greedy decoding (temperature 0, top_k 1, seed 1).

| config | GPU layers | tokens | gen time (s) | **tok/s (pooled)** | per-prompt tok/s (prose / code / quote) | draft accepted | lossless vs baseline |
|---|---|---|---|---|---|---|---|
| c0_default_fit | 16/81 (verbose probe log) | 144 | 146.1 | **0.986** | 0.97 / 0.96 / 0.96 | n/a | reference |
| c1_tuned_fit | 19/81 (verbose probe log) | 144 | 139.0 | **1.036** | 0.99 / 1.03 / 1.02 | n/a | identical; identical; identical |
| c2_ngram_simple | n/a | 144 | 123.7 | **1.164** | 0.94 / 1.04 / 1.66 | 35/56 (62%) | diverges@char 131/287; identical; identical |
| c3_draft1b_k4 | 11/81 (explicit -ngl) | 144 | 85.8 | **1.678** | 1.28 / 2.01 / 1.82 | 106/134 (79%) | identical; identical; identical |
| c4_draft1b_plus_ngram | 11/81 (explicit -ngl) | 144 | 102.0 | **1.411** | 0.99 / 1.62 / 1.84 | 108/152 (71%) | identical; identical; identical |
| c5_tuned_threads4 | n/a | 144 | 180.3 | **0.799** | 0.78 / 0.74 / 0.83 | n/a | identical; identical; identical |
| m7b_qwen25coder_resident | 81/81 (explicit -ngl) | 144 | 1.9 | **77.610** | 75.61 / 76.46 / 75.91 | n/a | n/a (different model: Qwen2.5-Coder-7B Q4_K_M) |

## Exact command lines and hardware state (raw)

### c0_default_fit
```
<USERPROFILE>\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe --model <USERPROFILE>\.ollama\models\blobs\sha256-ba1103315c449ad06c9f5fd94230bde5bcf977f794af70afb107d29153c3cd53 --load-mode none -np 1 --port 8089 --host 127.0.0.1 --no-webui -c 2048 -t 6
```
- git HEAD: `bf39f1c`  started: `2026-10-03T17:09:33`  free RAM before launch: 22.84 GB of 31.9 GB
- nvidia-smi before launch / after load:
```
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
1761 MiB, 6135 MiB, 8151 MiB, 2, 8, 810 MHz
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
7451 MiB, 445 MiB, 8151 MiB, 3, 8, 7001 MHz
```

### c1_tuned_fit
```
<USERPROFILE>\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe --model <USERPROFILE>\.ollama\models\blobs\sha256-ba1103315c449ad06c9f5fd94230bde5bcf977f794af70afb107d29153c3cd53 --load-mode none -np 1 --port 8089 --host 127.0.0.1 --no-webui -c 2048 -t 6 -fitt 384 -ctk q8_0 -ctv q8_0 -fa on -b 256 -ub 128
```
- git HEAD: `bf39f1c`  started: `2026-10-03T17:13:51`  free RAM before launch: 22.56 GB of 31.9 GB
- nvidia-smi before launch / after load:
```
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
1465 MiB, 6431 MiB, 8151 MiB, 3, 8, 7001 MHz
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
7624 MiB, 272 MiB, 8151 MiB, 3, 8, 13801 MHz
```

### c2_ngram_simple
```
<USERPROFILE>\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe --model <USERPROFILE>\.ollama\models\blobs\sha256-ba1103315c449ad06c9f5fd94230bde5bcf977f794af70afb107d29153c3cd53 --load-mode none -np 1 --port 8089 --host 127.0.0.1 --no-webui -c 2048 -t 6 -fitt 384 -ctk q8_0 -ctv q8_0 -fa on -b 256 -ub 128 --spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8
```
- git HEAD: `bf39f1c`  started: `2026-10-03T17:18:22`  free RAM before launch: 21.66 GB of 31.9 GB
- nvidia-smi before launch / after load:
```
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
1663 MiB, 6233 MiB, 8151 MiB, 3, 8, 7001 MHz
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
7653 MiB, 243 MiB, 8151 MiB, 3, 8, 7001 MHz
```

### c3_draft1b_k4
```
<USERPROFILE>\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe --model <USERPROFILE>\.ollama\models\blobs\sha256-ba1103315c449ad06c9f5fd94230bde5bcf977f794af70afb107d29153c3cd53 --load-mode none -np 1 --port 8089 --host 127.0.0.1 --no-webui -c 2048 -t 6 -ctk q8_0 -ctv q8_0 -fa on -b 256 -ub 128 -ngl 11 --model-draft <USERPROFILE>\.ollama\models\blobs\sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45 -ngld 99 --spec-draft-n-max 4 --spec-type draft-simple
```
- git HEAD: `bf39f1c`  started: `2026-10-03T17:22:23`  free RAM before launch: 22.33 GB of 31.9 GB
- nvidia-smi before launch / after load:
```
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
1122 MiB, 6774 MiB, 8151 MiB, 3, 8, 14001 MHz
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
6563 MiB, 1333 MiB, 8151 MiB, 3, 8, 7001 MHz
```

### c4_draft1b_plus_ngram
```
<USERPROFILE>\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe --model <USERPROFILE>\.ollama\models\blobs\sha256-ba1103315c449ad06c9f5fd94230bde5bcf977f794af70afb107d29153c3cd53 --load-mode none -np 1 --port 8089 --host 127.0.0.1 --no-webui -c 2048 -t 6 -ctk q8_0 -ctv q8_0 -fa on -b 256 -ub 128 -ngl 11 --model-draft <USERPROFILE>\.ollama\models\blobs\sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45 -ngld 99 --spec-draft-n-max 4 --spec-type ngram-simple,draft-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8
```
- git HEAD: `bf39f1c`  started: `2026-10-03T17:25:45`  free RAM before launch: 22.49 GB of 31.9 GB
- nvidia-smi before launch / after load:
```
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
1159 MiB, 6737 MiB, 8151 MiB, 3, 8, 7001 MHz
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
6576 MiB, 1320 MiB, 8151 MiB, 3, 8, 7001 MHz
```

### c5_tuned_threads4
```
<USERPROFILE>\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe --model <USERPROFILE>\.ollama\models\blobs\sha256-ba1103315c449ad06c9f5fd94230bde5bcf977f794af70afb107d29153c3cd53 --load-mode none -np 1 --port 8089 --host 127.0.0.1 --no-webui -c 2048 -t 4 -fitt 384 -ctk q8_0 -ctv q8_0 -fa on -b 256 -ub 128
```
- git HEAD: `bf39f1c`  started: `2026-10-03T17:29:33`  free RAM before launch: 22.24 GB of 31.9 GB
- nvidia-smi before launch / after load:
```
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
1553 MiB, 6343 MiB, 8151 MiB, 3, 8, 7001 MHz
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
7554 MiB, 342 MiB, 8151 MiB, 3, 8, 7001 MHz
```

### m7b_qwen25coder_resident
```
<USERPROFILE>\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe --model <USERPROFILE>\.ollama\models\blobs\sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463 --load-mode none -np 1 --port 8089 --host 127.0.0.1 --no-webui -c 2048 -t 6 -ngl 99
```
- git HEAD: `bf39f1c`  started: `2026-10-03T17:40:32`  free RAM before launch: 22.19 GB of 31.9 GB
- nvidia-smi before launch / after load:
```
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
1403 MiB, 6493 MiB, 8151 MiB, 3, 8, 7001 MHz
memory.used [MiB], memory.free [MiB], memory.total [MiB], pcie.link.gen.current, pcie.link.width.current, clocks.current.memory [MHz]
5962 MiB, 1934 MiB, 8151 MiB, 3, 8, 7001 MHz
```

## Computed cross-check: implied CPU-side weight read rate (plain autoregressive configs only)

`(26,375,113,056 B - GPU model buffer from the verbose probe log) x pooled tok/s`. This is arithmetic on logged values, not a bandwidth measurement; `hardware.txt` gives the calculated DDR4 peak for comparison.

| config | GPU layers | CPU-resident weights (GB) | pooled tok/s | implied read rate (GB/s) |
|---|---|---|---|---|
| c0_default_fit | 16/81 | 20.80 | 0.986 | 20.5 |
| c1_tuned_fit | 19/81 | 19.85 | 1.036 | 20.6 |
