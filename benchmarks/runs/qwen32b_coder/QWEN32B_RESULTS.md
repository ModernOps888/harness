# Qwen2.5-Coder-32B Factual Benchmark & Speculative Optimization

Evaluated on bare metal: NVIDIA RTX 5060 8GB GDDR7 + 32GB DDR4-2133 Host RAM.
Generated: 2026-10-07T13:20:01.863290

| Configuration | Offloaded Layers | Tokens | Total Gen Time | **Pooled tok/s** | Per-Prompt tok/s (LRU / Async / VecLog) | Draft Acceptance |
|---|---|---|---|---|---|---|
| c0_raw_baseline | 18/65 | 576 | 498.3s | **1.16** | 1.22 / 1.29 / 0.99 | n/a |
| c1_tuned_kv | 21/65 | 576 | 365.2s | **1.58** | 1.56 / 1.59 / 1.56 | n/a |
| c2_spec_draft_1_5b | 17/65 | 576 | 192.5s | **2.99** | 3.65 / 3.24 / 2.35 | 419/610 (69%) |
| c3_spec_ngram | 21/65 | 576 | 362.6s | **1.59** | 1.72 / 1.55 / 1.49 | 89/197 (45%) |

## Prompt Quality & Code Assessment

### Configuration: `c0_raw_baseline`
#### Prompt `humaneval_merge_intervals` Sample:
```rust
```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        if current[0] <= last_merged[1]:
            merged[-1][1] = max(last_merged[1], current[1])
        else:
            merged.appe...
```

#### Prompt `lru_cache_rust` Sample:
```rust
Here's a production-quality, thread-safe LRU Cache in Rust using the standard library's synchronization primitives. The implementation ensures O(1) get/put operations and includes unit tests.

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

struct Node {
    key: String,
    value: String,
    prev: Option<Arc<RwLock<Node>>>,
    nex...
```

#### Prompt `async_deadlock_audit` Sample:
```rust
The provided Rust code snippet is mostly correct in terms of avoiding deadlocks and race conditions, but there are a few considerations and potential improvements:

1. **Deadlocks**: The code does not inherently contain any deadlock conditions. The `Mutex` ensures that only one task can modify the `State` at a time, preventing race conditions on the `count` and `waiters` fields.

2. **Race Conditi...
```

### Configuration: `c1_tuned_kv`
#### Prompt `humaneval_merge_intervals` Sample:
```rust
```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        if current[0] <= last_merged[1]:
            merged[-1][1] = max(last_merged[1], current[1])
        else:
            merged.appe...
```

#### Prompt `lru_cache_rust` Sample:
```rust
Here's a production-quality, thread-safe LRU Cache in Rust using the standard library's synchronization primitives. The implementation ensures O(1) get/put operations and handles evictions and capacity checks.

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

struct LruNode<K, V> {
    key: K,
    value: V,
    prev: Option<Arc<RwLock...
```

#### Prompt `async_deadlock_audit` Sample:
```rust
The provided Rust code snippet is mostly correct in terms of avoiding deadlocks and race conditions, but there are a few considerations and potential improvements:

1. **Deadlocks**: The code does not inherently contain any deadlock conditions. The `Mutex` ensures that only one task can modify the `State` at a time, preventing race conditions on the `count` and `waiters` fields.

2. **Race Conditi...
```

### Configuration: `c2_spec_draft_1_5b`
#### Prompt `humaneval_merge_intervals` Sample:
```rust
```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        if current[0] <= last_merged[1]:
            merged[-1][1] = max(last_merged[1], current[1])
        else:
            merged.appe...
```

#### Prompt `lru_cache_rust` Sample:
```rust
Here's a production-quality, thread-safe LRU Cache in Rust using the standard library's synchronization primitives. The implementation ensures O(1) get/put operations and includes unit tests.

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

struct LruNode<K, V> {
    key: K,
    value: V,
    prev: Option<Arc<RwLock<LruNode<K, V>>>>,...
```

#### Prompt `async_deadlock_audit` Sample:
```rust
The provided Rust code snippet is mostly correct in terms of avoiding deadlocks and race conditions, but there are a few considerations and potential improvements:

1. **Deadlocks**: The code does not inherently contain any deadlock conditions. The `Mutex` ensures that only one task can modify the `State` at a time, preventing race conditions on the `count` and `waiters` fields.

2. **Race Conditi...
```

### Configuration: `c3_spec_ngram`
#### Prompt `humaneval_merge_intervals` Sample:
```rust
```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        if current[0] <= last_merged[1]:
            merged[-1][1] = max(last_merged[1], current[1])
        else:
            merged.appe...
```

#### Prompt `lru_cache_rust` Sample:
```rust
Here's a production-quality, thread-safe LRU Cache in Rust using the standard library's synchronization primitives. The implementation ensures O(1) get/put operations and includes unit tests.

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

struct Node {
    key: String,
    value: String,
    prev: Option<Arc<RwLock<Node>>>,
    nex...
```

#### Prompt `async_deadlock_audit` Sample:
```rust
The provided Rust code snippet is mostly correct in terms of avoiding deadlocks and race conditions, but there are a few considerations and potential improvements:

1. **Deadlocks**: The code does not inherently contain any deadlock conditions. The `Mutex` ensures that only one task can modify the `State` at a time, preventing race conditions on the `count` and `waiters` fields.

2. **Race Conditi...
```
