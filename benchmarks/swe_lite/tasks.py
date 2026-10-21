"""HARNESS-SWE-Lite: 16 self-contained bug-fix tasks (custom, NOT a published benchmark).

Each task = a tiny Python project with a real bug, an issue statement, a VISIBLE failing test and a
HIDDEN test file the agent never sees (extra edge cases). A task counts as resolved only if, after the
agent's edits, BOTH the visible and the hidden tests pass. `bench_swe_lite.py --validate` proves every
task is well-posed: the buggy snapshot must FAIL, the reference solution must PASS.
"""

TASKS = []


def task(id, issue, files, hidden, solution):
    TASKS.append(dict(id=id, issue=issue.strip(), files=files, hidden=hidden, solution=solution))


# 01 ---------------------------------------------------------------------------------------------
task(
    "t01_lru_get_refresh",
    "`LRUCache.get` does not mark an entry as recently used, so the wrong key is evicted. "
    "Reading a key must make it the most recently used. Fix the bug in cache.py.",
    {
        "cache.py": r'''
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._d = OrderedDict()

    def get(self, key):
        return self._d.get(key)

    def put(self, key, value):
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)
''',
        "test_cache.py": r'''
from cache import LRUCache


def test_get_refreshes_recency():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)
    assert c.get("b") is None
    assert c.get("a") == 1 and c.get("c") == 3
''',
    },
    {
        "test_hidden.py": r'''
import pytest
from cache import LRUCache


def test_update_refreshes_and_len():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2); c.put("a", 10); c.put("c", 3)
    assert c.get("b") is None and c.get("a") == 10 and len(c) == 2


def test_capacity_one_and_missing():
    c = LRUCache(1)
    c.put("x", 1); c.put("y", 2)
    assert c.get("x") is None and c.get("y") == 2


def test_invalid_capacity():
    with pytest.raises(ValueError):
        LRUCache(0)


def test_get_chain_order():
    c = LRUCache(3)
    for k in "abc":
        c.put(k, k)
    c.get("a"); c.get("b")
    c.put("d", "d")
    assert c.get("c") is None
    assert [c.get(k) for k in "abd"] == ["a", "b", "d"]
''',
    },
    {
        "cache.py": r'''
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._d = OrderedDict()

    def get(self, key):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value):
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)
''',
    },
)

# 02 ---------------------------------------------------------------------------------------------
task(
    "t02_merge_intervals",
    "`merge_intervals` must merge overlapping AND touching intervals (e.g. [1,2] and [2,3] -> [1,3]) "
    "and must work on unsorted input (tuples or lists in) without modifying its input. It currently returns wrong results. Fix intervals.py.",
    {
        "intervals.py": r'''
def merge_intervals(intervals):
    """Merge overlapping or touching [start, end] intervals; return a sorted list of [start, end]."""
    result = []
    for start, end in intervals:
        if result and start < result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
''',
        "test_intervals.py": r'''
from intervals import merge_intervals


def test_unsorted():
    assert merge_intervals([[8, 10], [1, 3], [2, 6]]) == [[1, 6], [8, 10]]
''',
    },
    {
        "test_hidden.py": r'''
from intervals import merge_intervals


def test_touching():
    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]


def test_nested_and_empty_and_single():
    assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
    assert merge_intervals([]) == []
    assert merge_intervals([[5, 5]]) == [[5, 5]]


def test_input_not_mutated_and_disjoint():
    data = [[5, 6], [1, 2]]
    assert merge_intervals(data) == [[1, 2], [5, 6]]
    assert data == [[5, 6], [1, 2]]


def test_tuples_in():
    assert merge_intervals([(3, 4), (1, 3)]) == [[1, 4]]
''',
    },
    {
        "intervals.py": r'''
def merge_intervals(intervals):
    """Merge overlapping or touching [start, end] intervals; return a sorted list of [start, end]."""
    result = []
    for start, end in sorted(([s, e] for s, e in intervals), key=lambda x: x[0]):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
''',
    },
)

# 03 ---------------------------------------------------------------------------------------------
task(
    "t03_mutable_default",
    "`add_tag` leaks state between calls because of a shared default argument, and it also mutates the list "
    "passed in. It must return a NEW list and never mutate or share state. Fix tags.py.",
    {
        "tags.py": r'''
def add_tag(tag, tags=[]):
    """Return a list containing the existing tags plus `tag` (no duplicates, order preserved)."""
    if tag not in tags:
        tags.append(tag)
    return tags
''',
        "test_tags.py": r'''
from tags import add_tag


def test_calls_are_independent():
    assert add_tag("a") == ["a"]
    assert add_tag("b") == ["b"]
''',
    },
    {
        "test_hidden.py": r'''
from tags import add_tag


def test_input_not_mutated_and_new_list():
    base = ["x"]
    out = add_tag("y", base)
    assert base == ["x"] and out == ["x", "y"] and out is not base


def test_duplicates_ignored_but_still_new_list():
    base = ["x"]
    out = add_tag("x", base)
    assert out == ["x"] and out is not base


def test_default_never_shared():
    a = add_tag("p"); a.append("junk")
    assert add_tag("q") == ["q"]
''',
    },
    {
        "tags.py": r'''
def add_tag(tag, tags=None):
    """Return a list containing the existing tags plus `tag` (no duplicates, order preserved)."""
    out = list(tags) if tags else []
    if tag not in out:
        out.append(tag)
    return out
''',
    },
)

# 04 ---------------------------------------------------------------------------------------------
task(
    "t04_split_bill_cents",
    "`split_bill(total, n)` splits an amount (a decimal string like '100.00') into n shares in cents. The shares "
    "must always sum to exactly the total: the first (total_cents % n) shares get one extra cent. Currently "
    "cents are lost. Return a list of strings with two decimals. Fix billing.py.",
    {
        "billing.py": r'''
def split_bill(total, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    cents = int(round(float(total) * 100))
    share = cents // n
    return ["%d.%02d" % (share // 100, share % 100)] * n
''',
        "test_billing.py": r'''
from billing import split_bill


def test_three_way():
    assert split_bill("100.00", 3) == ["33.34", "33.33", "33.33"]
''',
    },
    {
        "test_hidden.py": r'''
import pytest
from decimal import Decimal
from billing import split_bill


def test_tiny_amount():
    assert split_bill("0.01", 3) == ["0.01", "0.00", "0.00"]


def test_even_split_and_single():
    assert split_bill("10.00", 4) == ["2.50"] * 4
    assert split_bill("19.99", 1) == ["19.99"]


def test_sum_invariant():
    for total, n in [("0.10", 7), ("1234.56", 11), ("99.99", 2), ("5.00", 6)]:
        shares = split_bill(total, n)
        assert sum(Decimal(s) for s in shares) == Decimal(total)
        assert max(Decimal(s) for s in shares) - min(Decimal(s) for s in shares) <= Decimal("0.01")


def test_float_artifacts():
    assert split_bill("1.15", 1) == ["1.15"]
    assert split_bill("8.20", 2) == ["4.10", "4.10"]


def test_invalid_n():
    with pytest.raises(ValueError):
        split_bill("1.00", 0)
''',
    },
    {
        "billing.py": r'''
from decimal import Decimal


def split_bill(total, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    cents = int((Decimal(total) * 100).to_integral_value())
    share, extra = divmod(cents, n)
    out = []
    for i in range(n):
        c = share + (1 if i < extra else 0)
        out.append("%d.%02d" % (c // 100, c % 100))
    return out
''',
    },
)

# 05 ---------------------------------------------------------------------------------------------
task(
    "t05_lower_bound",
    "`lower_bound(a, x)` must return the first index i with a[i] >= x (len(a) if there is none) for a sorted list. "
    "It returns the wrong index in some cases. Fix search.py.",
    {
        "search.py": r'''
def lower_bound(a, x):
    lo, hi = 0, len(a) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        "test_search.py": r'''
from search import lower_bound


def test_basic():
    assert lower_bound([1, 3, 5, 7], 5) == 2
    assert lower_bound([1, 3, 5, 7], 4) == 2
    assert lower_bound([1, 3, 5, 7], 9) == 4
''',
    },
    {
        "test_hidden.py": r'''
import bisect, random
from search import lower_bound


def test_edges():
    assert lower_bound([], 5) == 0
    assert lower_bound([1, 2, 3], 10) == 3
    assert lower_bound([1, 2, 3], -1) == 0
    assert lower_bound([2], 2) == 0
    assert lower_bound([2], 3) == 1


def test_duplicates_first_index():
    assert lower_bound([1, 2, 2, 2, 3], 2) == 1


def test_matches_bisect_randomized():
    rnd = random.Random(7)
    for _ in range(300):
        a = sorted(rnd.randint(0, 20) for _ in range(rnd.randint(0, 15)))
        x = rnd.randint(-2, 22)
        assert lower_bound(a, x) == bisect.bisect_left(a, x)
''',
    },
    {
        "search.py": r'''
def lower_bound(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
    },
)

# 06 ---------------------------------------------------------------------------------------------
task(
    "t06_csv_escaped_quotes",
    "`parse_line` must follow RFC 4180 quoting: inside a quoted field a doubled quote (\"\") is a literal quote. "
    "Right now escaped quotes break parsing. Fix csvparse.py.",
    {
        "csvparse.py": r'''
def parse_line(line):
    fields, cur, in_q = [], [], False
    for ch in line:
        if ch == '"':
            in_q = not in_q
        elif ch == "," and not in_q:
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    fields.append("".join(cur))
    return fields
''',
        "test_csvparse.py": r'''
from csvparse import parse_line


def test_escaped_quotes():
    assert parse_line('a,"he said ""hi""",c') == ["a", 'he said "hi"', "c"]
''',
    },
    {
        "test_hidden.py": r'''
from csvparse import parse_line


def test_plain_and_embedded_comma():
    assert parse_line("a,b,c") == ["a", "b", "c"]
    assert parse_line('x,"y,z",w') == ["x", "y,z", "w"]


def test_empty_and_trailing():
    assert parse_line("") == [""]
    assert parse_line("a,,b,") == ["a", "", "b", ""]
    assert parse_line('""') == [""]


def test_only_escaped_quote_field():
    assert parse_line('"""x"""') == ['"x"']
    assert parse_line('"a""b"') == ['a"b']

''',
    },
    {
        "csvparse.py": r'''
def parse_line(line):
    fields, cur, in_q = [], [], False
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if in_q:
            if ch == '"':
                if i + 1 < n and line[i + 1] == '"':
                    cur.append('"')
                    i += 1
                else:
                    in_q = False
            else:
                cur.append(ch)
        else:
            if ch == '"':
                in_q = True
            elif ch == ",":
                fields.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
        i += 1
    fields.append("".join(cur))
    return fields
''',
    },
)

# 07 ---------------------------------------------------------------------------------------------
task(
    "t07_toposort_cycles",
    "`toposort(graph)` (graph maps node -> list of nodes it depends on) silently ignores cycles. It must raise "
    "`ValueError` when the graph has a cycle (a self-dependency counts as a cycle) and otherwise return an order "
    "where every node appears after all of its dependencies, each node exactly once. Nodes that only appear as dependencies must be included. The input graph must not be modified.",
    {
        "topo.py": r'''
def toposort(graph):
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for d in graph.get(n, []):
            visit(d)
        order.append(n)

    for node in list(graph):
        visit(node)
    return order
''',
        "test_topo.py": r'''
import pytest
from topo import toposort


def test_cycle_raises():
    with pytest.raises(ValueError):
        toposort({"a": ["b"], "b": ["a"]})


def test_simple_order():
    order = toposort({"app": ["lib"], "lib": ["core"], "core": []})
    assert order.index("core") < order.index("lib") < order.index("app")
''',
    },
    {
        "test_hidden.py": r'''
import pytest
from topo import toposort


def check(graph, order):
    pos = {n: i for i, n in enumerate(order)}
    assert len(pos) == len(order), "duplicates in order"
    for n, deps in graph.items():
        for d in deps:
            assert pos[d] < pos[n]


def test_self_loop():
    with pytest.raises(ValueError):
        toposort({"a": ["a"]})


def test_long_cycle():
    with pytest.raises(ValueError):
        toposort({"a": ["b"], "b": ["c"], "c": ["d"], "d": ["b"]})


def test_implicit_nodes_and_diamond():
    g = {"d": ["b", "c"], "b": ["a"], "c": ["a"]}
    order = toposort(g)
    assert set(order) == {"a", "b", "c", "d"}
    check(g, order)


def test_disconnected_and_empty():
    g = {"x": [], "y": ["z"], "w": []}
    order = toposort(g)
    assert set(order) == {"x", "y", "z", "w"}
    check(g, order)
    assert toposort({}) == []


def test_graph_not_mutated():
    g = {"a": ["b"]}
    toposort(g)
    assert g == {"a": ["b"]}
''',
    },
    {
        "topo.py": r'''
def toposort(graph):
    order, state = [], {}

    def visit(n):
        s = state.get(n)
        if s == 2:
            return
        if s == 1:
            raise ValueError("cycle detected at %r" % (n,))
        state[n] = 1
        for d in graph.get(n, []):
            visit(d)
        state[n] = 2
        order.append(n)

    for node in list(graph):
        visit(node)
    return order
''',
    },
)

# 08 ---------------------------------------------------------------------------------------------
task(
    "t08_token_bucket_cap",
    "The token-bucket `RateLimiter` lets bursts exceed its capacity after an idle period (tokens are never capped). "
    "Tokens must never exceed `capacity`. Fix ratelimit.py.",
    {
        "ratelimit.py": r'''
class RateLimiter:
    """Token bucket: `rate` tokens/second, holding at most `capacity` tokens. `clock` returns seconds."""

    def __init__(self, rate, capacity, clock):
        self.rate = rate
        self.capacity = capacity
        self.clock = clock
        self.tokens = capacity
        self.last = clock()

    def allow(self, cost=1):
        now = self.clock()
        self.tokens += (now - self.last) * self.rate
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False
''',
        "test_ratelimit.py": r'''
from ratelimit import RateLimiter


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_burst_capped_after_idle():
    clk = Clock()
    rl = RateLimiter(rate=1, capacity=3, clock=clk)
    assert [rl.allow() for _ in range(3)] == [True, True, True]
    clk.t = 1000.0
    assert [rl.allow() for _ in range(4)] == [True, True, True, False]
''',
    },
    {
        "test_hidden.py": r'''
import pytest
from ratelimit import RateLimiter


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_refill_partial():
    clk = Clock()
    rl = RateLimiter(rate=2, capacity=4, clock=clk)
    for _ in range(4):
        assert rl.allow()
    assert not rl.allow()
    clk.t = 0.5
    assert rl.allow() and not rl.allow()


def test_cost_greater_than_one():
    clk = Clock()
    rl = RateLimiter(rate=1, capacity=5, clock=clk)
    assert rl.allow(cost=5)
    assert not rl.allow(cost=1)
    clk.t = 3.0
    assert rl.allow(cost=3) and not rl.allow()


def test_cost_over_capacity_never_allowed():
    clk = Clock()
    rl = RateLimiter(rate=10, capacity=2, clock=clk)
    clk.t = 100.0
    assert not rl.allow(cost=3)
    assert rl.allow(cost=2)


def test_cap_applies_between_calls():
    clk = Clock()
    rl = RateLimiter(rate=1, capacity=2, clock=clk)
    clk.t = 10.0
    assert rl.allow(); clk.t = 10.0
    assert rl.allow() and not rl.allow()
''',
    },
    {
        "ratelimit.py": r'''
class RateLimiter:
    """Token bucket: `rate` tokens/second, holding at most `capacity` tokens. `clock` returns seconds."""

    def __init__(self, rate, capacity, clock):
        self.rate = rate
        self.capacity = capacity
        self.clock = clock
        self.tokens = capacity
        self.last = clock()

    def allow(self, cost=1):
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False
''',
    },
)

# 09 ---------------------------------------------------------------------------------------------
task(
    "t09_deep_merge",
    "`deep_merge(a, b)` should recursively merge nested dicts (values from b win; lists and scalars from b replace "
    "a's), return a NEW dict and never mutate either argument. Currently nested dicts are overwritten and `a` is mutated. "
    "Fix config.py.",
    {
        "config.py": r'''
def deep_merge(a, b):
    a.update(b)
    return a
''',
        "test_config.py": r'''
from config import deep_merge


def test_nested_merge():
    assert deep_merge({"db": {"host": "x", "port": 1}}, {"db": {"port": 2}}) == {"db": {"host": "x", "port": 2}}
''',
    },
    {
        "test_hidden.py": r'''
import copy
from config import deep_merge


def test_no_mutation_and_new_object():
    a = {"k": {"x": 1}, "l": [1]}
    b = {"k": {"y": 2}, "l": [2, 3]}
    a0, b0 = copy.deepcopy(a), copy.deepcopy(b)
    out = deep_merge(a, b)
    assert a == a0 and b == b0
    assert out == {"k": {"x": 1, "y": 2}, "l": [2, 3]}
    out["k"]["x"] = 99
    assert a["k"]["x"] == 1


def test_type_conflicts_b_wins():
    assert deep_merge({"a": {"x": 1}}, {"a": 5}) == {"a": 5}
    assert deep_merge({"a": 5}, {"a": {"x": 1}}) == {"a": {"x": 1}}


def test_empty_and_deeper():
    assert deep_merge({}, {}) == {}
    assert deep_merge({"a": {"b": {"c": 1}}}, {"a": {"b": {"d": 2}}}) == {"a": {"b": {"c": 1, "d": 2}}}
    assert deep_merge({"a": 1}, {}) == {"a": 1}
''',
    },
    {
        "config.py": r'''
import copy


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
''',
    },
)

# 10 ---------------------------------------------------------------------------------------------
task(
    "t10_slugify",
    "`slugify` must: lowercase, fold accents to ASCII (é -> e), turn every run of non-alphanumeric characters into a "
    "single '-', and strip leading/trailing '-'. It currently leaves repeated and edge dashes. Fix slug.py.",
    {
        "slug.py": r'''
import re


def slugify(text):
    return re.sub(r"[^a-z0-9]", "-", text.lower())
''',
        "test_slug.py": r'''
from slug import slugify


def test_collapse_and_strip():
    assert slugify("  Hello,   World!  ") == "hello-world"
''',
    },
    {
        "test_hidden.py": r'''
from slug import slugify


def test_accents():
    assert slugify("Héllo Wörld") == "hello-world"
    assert slugify("Crème Brûlée") == "creme-brulee"


def test_underscores_dashes_and_digits():
    assert slugify("  --a__b  ") == "a-b"
    assert slugify("Python 3.12 rocks") == "python-3-12-rocks"


def test_empty_and_symbols_only():
    assert slugify("") == ""
    assert slugify("!!!") == ""
''',
    },
    {
        "slug.py": r'''
import re
import unicodedata


def slugify(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")
''',
    },
)

# 11 ---------------------------------------------------------------------------------------------
task(
    "t11_async_gather_limit",
    "`fetch_all(items, fetch, limit)` must (1) return results in input order, (2) NOT abort when one fetch raises: "
    "that slot must hold the exception object, the others still complete, and (3) never run more than `limit` fetches "
    "concurrently. Right now one failure aborts everything and the limit is ignored. Fix fetcher.py.",
    {
        "fetcher.py": r'''
import asyncio


async def fetch_all(items, fetch, limit):
    return await asyncio.gather(*(fetch(i) for i in items))
''',
        "test_fetcher.py": r'''
import asyncio
from fetcher import fetch_all


def test_exceptions_are_collected():
    async def fetch(i):
        if i == 2:
            raise RuntimeError("boom")
        return i * 10

    out = asyncio.run(fetch_all([1, 2, 3], fetch, 2))
    assert out[0] == 10 and out[2] == 30
    assert isinstance(out[1], RuntimeError)
''',
    },
    {
        "test_hidden.py": r'''
import asyncio
from fetcher import fetch_all


def test_concurrency_limit_respected():
    state = {"cur": 0, "max": 0}

    async def fetch(i):
        state["cur"] += 1
        state["max"] = max(state["max"], state["cur"])
        await asyncio.sleep(0.01)
        state["cur"] -= 1
        return i

    out = asyncio.run(fetch_all(list(range(12)), fetch, 3))
    assert out == list(range(12))
    assert state["max"] <= 3


def test_order_preserved_with_varying_delays():
    async def fetch(i):
        await asyncio.sleep(0.03 if i == 0 else 0.0)
        return i

    assert asyncio.run(fetch_all([0, 1, 2, 3], fetch, 4)) == [0, 1, 2, 3]


def test_empty_and_limit_one():
    async def fetch(i):
        return i

    assert asyncio.run(fetch_all([], fetch, 1)) == []
    assert asyncio.run(fetch_all([5, 6], fetch, 1)) == [5, 6]


def test_all_fail():
    async def fetch(i):
        raise ValueError(i)

    out = asyncio.run(fetch_all([1, 2], fetch, 2))
    assert all(isinstance(x, ValueError) for x in out)
''',
    },
    {
        "fetcher.py": r'''
import asyncio


async def fetch_all(items, fetch, limit):
    sem = asyncio.Semaphore(limit)

    async def run(i):
        async with sem:
            return await fetch(i)

    return await asyncio.gather(*(run(i) for i in items), return_exceptions=True)
''',
    },
)

# 12 ---------------------------------------------------------------------------------------------
task(
    "t12_thread_safe_counter",
    "`Counter` loses updates when used from several threads. `increment`, `decrement` and `reset` must be thread-safe "
    "(no lost updates). Fix counter.py.",
    {
        "counter.py": r'''
import time


class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        tmp = self.value
        time.sleep(0)  # yields the GIL, exposing the race
        self.value = tmp + 1

    def decrement(self):
        tmp = self.value
        time.sleep(0)
        self.value = tmp - 1

    def reset(self):
        self.value = 0
''',
        "test_counter.py": r'''
import threading
from counter import Counter


def test_no_lost_updates():
    c = Counter()
    ts = [threading.Thread(target=lambda: [c.increment() for _ in range(500)]) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert c.value == 4000
''',
    },
    {
        "test_hidden.py": r'''
import threading
from counter import Counter


def run(c, fn, n_threads=8, n=400):
    ts = [threading.Thread(target=lambda: [fn() for _ in range(n)]) for _ in range(n_threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]


def test_mixed_inc_dec_balance():
    c = Counter()
    ts = []
    for _ in range(4):
        ts.append(threading.Thread(target=lambda: [c.increment() for _ in range(500)]))
        ts.append(threading.Thread(target=lambda: [c.decrement() for _ in range(500)]))
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert c.value == 0


def test_value_still_a_plain_int_attribute_and_reset():
    c = Counter()
    run(c, c.increment, 4, 100)
    assert c.value == 400
    c.reset()
    assert c.value == 0
''',
    },
    {
        "counter.py": r'''
import threading
import time


class Counter:
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            tmp = self.value
            time.sleep(0)
            self.value = tmp + 1

    def decrement(self):
        with self._lock:
            tmp = self.value
            time.sleep(0)
            self.value = tmp - 1

    def reset(self):
        with self._lock:
            self.value = 0
''',
    },
)

# 13 ---------------------------------------------------------------------------------------------
task(
    "t13_semver_compare",
    "`compare(a, b)` must follow Semantic Versioning precedence and return -1, 0 or 1: compare major, minor, patch "
    "numerically ('1.10.0' > '1.9.0'); a version WITH a pre-release ('1.0.0-alpha') is lower than the same without; "
    "pre-release identifiers are compared left to right, numeric identifiers numerically and lower than alphanumeric ones; "
    "a shorter identifier list is lower when all preceding ones are equal. Build metadata ('+...') is ignored. Fix semver.py.",
    {
        "semver.py": r'''
def compare(a, b):
    return (a > b) - (a < b)
''',
        "test_semver.py": r'''
from semver import compare


def test_numeric_not_lexicographic():
    assert compare("1.10.0", "1.9.0") == 1
    assert compare("2.0.0", "10.0.0") == -1
''',
    },
    {
        "test_hidden.py": r'''
from semver import compare


def test_prerelease_lower_than_release():
    assert compare("1.0.0-alpha", "1.0.0") == -1
    assert compare("1.0.0", "1.0.0-rc.1") == 1


def test_spec_ordering_chain():
    chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
    for i in range(len(chain)):
        for j in range(len(chain)):
            assert compare(chain[i], chain[j]) == (i > j) - (i < j), (chain[i], chain[j])


def test_equal_and_build_metadata():
    assert compare("1.2.3", "1.2.3") == 0
    assert compare("1.2.3+build.5", "1.2.3+build.9") == 0
    assert compare("1.2.3+x", "1.2.4") == -1
''',
    },
    {
        "semver.py": r'''
def _parse(v):
    v = v.split("+", 1)[0]
    core, _, pre = v.partition("-")
    nums = tuple(int(x) for x in core.split("."))
    ids = pre.split(".") if pre else []
    return nums, ids


def _cmp_id(x, y):
    xn, yn = x.isdigit(), y.isdigit()
    if xn and yn:
        return (int(x) > int(y)) - (int(x) < int(y))
    if xn:
        return -1
    if yn:
        return 1
    return (x > y) - (x < y)


def compare(a, b):
    (na, ia), (nb, ib) = _parse(a), _parse(b)
    if na != nb:
        return (na > nb) - (na < nb)
    if not ia and not ib:
        return 0
    if not ia:
        return 1
    if not ib:
        return -1
    for x, y in zip(ia, ib):
        c = _cmp_id(x, y)
        if c:
            return c
    return (len(ia) > len(ib)) - (len(ia) < len(ib))
''',
    },
)

# 14 ---------------------------------------------------------------------------------------------
task(
    "t14_normalize_path",
    "`normalize(path)` must behave like `posixpath.normpath` for both absolute and relative paths: collapse repeated "
    "slashes, drop '.', resolve '..' (never above root for absolute paths, kept as leading '..' for relative ones), "
    "return '.' for an empty result, and have no trailing slash (except root '/'). It is currently wrong. Fix paths.py "
    "WITHOUT importing posixpath/os.path.",
    {
        "paths.py": r'''
def normalize(path):
    parts = []
    for p in path.split("/"):
        if p == "" or p == ".":
            continue
        if p == "..":
            parts.pop()
        else:
            parts.append(p)
    return "/" + "/".join(parts)
''',
        "test_paths.py": r'''
from paths import normalize


def test_basics():
    assert normalize("/a/b/../c//d/.") == "/a/c/d"
    assert normalize("a/./b/../../c") == "c"
''',
    },
    {
        "test_hidden.py": r'''
import posixpath
from paths import normalize

CASES = ["/", "///a", ".", "", "..", "../..", "a/..", "a/../..", "/..", "/../a", "/a/../../b",
         "a//b///c/", "./a", "../a/./b/..", "/a/b/c/../../../..", "x/y/../../..", "..//..", "a/b/./././c/"]


def test_matches_posixpath():
    for c in CASES:
        assert normalize(c) == posixpath.normpath(c), c


def test_does_not_import_posixpath():
    import ast, paths
    tree = ast.parse(open(paths.__file__).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] in ("posixpath", "os", "pathlib") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in ("posixpath", "os", "pathlib")
''',
    },
    {
        "paths.py": r'''
def normalize(path):
    absolute = path.startswith("/")
    parts = []
    for p in path.split("/"):
        if p == "" or p == ".":
            continue
        if p == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            elif not absolute:
                parts.append("..")
        else:
            parts.append(p)
    joined = "/".join(parts)
    if absolute:
        return "/" + joined
    return joined or "."
''',
    },
)

# 15 ---------------------------------------------------------------------------------------------
task(
    "t15_cross_module_discount",
    "Cart totals are wrong: items stored with a 15% discount are only reduced by 0.15%. The bug involves how the two "
    "modules interact. `pricing.apply_discount(price, pct)` takes a PERCENT (15 means 15%) and must keep that contract; "
    "`inventory.Item.discount` is stored as a FRACTION (0.15). Fix the project so totals are correct.",
    {
        "pricing.py": r'''
def apply_discount(price, pct):
    """Return price reduced by pct percent (pct=15 means 15%)."""
    return round(price * (1 - pct / 100), 2)
''',
        "inventory.py": r'''
from pricing import apply_discount


class Item:
    def __init__(self, name, price, discount=0.0):
        self.name = name
        self.price = price
        self.discount = discount  # fraction, e.g. 0.15 == 15%


def cart_total(items):
    return round(sum(apply_discount(i.price, i.discount) for i in items), 2)
''',
        "test_inventory.py": r'''
from inventory import Item, cart_total


def test_discounted_total():
    assert cart_total([Item("a", 100.0, 0.15), Item("b", 50.0)]) == 135.0
''',
    },
    {
        "test_hidden.py": r'''
from inventory import Item, cart_total
from pricing import apply_discount


def test_pricing_contract_unchanged():
    assert apply_discount(200.0, 25) == 150.0
    assert apply_discount(80.0, 0) == 80.0


def test_totals():
    assert cart_total([]) == 0
    assert cart_total([Item("x", 20.0, 0.5)]) == 10.0
    assert cart_total([Item("x", 10.0, 1.0), Item("y", 10.0, 0.1)]) == 9.0
    assert cart_total([Item("x", 33.33, 0.333)]) == round(33.33 * (1 - 33.3 / 100), 2)
''',
    },
    {
        "inventory.py": r'''
from pricing import apply_discount


class Item:
    def __init__(self, name, price, discount=0.0):
        self.name = name
        self.price = price
        self.discount = discount  # fraction, e.g. 0.15 == 15%


def cart_total(items):
    return round(sum(apply_discount(i.price, i.discount * 100) for i in items), 2)
''',
    },
)

# 16 ---------------------------------------------------------------------------------------------
task(
    "t16_add_business_days",
    "`add_business_days(d, n)` (n >= 0) must return the date that is n business days (Mon-Fri) after `d`. "
    "For n == 0 return `d` unchanged. For n > 0 weekends are skipped; if `d` is a Saturday or Sunday, counting starts as if "
    "`d` were the preceding Friday. The current implementation just adds calendar days. Fix bizdays.py.",
    {
        "bizdays.py": r'''
from datetime import date, timedelta


def add_business_days(d, n):
    return d + timedelta(days=n)
''',
        "test_bizdays.py": r'''
from datetime import date
from bizdays import add_business_days


def test_over_weekend():
    assert add_business_days(date(2024, 5, 3), 1) == date(2024, 5, 6)  # Fri +1 -> Mon
''',
    },
    {
        "test_hidden.py": r'''
from datetime import date, timedelta
from bizdays import add_business_days


def oracle(d, n):
    if n == 0:
        return d
    cur = d
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)  # weekend start behaves like the preceding Friday
    steps = 0
    while steps < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            steps += 1
    return cur


def test_matches_oracle_for_two_years_of_starts():
    d = date(2024, 1, 1)
    for i in range(0, 740, 3):
        for n in (0, 1, 2, 4, 5, 6, 9, 10, 23):
            s = d + timedelta(days=i)
            assert add_business_days(s, n) == oracle(s, n), (s, n)


def test_zero_unchanged_even_on_weekend():
    assert add_business_days(date(2024, 5, 4), 0) == date(2024, 5, 4)


def test_weekend_start():
    assert add_business_days(date(2024, 5, 4), 1) == date(2024, 5, 6)  # Sat +1 -> Mon (as if Friday)
    assert add_business_days(date(2024, 5, 5), 2) == date(2024, 5, 7)  # Sun +2 -> Tue
''',
    },
    {
        "bizdays.py": r'''
from datetime import date, timedelta


def add_business_days(d, n):
    if n == 0:
        return d
    cur = d
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    steps = 0
    while steps < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            steps += 1
    return cur
''',
    },
)
