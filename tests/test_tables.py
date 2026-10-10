import json
import os
import sys
import textwrap
import weakref

import numpy as np
import pytest

from tablely import tables
from tablely.ledger import Ledger, process_identity
from tablely.tables import Arena, NumpyBackend, Store

MiB = 1024 ** 2


class FakeOOM(RuntimeError):
    pass


class Dev(np.ndarray):
    """An array 'on the fake GPU'."""


class FakeGpu(NumpyBackend):
    """A GPU with ``capacity`` bytes: allocations past it raise like CUDA does."""

    gpu = True
    device = "fake"

    def __init__(self, capacity):
        self.capacity = capacity
        self.used = 0
        self.bus_bytes = 0  # RAM -> GPU traffic

    def alloc(self, nbytes):
        if self.used + nbytes > self.capacity:
            raise FakeOOM("CUDA out of memory (fake)")
        array = np.zeros(nbytes, dtype=np.uint8).view(Dev)
        self.used += nbytes
        weakref.finalize(array, self._release, nbytes)
        return array

    def _release(self, nbytes):
        self.used -= nbytes

    def upload(self, data, shape, dtype):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        table = self.alloc(nbytes).view(dtype).reshape(shape).view(Dev)
        if data is not None:
            table[...] = np.asarray(data)
            self.bus_bytes += nbytes
        return table

    def download(self, device_table, host):
        host[...] = device_table

    def take(self, table, ids, on_gpu):
        rows = super().take(table, ids, on_gpu)
        if not on_gpu:
            self.bus_bytes += rows.nbytes
        return rows

    def is_oom(self, exc):
        return isinstance(exc, FakeOOM)

    def vram(self):
        return self.capacity - self.used, self.capacity


def emb(rows, dim=16, seed=0):
    return np.random.default_rng(seed).standard_normal((rows, dim)).astype(np.float32)


def test_arena_first_fit_free_and_merge():
    arena = Arena(4096)
    a, b, c = arena.alloc(1000), arena.alloc(1), arena.alloc(1024)
    assert (a, b, c) == (0, 1024, 1280)  # 256-byte aligned
    assert arena.used == 1024 + 256 + 1024 and arena.alloc(4096) is None
    arena.free(b, 1)
    arena.free(a, 1000)
    assert arena.alloc(1280) == 0  # the two neighbours merged into one block
    arena.free(0, 1280)
    arena.free(c, 1024)
    assert arena.used == 0 and arena.largest_free == 4096


def test_ram_tables_live_in_the_reserved_ram():
    store = Store(ram=1 * MiB, backend=NumpyBackend(), report=False)
    weights = emb(1000)
    table = store.put("item", weights)
    assert table.where == "ram" and store.ram_used == weights.nbytes
    assert np.shares_memory(table.data, store._buffer)  # inside the block set aside up front
    np.testing.assert_array_equal(table.gather([3, 1, 3]), weights[[3, 1, 3]])
    np.testing.assert_array_equal(table.rows(10, 12), weights[10:12])

    table.add_rows([1, 1], np.ones((2, 16), np.float32))  # repeated ids add up
    np.testing.assert_allclose(table.gather([1])[0], weights[1] + 2, rtol=1e-6)
    table.set_rows([0], np.zeros((1, 16)))
    assert not table.numpy()[0].any()

    with pytest.raises(MemoryError, match="reserved for this job"):
        store.put("too-big", emb(20000))
    store.drop("item")
    assert store.arena.used == 0 and store.put("fits-now", emb(16384)).where == "ram"


def test_tables_fill_the_vram_budget_then_go_to_ram():
    gpu = FakeGpu(capacity=8 * MiB)
    store = Store(ram=8 * MiB, vram=1 * MiB, backend=gpu, report=False)
    small = store.put("small", emb(8192))  # 512 KiB
    big = store.put("big", emb(16384))  # 1 MiB: over the budget
    pinned = store.put("pinned", emb(100), where="ram")
    assert (small.where, big.where, pinned.where) == ("gpu", "ram", "ram")
    assert store.vram_used == small.nbytes and gpu.used == small.nbytes

    gpu.bus_bytes = 0
    np.testing.assert_array_equal(big.gather([5, 9]), emb(16384)[[5, 9]])
    assert gpu.bus_bytes == 2 * 16 * 4  # only the two rows crossed the bus
    small.gather([1])
    assert gpu.bus_bytes == 2 * 16 * 4  # GPU tables need no transfer


def test_failed_upload_falls_back_to_ram():
    gpu = FakeGpu(capacity=256 * 1024)
    store = Store(vram=10 * MiB, backend=gpu, report=False)  # budget says yes, the GPU says no
    table = store.put("t", emb(8192))
    assert table.where == "ram" and gpu.used == 0
    with pytest.raises(FakeOOM):
        store.put("forced", emb(8192), where="gpu")


def train_step(gpu, activations, log):
    """A step that needs ``activations`` bytes of GPU memory while it runs."""
    scratch = gpu.alloc(activations)
    log.append(len(scratch))
    return "ok"


def test_guard_moves_tables_to_ram_until_the_step_fits():
    gpu = FakeGpu(capacity=4 * MiB)
    store = Store(ram=8 * MiB, vram=4 * MiB, backend=gpu, report=False)
    a = store.put("a", emb(16384))  # 1 MiB
    b = store.put("b", emb(32768))  # 2 MiB
    keep = store.put("keep", emb(8192), keep_on_gpu=True)  # 0.5 MiB
    log = []
    step = store.guard(train_step)

    assert step(gpu, MiB // 4, log) == "ok" and store.spills == 0  # fits as is
    assert step(gpu, 2 * MiB, log) == "ok"
    assert (b.where, a.where, store.spills) == ("ram", "gpu", 1)  # the biggest table moved first
    np.testing.assert_array_equal(b.gather([7]), emb(32768)[[7]])  # still usable, from RAM

    assert step(gpu, 3 * MiB, log) == "ok"
    assert (a.where, keep.where, store.spills) == ("ram", "gpu", 2)
    with pytest.raises(FakeOOM):  # only keep_on_gpu is left: the error goes through
        step(gpu, 4 * MiB, log)
    assert gpu.used == keep.nbytes


def test_other_errors_are_not_retried():
    store = Store(vram=MiB, backend=FakeGpu(4 * MiB), report=False)
    store.put("a", emb(100))
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("bug")

    with pytest.raises(ValueError):
        store.run(broken)
    assert calls == [1] and store["a"].where == "gpu"


def test_promote_brings_tables_back_when_room_frees_up():
    gpu = FakeGpu(capacity=4 * MiB)
    store = Store(ram=4 * MiB, vram=2 * MiB, backend=gpu, report=False)
    store.put("a", emb(16384))
    b = store.put("b", emb(16384))
    c = store.put("c", emb(16384))
    assert c.where == "ram"
    c.gather([0])
    store.spill("b")
    assert store.promote() == [c]  # most used first; b no longer fits the budget with c back
    assert c.where == "gpu" and b.where == "ram" and store.arena.used == Arena.rounded(b.nbytes)
    np.testing.assert_array_equal(c.numpy(), emb(16384))


def test_without_a_gpu_everything_is_in_ram():
    store = Store(backend=NumpyBackend(), report=False)
    assert store.vram_budget == 0
    assert store.zeros("z", (10, 4)).where == "ram"
    with pytest.raises(ValueError, match="no GPU"):
        store.put("g", emb(4), where="gpu")


def test_memory_mapped_sources_are_copied_piece_by_piece(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "COPY_CHUNK", 4096)
    path = tmp_path / "features.npy"
    np.save(path, emb(1000))
    store = Store(ram=MiB, backend=NumpyBackend(), report=False)
    table = store.put("features", np.load(path, mmap_mode="r"))
    np.testing.assert_array_equal(table.numpy(), emb(1000))


def test_store_reserves_what_tablely_set_aside(monkeypatch):
    monkeypatch.setattr(tables, "_default", None)
    monkeypatch.setattr(tables, "default_backend", lambda device=None: NumpyBackend())
    monkeypatch.setenv("TABLELY_RAM", str(2 * MiB))
    store = tables.store()
    assert store.arena.size == 2 * MiB and tables.store() is store


def test_moves_show_up_on_the_board(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path / "home")
    with ledger.locked() as board:
        board.add_run("r0", {"agent": "me", "task": None, "pid": os.getpid(),
                             "pid_identity": process_identity(os.getpid()), "started_at": 0})
        board.put_job("r0.train", {"run": "r0", "agent": "me", "name": "train", "task": "recsys", "priority": 1,
                                   "device": "gpu", "gpus": 1, "cpus": 1, "max_cpus": None, "ram": 8 * MiB,
                                   "state": "running", "seq": 1, "command": "x",
                                   "allocation": {"device": "gpu", "gpus": ["0"], "cpus": [0]}})
    monkeypatch.setenv("TABLELY_HOME", str(ledger.home))
    monkeypatch.setenv("TABLELY_JOB_KEY", "r0.train")
    store = Store(ram=8 * MiB, vram=2 * MiB, backend=FakeGpu(8 * MiB))
    store.put("a", emb(16384))
    store.put("b", emb(16384))
    store.spill()
    entry = ledger.snapshot()["jobs"]["r0.train"]
    assert entry["tables"] == {"gpu": MiB, "ram": MiB, "reserved": 8 * MiB, "count": 2, "spills": 1}
    events = [e for e in ledger.history() if e["event"] == "tables"]
    assert events and "moved table a (1MiB) from GPU to RAM" in events[-1]["detail"]

    from tablely.board_view import ram_label
    assert ram_label(entry) == "ram 1MiB/8MiB"


def test_a_job_uses_the_ram_tablely_reserved(tmp_path):
    """End to end: Tablely reserves RAM, the job pins it and keeps its table there."""
    import io

    from tablely.resources import Inventory, detect_cpus
    from tablely.runner import Runner
    from tablely.spec import JobSpec

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = tmp_path / "out.json"
    script = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {repo!r})
        import numpy as np
        from tablely import client, tables
        tables.default_backend = lambda device=None: tables.NumpyBackend()  # same path with or without torch
        store = tables.store()
        t = store.put("emb", np.ones((1000, 16), np.float32))
        json.dump({{"ram": client.ram(), "reserved": store.arena.size, "where": t.where,
                   "row": t.gather([3]).tolist()}}, open({str(out)!r}, "w"))
    """)
    runner = Runner(Inventory(cpus=tuple(detect_cpus()), gpus=(), ram=64 * MiB),
                    [JobSpec(name="recsys", command=[sys.executable, "-c", script], device="cpu", ram="4MiB")],
                    log_dir=tmp_path / "logs", out=io.StringIO(), poll_interval=0.02,
                    ledger=Ledger(tmp_path / "home"))
    assert runner.run() == 0, (tmp_path / "logs" / "recsys.log").read_text()
    result = json.loads(out.read_text())
    assert result == {"ram": 4 * MiB, "reserved": 4 * MiB, "where": "ram", "row": [[1.0] * 16]}
