import threading
import time

import numpy as np
import pytest

from tablely import stream
from tablely.stream import Backend, ChunkPlan, ChunkStream, HostBackend


def test_l3_size_is_read_from_sysfs(tmp_path):
    for i, (level, kind, size) in enumerate([("1", "Data", "32K"), ("2", "Unified", "1024K"), ("3", "Unified", "33792K")]):
        d = tmp_path / "cpu0" / "cache" / f"index{i}"
        d.mkdir(parents=True)
        (d / "level").write_text(level)
        (d / "type").write_text(kind)
        (d / "size").write_text(size)
    assert stream.l3_cache_bytes(tmp_path) == 33792 * 1024
    assert stream.l3_cache_bytes(tmp_path / "nothing") > 0  # falls back


def test_cpu_chunks_follow_the_l3_cache(monkeypatch):
    monkeypatch.setattr(stream, "l3_cache_bytes", lambda: 8 * stream.MiB)
    data = np.zeros((100_000, 256), dtype=np.float32)  # 1 KiB rows
    plan = stream.plan_chunks(data, "cpu")
    assert plan.rows == 4096  # half of 8 MiB
    assert plan.count == 25 and plan.slices()[-1] == slice(98304, 100_000)
    shared = stream.plan_chunks(data, "cpu", sharing=4)
    assert shared.rows == 1024  # four workers share one L3


def test_chunk_size_is_capped_by_the_ram_budget():
    data = np.zeros((1000, 1024), dtype=np.uint8)
    plan = stream.plan_chunks(data, "cuda:0", buffers=4, ram_budget=8 * 1024)
    assert plan.rows == 2  # 4 staging buffers of 2 KiB each
    huge_rows = np.zeros((3, 1 << 20), dtype=np.uint8)
    assert stream.plan_chunks(huge_rows, "cpu", chunk_bytes=10).rows == 1  # never below one row


def test_memory_mapped_file_streams_in_order_and_is_read_ahead(tmp_path):
    path = tmp_path / "big.npy"
    original = np.arange(200_000 * 16, dtype=np.float32).reshape(200_000, 16)
    np.save(path, original)
    data = stream.open_array(path)
    assert isinstance(data, np.memmap)

    s = stream.chunks(data, "cpu", chunk_bytes=256 * 1024)
    parts = list(s)
    assert len(parts) == s.plan.count == 49  # 64-byte rows, 4096 per 256 KiB chunk
    assert all(type(p) is np.ndarray for p in parts)  # copied into RAM by the reader, not lazy views
    assert np.array_equal(np.concatenate(parts), original)
    assert s.stats.chunks == 49 and s.stats.bytes == original.nbytes


def test_in_ram_arrays_are_not_copied():
    data = np.arange(1000, dtype=np.int64).reshape(100, 10)
    first = next(iter(stream.chunks(data, "cpu", chunk_bytes=400)))
    assert np.shares_memory(first, data)


class Recorder(Backend):
    """Stands in for the GPU path: records the order of stage/send/receive."""

    def __init__(self, stage_delay=0.0):
        self.log, self.stage_delay, self.lock = [], stage_delay, threading.Lock()
        self.in_flight = set()

    def stage(self, slot, rows):
        with self.lock:
            assert slot not in self.in_flight, "slot refilled before its transfer was sent"
            self.in_flight.add(slot)
            self.log.append(("stage", int(rows[0, 0])))
        time.sleep(self.stage_delay)
        return np.array(rows)

    def send(self, slot, staged):
        with self.lock:
            self.in_flight.discard(slot)
            self.log.append(("send", int(staged[0, 0])))
        return staged

    def receive(self, handle):
        self.log.append(("use", int(handle[0, 0])))
        return handle


def plan_for(data, rows, buffers=2):
    n = data.shape[0]
    return ChunkPlan(rows=rows, chunk_bytes=rows * data[0].nbytes, total_rows=n,
                     count=-(-n // rows), buffers=buffers)


def test_next_chunk_is_sent_before_the_current_one_is_used():
    data = np.arange(60).reshape(60, 1)
    rec = Recorder()
    out = list(ChunkStream(data, device="fake", plan=plan_for(data, 10), backend=rec))
    assert np.array_equal(np.concatenate(out), data)
    events = rec.log
    for i in range(0, 50, 10):  # chunk i+10 is on its way before chunk i is handed out
        assert events.index(("send", i + 10)) < events.index(("use", i))


def test_reading_overlaps_with_compute():
    data = np.arange(10).reshape(10, 1)
    delay = 0.05
    started = time.perf_counter()
    for _ in ChunkStream(data, device="fake", plan=plan_for(data, 1, buffers=3), backend=Recorder(delay)):
        time.sleep(delay)  # "compute"
    elapsed = time.perf_counter() - started
    assert elapsed < 0.8 * (2 * delay * 10)  # serial would take 1.0s


def test_reader_errors_reach_the_consumer():
    class Broken(HostBackend):
        def stage(self, slot, rows):
            if rows[0, 0] >= 20:
                raise OSError("disk went away")
            return rows

    data = np.arange(50).reshape(50, 1)
    s = ChunkStream(data, plan=plan_for(data, 10), backend=Broken(read_ahead=False))
    with pytest.raises(OSError, match="disk went away"):
        list(s)


def test_breaking_out_early_stops_the_reader():
    data = np.arange(1000).reshape(1000, 1)
    s = ChunkStream(data, plan=plan_for(data, 1), backend=HostBackend(read_ahead=False))
    for i, _ in enumerate(s):
        if i == 3:
            break
    assert s._thread is not None and not s._thread.is_alive()


def test_map_chunks_on_cpu_keeps_order_and_covers_everything(tmp_path):
    path = tmp_path / "x.npy"
    original = np.random.default_rng(0).random((50_000, 8))
    np.save(path, original)
    data = stream.open_array(path)
    sums = stream.map_chunks(lambda x: (x[0, 0], x.sum()), data, "cpu", chunk_bytes=64 * 1024, workers=4)
    assert len(sums) == stream.plan_chunks(data, "cpu", chunk_bytes=64 * 1024).count
    firsts = [first for first, _ in sums]
    assert firsts == [original[i * 1024, 0] for i in range(len(sums))]  # chunk order
    assert np.isclose(sum(total for _, total in sums), original.sum())


def test_preload_brings_small_files_into_ram(tmp_path):
    path = tmp_path / "small.npy"
    np.save(path, np.ones((100, 4)))
    mapped = stream.open_array(path)
    loaded = stream.preload(mapped)
    assert not isinstance(loaded, np.memmap) and np.array_equal(loaded, mapped)
    with pytest.warns(UserWarning, match="does not fit"):
        assert stream.preload(mapped, ram_fraction=1e-12) is mapped
    in_ram = np.zeros(3)
    assert stream.preload(in_ram) is in_ram


def test_raw_files_need_dtype_and_shape(tmp_path):
    path = tmp_path / "raw.bin"
    np.arange(12, dtype=np.int16).tofile(path)
    assert stream.open_array(path, dtype=np.int16, shape=(3, 4))[2, 3] == 11
    with pytest.raises(ValueError):
        stream.open_array(path, dtype=np.int16)


def test_device_resolution_follows_tablely(monkeypatch):
    monkeypatch.setenv("TABLELY_DEVICE", "cpu")
    assert stream.resolve_devices("auto") == ["cpu"]
    monkeypatch.setattr(stream, "_torch", lambda: None)
    monkeypatch.setenv("TABLELY_DEVICE", "cuda")
    with pytest.warns(UserWarning, match="cannot use CUDA"):
        assert stream.resolve_devices("auto") == ["cpu"]  # falls back to CPU compute
    with pytest.raises(RuntimeError, match="needs PyTorch"):
        stream.resolve_devices("cuda")
    assert stream.resolve_devices(["cpu"]) == ["cpu"]
