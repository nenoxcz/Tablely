"""Sequential (chunked) upload of data too big for GPU memory, or for RAM.

    from tablely import stream

    data = stream.open_array("features.npy")    # memory-mapped: nothing is read yet
    data = stream.preload(data)                  # into RAM if it fits the budget, else stays on disk

    for x in stream.chunks(data):                # on the GPU Tablely gave this job, or on the CPU
        loss = model(x)

    outs = stream.map_chunks(lambda x: model(x).cpu(), data)   # every assigned GPU at once

How a chunk reaches a GPU::

    disk / RAM --(reader thread)--> pinned RAM buffer --(PCIe DMA, own CUDA stream)--> GPU

A small ring of pinned (page-locked) buffers lets the copy engine DMA straight
from RAM. While the model works on chunk i, chunk i+1 is crossing the bus and
chunk i+2 is being read from disk, so the GPU waits on the bus or the disk only
when those are truly slower than the compute. Chunk size follows a byte budget
(64 MiB by default, capped by free GPU memory and a RAM budget).

On CPU nothing is uploaded: chunks are sized to half the L3 cache so a block
stays cache-resident while it is processed, and a memory-mapped source is read
ahead into RAM by the reader thread. (A cache cannot be targeted by a DMA
transfer, so the GPU path stages in RAM.)

The GPU path needs PyTorch; the CPU path needs only NumPy.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

KiB = 1024
MiB = 1024 * KiB
DEFAULT_GPU_CHUNK = 64 * MiB
MIN_CPU_CHUNK = 256 * KiB
FALLBACK_L3 = 8 * MiB


# -- machine facts ---------------------------------------------------------------


def l3_cache_bytes(sysfs: Union[str, Path] = "/sys/devices/system/cpu") -> int:
    """Size of the last-level (L3) cache shared by this CPU's cores."""
    for index in sorted(Path(sysfs).glob("cpu0/cache/index*")):
        try:
            level = (index / "level").read_text().strip()
            kind = (index / "type").read_text().strip()
            size = (index / "size").read_text().strip()
        except OSError:
            continue
        if level == "3" and kind in ("Unified", "Data"):
            return _parse_size(size)
    try:
        value = os.sysconf("SC_LEVEL3_CACHE_SIZE")
        if value > 0:
            return value
    except (ValueError, OSError, AttributeError):
        pass
    return FALLBACK_L3


def available_ram_bytes() -> int:
    """RAM that can be used without swapping (MemAvailable on Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * KiB
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 4 * 1024 * MiB


def _parse_size(text: str) -> int:
    text = text.strip().upper()
    units = {"K": KiB, "M": MiB, "G": 1024 * MiB}
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text)


# -- sources ---------------------------------------------------------------------


def open_array(
    path: Union[str, Path], dtype: Any = None, shape: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Memory-map a ``.npy`` file (or a raw binary file given ``dtype`` and ``shape``).

    Nothing is read until rows are touched, so the file may be larger than RAM.
    """
    path = Path(path)
    if dtype is None and shape is None:
        return np.load(path, mmap_mode="r")
    if dtype is None or shape is None:
        raise ValueError("raw files need both dtype and shape")
    return np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))


def preload(data: Any, ram_fraction: float = 0.5) -> Any:
    """Read a memory-mapped array fully into RAM if it fits in ``ram_fraction`` of free RAM.

    Repeated passes (epochs) then never touch the disk. If it does not fit,
    the array is returned unchanged and ``chunks`` streams it from disk.
    """
    if not _is_mapped(data):
        return data
    budget = int(available_ram_bytes() * ram_fraction)
    if data.nbytes <= budget:
        return np.array(data)
    warnings.warn(
        f"{data.nbytes / MiB:.0f} MiB does not fit the RAM budget ({budget / MiB:.0f} MiB); "
        "it stays on disk and is streamed chunk by chunk",
        stacklevel=2,
    )
    return data


def _is_mapped(data: Any) -> bool:
    return isinstance(data, np.memmap) or (
        isinstance(data, np.ndarray) and isinstance(getattr(data, "base", None), np.memmap)
    )


def _as_numpy(data: Any) -> Tuple[np.ndarray, bool]:
    """(array view of ``data``, whether it came from torch)."""
    if isinstance(data, np.ndarray):
        return data, False
    torch = _torch()
    if torch is not None and isinstance(data, torch.Tensor):
        if data.device.type != "cpu":
            raise ValueError("data is already on a GPU; stream sources live in RAM or on disk")
        return data.numpy(), True
    return np.asarray(data), False


# -- planning --------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkPlan:
    rows: int  # rows (along axis 0) per chunk
    chunk_bytes: int
    total_rows: int
    count: int  # number of chunks
    buffers: int  # staging buffers in RAM

    def slices(self) -> List[slice]:
        return [slice(i * self.rows, min((i + 1) * self.rows, self.total_rows)) for i in range(self.count)]


def plan_chunks(
    data: Any,
    device: str = "cpu",
    chunk_bytes: Optional[int] = None,
    buffers: int = 3,
    ram_budget: Optional[int] = None,
    sharing: int = 1,
) -> ChunkPlan:
    """How to cut ``data`` along its first axis for ``device``.

    CPU: half the L3 cache per chunk (divided among ``sharing`` workers that
    share it). GPU: ``DEFAULT_GPU_CHUNK``, at most a fraction of the free GPU
    memory. Both are capped so ``buffers`` staged chunks fit ``ram_budget``
    (default: a quarter of available RAM).
    """
    array, _ = _as_numpy(data)
    if array.ndim == 0:
        raise ValueError("cannot chunk a scalar")
    if buffers < 1:
        raise ValueError("buffers must be at least 1")
    total_rows = array.shape[0]
    row_bytes = max(array.itemsize * math.prod(array.shape[1:]), 1)
    if chunk_bytes is None:
        if device == "cpu":
            chunk_bytes = max(l3_cache_bytes() // (2 * max(sharing, 1)), MIN_CPU_CHUNK)
        else:
            chunk_bytes = DEFAULT_GPU_CHUNK
            free = _gpu_free_bytes(device)
            if free:
                chunk_bytes = min(chunk_bytes, free // (2 * (buffers + 1)))
        ram = ram_budget if ram_budget is not None else available_ram_bytes() // 4
        chunk_bytes = max(min(chunk_bytes, ram // buffers), row_bytes)
    rows = max(1, min(chunk_bytes // row_bytes, max(total_rows, 1)))
    return ChunkPlan(
        rows=rows,
        chunk_bytes=rows * row_bytes,
        total_rows=total_rows,
        count=math.ceil(total_rows / rows) if total_rows else 0,
        buffers=buffers,
    )


# -- devices ---------------------------------------------------------------------


def resolve_devices(devices: Union[str, Sequence[str], None] = "auto") -> List[str]:
    """Turn ``"auto"``, ``"cuda"``, ``"cuda:1"``, ``"cpu"`` or a list into device names.

    ``"auto"`` follows Tablely: every GPU this job was given (they are the
    only ones CUDA can see), or the CPU when it was placed on CPU or no GPU
    is usable.
    """
    if devices is None or devices == "auto":
        assigned = os.environ.get("TABLELY_DEVICE")
        if assigned == "cpu":
            return ["cpu"]
        torch = _torch()
        if torch is not None and torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if assigned == "cuda":
            warnings.warn("Tablely gave this job a GPU but PyTorch cannot use CUDA here; using the CPU",
                          stacklevel=2)
        return ["cpu"]
    names = [devices] if isinstance(devices, str) else list(devices)
    resolved = []
    for name in names:
        name = "cuda:0" if name in ("cuda", "gpu") else name
        if name.startswith("cuda") and _torch() is None:
            raise RuntimeError(f"device {name!r} needs PyTorch (pip install torch)")
        resolved.append(name)
    return resolved


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _gpu_free_bytes(device: str) -> Optional[int]:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        free, _ = torch.cuda.mem_get_info(torch.device(device))
    except Exception:
        return None
    return int(free)


# -- backends --------------------------------------------------------------------


class Backend:
    """How a chunk moves. ``stage`` runs in the reader thread, the rest in the consumer.

    ``stage(slot, rows)`` copies source rows into staging slot ``slot`` (one of
    ``buffers``); it may block until that slot's previous transfer is done.
    ``send(slot, staged)`` starts moving it to the device and returns a handle;
    ``receive(handle)`` makes the chunk usable and returns it.
    """

    def stage(self, slot: int, rows: np.ndarray) -> Any:
        raise NotImplementedError

    def send(self, slot: int, staged: Any) -> Any:
        return staged

    def receive(self, handle: Any) -> Any:
        return handle


class HostBackend(Backend):
    """CPU target: chunks stay in RAM. Memory-mapped rows are read in by the reader thread."""

    def __init__(self, read_ahead: bool, as_torch: bool = False) -> None:
        self.read_ahead = read_ahead
        self.as_torch = as_torch

    def stage(self, slot: int, rows: np.ndarray) -> Any:
        return np.array(rows) if self.read_ahead else rows  # np.array forces the disk read here

    def receive(self, handle: Any) -> Any:
        return _torch().from_numpy(np.ascontiguousarray(handle)) if self.as_torch else handle


class CudaBackend(Backend):
    """GPU target: pinned RAM ring -> async DMA on a dedicated CUDA stream -> device tensor."""

    def __init__(self, device: str, plan: ChunkPlan, row_shape: Tuple[int, ...], dtype: np.dtype) -> None:
        torch = _torch()
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("streaming to a GPU needs PyTorch with CUDA")
        self.torch = torch
        self.device = torch.device(device)
        torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        self.pinned = [
            torch.empty((plan.rows, *row_shape), dtype=torch_dtype, pin_memory=True) for _ in range(plan.buffers)
        ]
        self.done: List[Any] = [None] * plan.buffers  # event: last DMA out of each pinned buffer finished
        self.copy_stream = torch.cuda.Stream(device=self.device)

    def stage(self, slot: int, rows: np.ndarray) -> Any:
        if self.done[slot] is not None:
            self.done[slot].synchronize()  # don't overwrite a buffer the DMA engine still reads
        n = len(rows)
        self.pinned[slot][:n].numpy()[...] = rows  # disk/RAM -> pinned RAM
        return n

    def send(self, slot: int, n: Any) -> Any:
        torch = self.torch
        with torch.cuda.stream(self.copy_stream):
            chunk = self.pinned[slot][:n].to(self.device, non_blocking=True)  # pinned RAM -> PCIe -> GPU
            event = torch.cuda.Event()
            event.record(self.copy_stream)
        self.done[slot] = event
        return chunk, event

    def receive(self, handle: Any) -> Any:
        chunk, event = handle
        compute = self.torch.cuda.current_stream(self.device)
        compute.wait_event(event)  # the compute stream waits for this chunk only, not for later ones
        chunk.record_stream(compute)  # allocated on the copy stream, used on the compute stream
        return chunk


# -- the pipeline ----------------------------------------------------------------


@dataclass
class StreamStats:
    chunks: int = 0
    bytes: int = 0
    wait_seconds: float = 0.0  # consumer time spent waiting for data (high = disk/bus bound)
    seconds: float = 0.0


_END = object()


class _Failure:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


@dataclass(eq=False)
class ChunkStream:
    """Iterate over chunks of ``data`` on one device, prefetching in the background.

    Use as an iterator (``for x in stream``) or a context manager; breaking out
    early stops the reader thread.
    """

    data: Any
    device: str = "cpu"
    plan: Optional[ChunkPlan] = None
    indices: Optional[Sequence[int]] = None  # which chunks (default all), e.g. one GPU's share
    backend: Optional[Backend] = None
    as_torch: Optional[bool] = None
    stats: StreamStats = field(default_factory=StreamStats)

    def __post_init__(self) -> None:
        self._array, from_torch = _as_numpy(self.data)
        if self.plan is None:
            self.plan = plan_chunks(self._array, self.device)
        if self.backend is None:
            if self.device == "cpu":
                as_torch = from_torch if self.as_torch is None else self.as_torch
                self.backend = HostBackend(read_ahead=_is_mapped(self._array), as_torch=as_torch)
            else:
                self.backend = CudaBackend(self.device, self.plan, self._array.shape[1:], self._array.dtype)
        self._closed = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __iter__(self) -> Iterator[Any]:
        for _, chunk in self.indexed():
            yield chunk

    def indexed(self) -> Iterator[Tuple[int, Any]]:
        """Like iterating, but yields ``(chunk_index, chunk)``."""
        slices = self.plan.slices()
        order = list(range(len(slices))) if self.indices is None else list(self.indices)
        free: "queue.Queue[int]" = queue.Queue()
        for slot in range(self.plan.buffers):
            free.put(slot)
        ready: "queue.Queue[Any]" = queue.Queue()
        self._thread = threading.Thread(target=self._read, args=(order, slices, free, ready), daemon=True)
        started = time.perf_counter()
        self._thread.start()
        try:
            item = self._take(ready)
            pending = self._send(item, free)
            while pending is not None:
                index, handle = pending
                item = self._take(ready)
                pending = self._send(item, free)  # the next chunk starts moving before this one is used
                chunk = self.backend.receive(handle)
                self.stats.chunks += 1
                yield index, chunk
        finally:
            self.close()
            self.stats.seconds = time.perf_counter() - started

    def close(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> "ChunkStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _read(self, order: List[int], slices: List[slice], free: "queue.Queue[int]", ready: "queue.Queue[Any]"):
        try:
            for index in order:
                slot = self._wait(free)
                if slot is None:
                    return
                rows = self._array[slices[index]]
                staged = self.backend.stage(slot, rows)
                ready.put((index, slot, staged, rows.nbytes))
            ready.put(_END)
        except BaseException as exc:  # handed to the consumer, raised there
            ready.put(_Failure(exc))

    def _wait(self, free: "queue.Queue[int]") -> Optional[int]:
        while not self._closed.is_set():
            try:
                return free.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    def _take(self, ready: "queue.Queue[Any]") -> Any:
        t0 = time.perf_counter()
        item = ready.get()
        self.stats.wait_seconds += time.perf_counter() - t0
        if isinstance(item, _Failure):
            raise item.exc
        return item

    def _send(self, item: Any, free: "queue.Queue[int]") -> Optional[Tuple[int, Any]]:
        if item is _END:
            return None
        index, slot, staged, nbytes = item
        handle = self.backend.send(slot, staged)
        free.put(slot)  # the reader may refill it once this transfer is done (the backend checks)
        self.stats.bytes += nbytes
        return index, handle


def chunks(
    data: Any,
    device: Optional[str] = None,
    *,
    chunk_bytes: Optional[int] = None,
    buffers: int = 3,
    as_torch: Optional[bool] = None,
) -> ChunkStream:
    """Stream ``data`` in order, chunk by chunk, to ``device`` (default: what Tablely assigned).

    On a GPU each chunk is a CUDA tensor; on CPU a NumPy array (a torch tensor
    if ``data`` is one, or with ``as_torch=True``).
    """
    device = device or resolve_devices("auto")[0]
    device = resolve_devices(device)[0]
    plan = plan_chunks(data, device, chunk_bytes=chunk_bytes, buffers=buffers)
    return ChunkStream(data, device=device, plan=plan, as_torch=as_torch)


def map_chunks(
    fn: Callable[[Any], Any],
    data: Any,
    devices: Union[str, Sequence[str], None] = "auto",
    *,
    chunk_bytes: Optional[int] = None,
    buffers: int = 3,
    workers: Optional[int] = None,
    as_torch: Optional[bool] = None,
) -> List[Any]:
    """Apply ``fn`` to every chunk on every available device; results in chunk order.

    GPUs: chunks are dealt round-robin, one pipeline (pinned ring, copy stream,
    reader) per GPU, all running at once. CPU: ``workers`` threads (default:
    the cores Tablely gave this job) each work on L3-sized blocks; NumPy and
    PyTorch release the GIL inside their kernels, so the threads run in
    parallel.
    """
    names = resolve_devices(devices)
    gpus = [d for d in names if d != "cpu"]
    if not gpus:
        return _map_on_cpu(fn, data, chunk_bytes, workers, as_torch)

    plan = plan_chunks(data, gpus[0], chunk_bytes=chunk_bytes, buffers=buffers)
    results: List[Any] = [None] * plan.count
    errors: List[BaseException] = []
    torch = _torch()

    def run(position: int, device: str) -> None:
        try:
            torch.cuda.set_device(torch.device(device))
            mine = list(range(position, plan.count, len(gpus)))
            for index, chunk in ChunkStream(data, device=device, plan=plan, indices=mine).indexed():
                results[index] = fn(chunk)
            torch.cuda.synchronize(torch.device(device))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i, d), daemon=True) for i, d in enumerate(gpus)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def _map_on_cpu(fn, data, chunk_bytes, workers, as_torch) -> List[Any]:
    array, from_torch = _as_numpy(data)
    if workers is None:
        try:
            workers = len(os.sched_getaffinity(0))
        except AttributeError:
            workers = os.cpu_count() or 1
    plan = plan_chunks(array, "cpu", chunk_bytes=chunk_bytes, sharing=workers)
    wrap = from_torch if as_torch is None else as_torch
    mapped = _is_mapped(array)

    def work(part: slice) -> Any:
        rows = np.array(array[part]) if mapped else array[part]
        return fn(_torch().from_numpy(np.ascontiguousarray(rows)) if wrap else rows)

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        return list(pool.map(work, plan.slices()))
