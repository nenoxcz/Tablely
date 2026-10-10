"""Big tables kept in RAM that Tablely set aside, used from the GPU as if they were in VRAM.

Embedding tables, feature tables and lookup tables often do not fit in GPU
memory next to the model. ``tables`` keeps them in RAM reserved for the job
before it started (``ram = "32GiB"`` in the job file) and sends the GPU only
the rows a step needs::

    from tablely import tables

    store = tables.store()                    # pins the RAM Tablely reserved for this job, once
    emb = store.put("item_emb", weights)      # on the GPU if it fits the VRAM budget, else in that RAM
    vecs = emb.gather(ids)                    # rows on the GPU, wherever the table lives
    emb.add_rows(ids, -lr * grads)            # sparse update in place (e.g. SGD on embeddings)

    step = store.guard(train_step)            # GPU out of memory -> move a table to RAM, run again
    for batch in loader:
        loss = step(batch)

Placement: tables go to the GPU while they fit ``vram_budget`` (half of the
GPU memory this job may use, by default) and to the reserved RAM after that.
When a guarded step runs out of GPU memory, the biggest table on the GPU
moves to RAM, cached GPU memory is released and the step runs again, so
training slows down a little instead of crashing. ``promote()`` moves tables
back while there is room.

How rows of a RAM table reach the GPU::

    pinned RAM table --(index_select on CPU)--> pinned rows --(PCIe DMA)--> GPU

Only the requested rows cross the bus, and the RAM is page-locked, so the
copy engine reads it directly without an extra copy.

Without a GPU (a job placed on CPU) every table lives in RAM. With PyTorch
installed tables are torch tensors; without it they are NumPy arrays.
"""

from __future__ import annotations

import functools
import math
import os
import sys
import threading
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .resources import format_bytes, parse_bytes

GPU = "gpu"
RAM = "ram"
ALIGN = 256  # bytes: every table starts aligned for any dtype and for DMA
COPY_CHUNK = 64 * 1024 * 1024  # big sources (e.g. memory-mapped files) are copied in pieces this size

Size = Union[int, str, None]


# -- RAM set aside up front ------------------------------------------------------


class Arena:
    """First-fit allocator over one block of RAM that was allocated (and pinned) up front."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.used = 0
        self._free: List[Tuple[int, int]] = [(0, size)]  # (offset, length), sorted, never touching

    @staticmethod
    def rounded(nbytes: int) -> int:
        return max(-(-nbytes // ALIGN) * ALIGN, ALIGN)

    def alloc(self, nbytes: int) -> Optional[int]:
        """Offset of a free block of ``nbytes``, or None if no block is big enough."""
        need = self.rounded(nbytes)
        for i, (offset, length) in enumerate(self._free):
            if length >= need:
                if length == need:
                    del self._free[i]
                else:
                    self._free[i] = (offset + need, length - need)
                self.used += need
                return offset
        return None

    def free(self, offset: int, nbytes: int) -> None:
        length = self.rounded(nbytes)
        self.used -= length
        merged: List[Tuple[int, int]] = []
        for start, size in sorted(self._free + [(offset, length)]):
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((start, size))
        self._free = merged

    @property
    def largest_free(self) -> int:
        return max((size for _, size in self._free), default=0)


# -- backends --------------------------------------------------------------------


def _row_slices(rows: int, row_bytes: int) -> Iterator[slice]:
    step = max(1, COPY_CHUNK // max(row_bytes, 1))
    for start in range(0, rows, step):
        yield slice(start, min(start + step, rows))


class NumpyBackend:
    """Every table in RAM as a NumPy array: no PyTorch available."""

    gpu = False
    device = "cpu"

    def dtype(self, dtype: Any) -> Any:
        return np.dtype(dtype)

    def itemsize(self, dtype: Any) -> int:
        return np.dtype(dtype).itemsize

    def describe(self, data: Any) -> Tuple[Tuple[int, ...], Any]:
        if not hasattr(data, "shape") or not hasattr(data, "dtype"):
            data = np.asarray(data)
        return tuple(int(n) for n in data.shape), self.dtype(data.dtype)

    def host_buffer(self, nbytes: int) -> Any:
        buffer = np.empty(nbytes, dtype=np.uint8)
        buffer.fill(0)  # touch every page now, so the RAM really is set aside before training
        return buffer

    def host_view(self, buffer: Any, offset: int, shape: Tuple[int, ...], dtype: Any) -> Any:
        nbytes = math.prod(shape) * self.itemsize(dtype)
        return buffer[offset:offset + nbytes].view(dtype).reshape(shape)

    def host_array(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        return np.empty(shape, dtype=dtype)

    def fill_host(self, host: Any, data: Any) -> None:
        if data is None:
            host[...] = 0
            return
        for part in _row_slices(len(host), host[:1].nbytes):
            host[part] = np.asarray(data[part])  # reads a memory-mapped source piece by piece

    def take(self, table: Any, ids: Any, on_gpu: bool) -> Any:
        return table[np.asarray(ids, dtype=np.int64)]

    def rows(self, table: Any, start: int, stop: int, on_gpu: bool) -> Any:
        return table[start:stop]

    def put_rows(self, table: Any, ids: Any, values: Any, add: bool, on_gpu: bool) -> None:
        index = np.asarray(ids, dtype=np.int64)
        values = np.asarray(values, dtype=table.dtype)
        if add:
            np.add.at(table, index, values)  # repeated ids add up
        else:
            table[index] = values

    def to_numpy(self, table: Any) -> np.ndarray:
        return np.array(table)

    def upload(self, data: Any, shape: Tuple[int, ...], dtype: Any) -> Any:
        raise RuntimeError("this job has no GPU")

    def download(self, device_table: Any, host: Any) -> None:
        raise RuntimeError("this job has no GPU")

    def is_oom(self, exc: BaseException) -> bool:
        return False

    def empty_cache(self) -> None:
        pass

    def sync(self) -> None:
        pass

    def vram(self) -> Tuple[Optional[int], Optional[int]]:
        """(free, total) bytes of GPU memory, or (None, None) without a GPU."""
        return None, None


class TorchBackend:
    """PyTorch tensors: RAM tables in pinned memory, GPU tables as CUDA tensors."""

    def __init__(self, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.gpu = self.device.type == "cuda"
        if self.gpu and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

    def dtype(self, dtype: Any) -> Any:
        torch = self.torch
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str) and isinstance(getattr(torch, dtype, None), torch.dtype):
            return getattr(torch, dtype)  # also "bfloat16", which NumPy lacks
        return torch.from_numpy(np.empty(0, dtype=np.dtype(dtype))).dtype

    def itemsize(self, dtype: Any) -> int:
        return self.torch.empty((), dtype=self.dtype(dtype)).element_size()

    def describe(self, data: Any) -> Tuple[Tuple[int, ...], Any]:
        if not hasattr(data, "shape") or not hasattr(data, "dtype"):
            data = np.asarray(data)
        return tuple(int(n) for n in data.shape), self.dtype(data.dtype)

    def host_buffer(self, nbytes: int) -> Any:
        torch = self.torch
        if self.gpu:
            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)  # page-locked from now on
        return torch.zeros(nbytes, dtype=torch.uint8)  # zeroing touches every page now

    def host_view(self, buffer: Any, offset: int, shape: Tuple[int, ...], dtype: Any) -> Any:
        nbytes = math.prod(shape) * self.itemsize(dtype)
        return buffer[offset:offset + nbytes].view(self.dtype(dtype)).view(shape)

    def host_array(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        return self.torch.empty(shape, dtype=self.dtype(dtype), pin_memory=self.gpu)

    def _copy_into(self, target: Any, data: Any) -> None:
        torch = self.torch
        if isinstance(data, torch.Tensor):
            target.copy_(data)
            return
        row_bytes = math.prod(target.shape[1:]) * target.element_size()
        for part in _row_slices(target.shape[0], row_bytes):
            target[part].copy_(torch.from_numpy(np.array(data[part])))  # a writable copy, even of a read-only map

    def fill_host(self, host: Any, data: Any) -> None:
        if data is None:
            host.zero_()
        else:
            self._copy_into(host, data)

    def upload(self, data: Any, shape: Tuple[int, ...], dtype: Any) -> Any:
        table = self.torch.empty(shape, dtype=self.dtype(dtype), device=self.device)
        if data is None:
            table.zero_()
        else:
            self._copy_into(table, data)
        return table

    def download(self, device_table: Any, host: Any) -> None:
        host.copy_(device_table)

    def _index(self, ids: Any, device: Any) -> Any:
        torch = self.torch
        index = ids if isinstance(ids, torch.Tensor) else torch.as_tensor(np.asarray(ids))
        return index.to(device=device, dtype=torch.long)

    def take(self, table: Any, ids: Any, on_gpu: bool) -> Any:
        torch = self.torch
        if on_gpu:
            return table.index_select(0, self._index(ids, self.device))
        index = self._index(ids, "cpu")
        if not self.gpu:
            return table.index_select(0, index)
        rows = torch.empty((len(index), *table.shape[1:]), dtype=table.dtype, pin_memory=True)
        torch.index_select(table, 0, index, out=rows)  # gathered in RAM by the CPU
        return rows.to(self.device, non_blocking=True)  # only these rows cross the bus

    def rows(self, table: Any, start: int, stop: int, on_gpu: bool) -> Any:
        part = table[start:stop]
        if on_gpu or not self.gpu:
            return part
        staged = self.torch.empty(part.shape, dtype=part.dtype, pin_memory=True)
        staged.copy_(part)  # a copy, so later writes to the table cannot race the transfer
        return staged.to(self.device, non_blocking=True)

    def put_rows(self, table: Any, ids: Any, values: Any, add: bool, on_gpu: bool) -> None:
        torch = self.torch
        device = self.device if on_gpu else torch.device("cpu")
        index = self._index(ids, device)
        values = torch.as_tensor(values).to(device=device, dtype=table.dtype)
        if add:
            table.index_add_(0, index, values)  # repeated ids add up
        else:
            table.index_copy_(0, index, values)

    def to_numpy(self, table: Any) -> np.ndarray:
        return table.detach().cpu().numpy()

    def is_oom(self, exc: BaseException) -> bool:
        oom = getattr(self.torch.cuda, "OutOfMemoryError", None)
        if oom is not None and isinstance(exc, oom):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def empty_cache(self) -> None:
        if self.gpu:
            self.torch.cuda.empty_cache()

    def sync(self) -> None:
        if self.gpu:
            self.torch.cuda.synchronize(self.device)

    def vram(self) -> Tuple[Optional[int], Optional[int]]:
        if not self.gpu:
            return None, None
        free, total = self.torch.cuda.mem_get_info(self.device)
        return int(free), int(total)


def default_backend(device: Optional[str] = None) -> Any:
    """PyTorch on the device Tablely gave this job (its GPU, or CPU), NumPy without PyTorch."""
    try:
        import torch
    except ImportError:
        return NumpyBackend()
    if device is None:
        on_gpu = os.environ.get("TABLELY_DEVICE") != "cpu" and torch.cuda.is_available()
        device = "cuda" if on_gpu else "cpu"
    return TorchBackend(device)


# -- tables ----------------------------------------------------------------------


class Table:
    """One table of a :class:`Store`. It lives on the GPU or in RAM (``where``) and may move."""

    def __init__(self, store: "Store", name: str, shape: Tuple[int, ...], dtype: Any, nbytes: int,
                 keep_on_gpu: bool) -> None:
        self._store = store
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes
        self.keep_on_gpu = keep_on_gpu  # never moved to RAM to make room
        self.where = RAM
        self.data: Any = None  # the GPU tensor or the RAM array; hold no reference across steps
        self.uses = 0
        self._offset: Optional[int] = None

    @property
    def on_gpu(self) -> bool:
        return self.where == GPU

    def __len__(self) -> int:
        return self.shape[0]

    def gather(self, ids: Any) -> Any:
        """Rows ``ids`` on this job's device. From RAM only these rows cross the bus."""
        self.uses += 1
        return self._store.backend.take(self.data, ids, self.on_gpu)

    def rows(self, start: int = 0, stop: Optional[int] = None) -> Any:
        """Rows ``start:stop`` on this job's device (a view when the table is on the GPU)."""
        self.uses += 1
        return self._store.backend.rows(self.data, start, len(self) if stop is None else stop, self.on_gpu)

    def set_rows(self, ids: Any, values: Any) -> None:
        """Overwrite rows ``ids`` with ``values``."""
        self._store.backend.put_rows(self.data, ids, values, False, self.on_gpu)

    def add_rows(self, ids: Any, values: Any) -> None:
        """Add ``values`` to rows ``ids`` in place, e.g. ``-lr * grad`` for sparse SGD."""
        self._store.backend.put_rows(self.data, ids, values, True, self.on_gpu)

    def numpy(self) -> np.ndarray:
        """A NumPy copy of the whole table."""
        return self._store.backend.to_numpy(self.data)

    def __repr__(self) -> str:
        return f"Table({self.name!r}, shape={self.shape}, {self.where}, {format_bytes(self.nbytes)})"


class Store:
    """Tables of one job, in its GPU memory and in the RAM set aside for it.

    ``ram``: bytes to allocate up front for RAM tables (default: none, each
    RAM table gets its own allocation). ``vram``: how much GPU memory tables
    may take before new ones go to RAM (default: ``vram_fraction`` of the GPU
    memory this job may use).
    """

    def __init__(
        self,
        ram: Size = None,
        vram: Size = None,
        *,
        vram_fraction: float = 0.5,
        backend: Any = None,
        report: bool = True,
    ) -> None:
        self.backend = backend if backend is not None else default_backend()
        ram_bytes = parse_bytes(ram) if ram is not None else None
        self.arena = Arena(ram_bytes) if ram_bytes else None
        self._buffer = self.backend.host_buffer(ram_bytes) if ram_bytes else None
        if vram is not None:
            self.vram_budget = parse_bytes(vram)
        elif not self.backend.gpu:
            self.vram_budget = 0
        else:
            limit = os.environ.get("TABLELY_GPU_MEMORY")  # set when this job shares its GPU
            total = int(limit) if limit and limit.isdigit() else self.backend.vram()[1] or 0
            self.vram_budget = int(total * vram_fraction)
        self.tables: Dict[str, Table] = {}
        self.spills = 0
        self._report = report
        self._lock = threading.RLock()

    # -- reading ----------------------------------------------------------------

    @property
    def vram_used(self) -> int:
        return sum(t.nbytes for t in self.tables.values() if t.on_gpu)

    @property
    def ram_used(self) -> int:
        return sum(t.nbytes for t in self.tables.values() if not t.on_gpu)

    def __getitem__(self, name: str) -> Table:
        return self.tables[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tables

    def __iter__(self) -> Iterator[Table]:
        return iter(list(self.tables.values()))

    def summary(self) -> str:
        lines = [f"{t.name}: {t.shape} on {t.where.upper()}, {format_bytes(t.nbytes)}" for t in self]
        reserved = f" of {format_bytes(self.arena.size)} reserved" if self.arena else ""
        lines.append(
            f"GPU {format_bytes(self.vram_used)} of {format_bytes(self.vram_budget)} budget · "
            f"RAM {format_bytes(self.ram_used)}{reserved} · {self.spills} move(s) to RAM"
        )
        return "\n".join(lines)

    # -- adding and removing ----------------------------------------------------

    def put(self, name: str, data: Any, *, where: str = "auto", keep_on_gpu: bool = False) -> Table:
        """Copy ``data`` (array, tensor or memory-mapped file) into a new table.

        ``where``: ``"auto"`` (GPU while it fits the budget, else RAM), ``"gpu"``
        or ``"ram"``. ``keep_on_gpu`` tables are never moved to RAM for room.
        """
        shape, dtype = self.backend.describe(data)
        return self._add(name, shape, dtype, data, where, keep_on_gpu)

    def zeros(self, name: str, shape: Union[int, Sequence[int]], dtype: Any = "float32", *,
              where: str = "auto", keep_on_gpu: bool = False) -> Table:
        """A new table of zeros, e.g. embeddings to be initialized in place."""
        dims = (shape,) if isinstance(shape, int) else tuple(shape)
        return self._add(name, tuple(int(n) for n in dims), self.backend.dtype(dtype), None, where, keep_on_gpu)

    def drop(self, name: str) -> None:
        with self._lock:
            table = self.tables.pop(name)
            if table.on_gpu:
                table.data = None
                self.backend.empty_cache()
            else:
                self._free_ram(table)
                table.data = None
        self._publish()

    def _add(self, name: str, shape: Tuple[int, ...], dtype: Any, data: Any, where: str,
             keep_on_gpu: bool) -> Table:
        if where not in ("auto", GPU, RAM):
            raise ValueError(f'where must be "auto", "gpu" or "ram" (got {where!r})')
        if not shape:
            raise ValueError(f"table {name!r} needs at least one dimension (its rows)")
        nbytes = math.prod(shape) * self.backend.itemsize(dtype)
        table = Table(self, name, shape, dtype, nbytes, keep_on_gpu)
        with self._lock:
            if name in self.tables:
                raise ValueError(f"table {name!r} already exists; drop it first")
            if where == GPU and not self.backend.gpu:
                raise ValueError(f"table {name!r}: this job has no GPU")
            fits = self.backend.gpu and self.vram_used + nbytes <= self.vram_budget
            if where == GPU or (where == "auto" and fits):
                full = False
                try:
                    table.data = self.backend.upload(data, shape, dtype)
                    table.where = GPU
                except Exception as exc:
                    if where == GPU or not self.backend.is_oom(exc):
                        raise
                    full = True
                if full:
                    self.backend.empty_cache()  # outside the except: the failed upload is gone
            if not table.on_gpu:
                table.data = self._alloc_ram(table)
                self.backend.fill_host(table.data, data)
            self.tables[name] = table
        self._publish()
        return table

    def _alloc_ram(self, table: Table) -> Any:
        if self.arena is None:
            return self.backend.host_array(table.shape, table.dtype)
        offset = self.arena.alloc(table.nbytes)
        if offset is None:
            left = self.arena.size - self.arena.used
            raise MemoryError(
                f"table {table.name!r} ({format_bytes(table.nbytes)}) does not fit in the RAM reserved for "
                f"this job ({format_bytes(left)} of {format_bytes(self.arena.size)} left); "
                'reserve more with ram = "..." in the job file'
            )
        table._offset = offset
        return self.backend.host_view(self._buffer, offset, table.shape, table.dtype)

    def _free_ram(self, table: Table) -> None:
        if table._offset is not None:
            self.backend.sync()  # no transfer may still be reading this RAM
            self.arena.free(table._offset, table.nbytes)
            table._offset = None

    # -- moving -----------------------------------------------------------------

    def spill(self, name: Optional[str] = None, reason: str = "asked") -> Optional[Table]:
        """Move table ``name`` (default: the biggest movable one on the GPU) to RAM.

        Returns the moved table, or None if nothing could move. Raises
        ``MemoryError`` when the reserved RAM has no room for it.
        """
        with self._lock:
            if name is not None:
                table = self.tables[name]
                if not table.on_gpu:
                    return None
            else:
                movable = [t for t in self.tables.values() if t.on_gpu and not t.keep_on_gpu]
                if not movable:
                    return None
                table = max(movable, key=lambda t: t.nbytes)
            host = self._alloc_ram(table)
            self.backend.download(table.data, host)
            table.data = host  # the GPU copy is no longer referenced here...
            table.where = RAM
            self.backend.empty_cache()  # ...so its memory goes back to the GPU
            self.spills += 1
        self._publish(f"moved table {table.name} ({format_bytes(table.nbytes)}) from GPU to RAM: {reason}")
        return table

    def promote(self, headroom: float = 0.1) -> List[Table]:
        """Move RAM tables back to the GPU, most used first, while they fit the budget.

        A table moves only if it also leaves ``headroom`` of the free GPU
        memory untouched. Returns the tables moved.
        """
        moved: List[Table] = []
        with self._lock:
            waiting = sorted((t for t in self.tables.values() if not t.on_gpu), key=lambda t: (-t.uses, t.nbytes))
            for table in waiting:
                if self.vram_used + table.nbytes > self.vram_budget:
                    continue
                free, _ = self.backend.vram()
                if free is not None and table.nbytes > free * (1 - headroom):
                    continue
                full = False
                try:
                    device_table = self.backend.upload(table.data, table.shape, table.dtype)
                except Exception as exc:
                    if not self.backend.is_oom(exc):
                        raise
                    full = True
                if full:
                    self.backend.empty_cache()
                    break
                self._free_ram(table)
                table.data = device_table
                table.where = GPU
                moved.append(table)
        if moved:
            names = ", ".join(f"{t.name} ({format_bytes(t.nbytes)})" for t in moved)
            self._publish(f"moved table(s) {names} from RAM back to GPU")
        return moved

    # -- running steps ----------------------------------------------------------

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``fn``; if it runs out of GPU memory, move a table to RAM and call it again.

        Repeats until the call succeeds or no table is left to move (then the
        out-of-memory error is raised). ``fn`` must be safe to run again: zero
        the gradients at its start, and step the optimizer at its end.
        """
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if not self.backend.is_oom(exc):
                    raise
                error = exc.with_traceback(None)  # drop the failed step's frames and their GPU tensors
            self.backend.empty_cache()
            try:
                moved = self.spill(reason="a step ran out of GPU memory")
            except MemoryError:
                moved = None
            if moved is None:
                raise error
            print(f"tablely.tables: GPU out of memory; moved {moved.name} ({format_bytes(moved.nbytes)}) "
                  "to RAM and running the step again", file=sys.stderr, flush=True)

    def guard(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """``fn`` wrapped with :meth:`run`, e.g. ``step = store.guard(train_step)``."""

        @functools.wraps(fn)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            return self.run(fn, *args, **kwargs)

        return guarded

    # -- bookkeeping ------------------------------------------------------------

    def _publish(self, event: Optional[str] = None) -> None:
        """Show where the tables are in ``tablely status``; log moves to the history."""
        if not self._report:
            return
        summary = {
            "gpu": self.vram_used,
            "ram": self.ram_used,
            "reserved": self.arena.size if self.arena else None,
            "count": len(self.tables),
            "spills": self.spills,
        }
        from . import client

        try:
            client._update_job({"tables": summary}, "tables" if event else None, event or "")
        except Exception:  # bookkeeping must never break training
            pass


_default: Optional[Store] = None
_default_lock = threading.Lock()


def store(**settings: Any) -> Store:
    """This job's :class:`Store`, created on the first call; later calls return the same one.

    The RAM Tablely reserved for the job (``ram`` in the job file,
    ``$TABLELY_RAM``) is allocated and pinned right away, before training
    starts. ``settings`` (``ram``, ``vram``, ``vram_fraction``, ``backend``)
    apply to the first call only.
    """
    global _default
    with _default_lock:
        if _default is None:
            if settings.get("ram") is None:
                reserved = os.environ.get("TABLELY_RAM")
                settings["ram"] = int(reserved) if reserved and reserved.isdigit() else None
            _default = Store(**settings)
        elif settings:
            warnings.warn("tables.store() already exists; settings only apply to the first call", stacklevel=2)
        return _default
