"""Example job: streams a dataset through in chunks every epoch, and moves
between CPU and GPU when Tablely asks (the job is marked switchable = true).

Runs anywhere: on CPU it needs only NumPy; on a GPU, PyTorch with CUDA.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tablely import client, stream


def make_dataset(path: Path, rows: int, cols: int) -> None:
    """Write a dataset chunk by chunk, so it never has to fit in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, cols))
    rng = np.random.default_rng(0)
    for start in range(0, rows, 65536):
        stop = min(start + 65536, rows)
        out[start:stop] = rng.random((stop - start, cols), dtype=np.float32)
    out.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=400_000)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    data_path = Path("data/features.npy")
    if not data_path.exists():
        make_dataset(data_path, args.rows, args.cols)
    # In RAM if it fits half of the free memory, otherwise streamed from disk every epoch.
    data = stream.preload(stream.open_array(data_path))

    checkpoint = Path("checkpoints") / f"{os.environ.get('TABLELY_JOB', 'stream-train')}.json"
    start = json.loads(checkpoint.read_text())["epoch"] if client.restarts() and checkpoint.exists() else 0
    device = client.device()
    print(f"run {client.restarts()}: device={device}, starting at epoch {start}", flush=True)

    for epoch in range(start, args.epochs):
        total = 0.0
        pipe = stream.chunks(data)  # GPU: disk/RAM -> pinned RAM -> PCIe -> GPU; CPU: L3-sized blocks
        for chunk in pipe:
            total += float(chunk.sum())  # the "training step"; works for NumPy arrays and torch tensors
        rate = pipe.stats.bytes / max(pipe.stats.seconds, 1e-9) / 2**20
        client.progress(f"epoch {epoch + 1}/{args.epochs} on {device}, {rate:.0f} MiB/s, "
                        f"waited {pipe.stats.wait_seconds:.2f}s for data")
        print(f"epoch {epoch + 1}: {pipe.plan.count} chunks of {pipe.plan.rows} rows, sum={total:.1f}", flush=True)

        if client.switch_requested():  # a GPU freed up, or a bigger job needs ours
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({"epoch": epoch + 1}))
            client.exit_for_switch()


if __name__ == "__main__":
    main()
