"""Stand-in for a real training script: keeps cores busy for a while and
reports what Tablely gave it. Replace it with your own training command."""

import argparse
import os
import threading
import time


def burn(seconds: float) -> None:
    end = time.time() + seconds
    x = 0
    while time.time() < end:
        x = (x * 31 + 7) % 1_000_003


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    print(
        f"job={os.environ.get('TABLELY_JOB')} device={args.device} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"cores={os.environ.get('TABLELY_CPU_LIST')} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}",
        flush=True,
    )
    workers = [threading.Thread(target=burn, args=(args.seconds,)) for _ in range(args.threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    print("finished", flush=True)


if __name__ == "__main__":
    main()
