"""Example job: learns user and item embeddings kept in tablely.tables.

The tables go to the GPU while they fit the VRAM budget and to the RAM Tablely
reserved for this job (ram = "..." in the job file) after that. Each step
only gathers the rows it needs, and if a step runs out of GPU memory a table
moves to RAM and the step runs again.

Runs anywhere: on CPU it needs only NumPy; on a GPU, PyTorch with CUDA.
"""

import argparse

import numpy as np

from tablely import client, tables


def main() -> None:
    parser = argparse.ArgumentParser()
    # Make --users/--items large (e.g. 50_000_000) to see tables spill out of a real GPU.
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=2_000)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    store = tables.store()  # pins the RAM Tablely set aside for this job, before training
    rng = np.random.default_rng(0)
    scale = args.dim ** -0.25
    true_users = store.put("true_users", rng.normal(0, scale, (args.users, args.dim)).astype(np.float32))
    true_items = store.put("true_items", rng.normal(0, scale, (args.items, args.dim)).astype(np.float32))
    users = store.put("users", rng.normal(0, 0.3, (args.users, args.dim)).astype(np.float32))
    items = store.put("items", rng.normal(0, 0.3, (args.items, args.dim)).astype(np.float32))
    print(store.summary(), flush=True)

    def train_step(user_ids, item_ids):
        u, v = users.gather(user_ids), items.gather(item_ids)
        target = (true_users.gather(user_ids) * true_items.gather(item_ids)).sum(1)
        err = (u * v).sum(1) - target
        du, dv = -args.lr * err[:, None] * v, -args.lr * err[:, None] * u
        users.add_rows(user_ids, du)  # updates come last, so a step cut short can run again
        items.add_rows(item_ids, dv)
        return float((err ** 2).mean())

    step = store.guard(train_step)  # GPU out of memory -> a table moves to RAM, the step runs again
    for i in range(args.steps):
        loss = step(rng.integers(0, args.users, args.batch), rng.integers(0, args.items, args.batch))
        if (i + 1) % 100 == 0 or i + 1 == args.steps:
            client.progress(f"step {i + 1}/{args.steps}, loss {loss:.4f}")
            print(f"step {i + 1}: loss {loss:.4f}", flush=True)
    print(store.summary(), flush=True)


if __name__ == "__main__":
    main()
