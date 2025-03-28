"""Microbenchmark: how fast is selection vs catalogue size?"""
from __future__ import annotations

import argparse
import time

from toolrouter import Tool, ToolRegistry, Router
from toolrouter.embedders import BagOfWordsEmbedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[10, 50, 200, 1000])
    ap.add_argument("--queries", type=int, default=100)
    args = ap.parse_args()

    print(f"{'N_tools':>8s} {'p50_us':>8s} {'p95_us':>8s}")
    for N in args.N:
        reg = ToolRegistry([
            Tool(f"tool_{i:04d}", f"Synthetic tool number {i} for benchmarking",
                 tags=[f"g{i%5}"])
            for i in range(N)
        ])
        router = Router(reg, BagOfWordsEmbedder(dim=512))
        latencies = []
        for q in range(args.queries):
            t0 = time.perf_counter()
            router.select(f"please find tool {q % N}", k=5)
            latencies.append((time.perf_counter() - t0) * 1e6)
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        print(f"{N:>8d} {p50:>8.1f} {p95:>8.1f}")


if __name__ == "__main__":
    main()
