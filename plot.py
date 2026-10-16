"""
Chart throughput and latency vs. batch size from benchmark.csv.

Usage:
    python plot.py                       # reads results/benchmark.csv
    python plot.py --csv results/foo.csv --out results/foo.png
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")  # headless: write PNG without a display
import matplotlib.pyplot as plt


def load(path: str):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    batch = [int(r["batch_size"]) for r in rows]
    throughput = [float(r["throughput_sps"]) for r in rows]
    lat_mean = [float(r["latency_ms_mean"]) for r in rows]
    lat_p99 = [float(r["latency_ms_p99"]) for r in rows]
    meta = rows[0] if rows else {}
    return batch, throughput, lat_mean, lat_p99, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/benchmark.csv")
    p.add_argument("--out", default="results/benchmark.png")
    args = p.parse_args()

    batch, throughput, lat_mean, lat_p99, meta = load(args.csv)
    title = (f"{meta.get('model','model')} inference — "
             f"{meta.get('device','?')} / {meta.get('dtype','?')}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1.plot(batch, throughput, "o-", color="#76b900")  # NVIDIA green, why not
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Throughput (samples/sec)")
    ax1.set_title("Throughput scaling")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)

    ax2.plot(batch, lat_mean, "o-", label="mean", color="#1f77b4")
    ax2.plot(batch, lat_p99, "s--", label="p99", color="#d62728")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Latency (ms/batch)")
    ax2.set_title("Latency vs. batch size")
    ax2.set_xscale("log", base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Wrote chart -> {args.out}")


if __name__ == "__main__":
    main()
