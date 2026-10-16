"""
GPU / ML inference performance benchmark.

Measures throughput (samples/sec) and latency (ms/batch) for a PyTorch model
across a sweep of batch sizes, on whichever accelerator is available
(CUDA > MPS > CPU). Uses CUDA events for accurate device-side timing when on
CUDA, and synchronizes the device around every timed region so wall-clock
numbers reflect real compute, not queue-fill time.

Usage:
    python benchmark.py                          # ResNet-50, default sweep
    python benchmark.py --model resnet18 --dtype float16
    python benchmark.py --batch-sizes 1 8 32 128 --iters 50
"""

import argparse
import csv
import platform
import time
from dataclasses import dataclass, asdict

import torch
import torchvision.models as models


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({platform.processor() or platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_model(name: str, device: torch.device, dtype: torch.dtype):
    factory = getattr(models, name, None)
    if factory is None:
        raise ValueError(f"Unknown torchvision model: {name}")
    model = factory(weights=None)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


@dataclass
class Result:
    model: str
    device: str
    dtype: str
    batch_size: int
    iters: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p99: float
    throughput_sps: float


def time_batch(model, x, device, iters, warmup) -> list[float]:
    """Return per-iteration latencies in milliseconds."""
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync(device)

        latencies_ms: list[float] = []
        use_events = device.type == "cuda"
        for _ in range(iters):
            if use_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                latencies_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                model(x)
                sync(device)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return latencies_ms


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def run(args) -> list[Result]:
    device = pick_device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    # float16 on CPU is not well supported; guard against it.
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print(f"[warn] {args.dtype} on CPU is slow/unsupported; falling back to float32")
        dtype = torch.float32

    print(f"Device : {device_name(device)}  ({device.type})")
    print(f"Model  : {args.model}   dtype={dtype}   iters={args.iters} warmup={args.warmup}")
    print("-" * 72)

    model = build_model(args.model, device, dtype)
    results: list[Result] = []

    for bs in args.batch_sizes:
        try:
            x = torch.randn(bs, 3, 224, 224, device=device, dtype=dtype)
        except RuntimeError as e:
            print(f"batch={bs:<5} skipped (alloc failed: {e})")
            continue

        latencies = time_batch(model, x, device, args.iters, args.warmup)
        latencies.sort()
        mean_ms = sum(latencies) / len(latencies)
        p50 = percentile(latencies, 0.50)
        p99 = percentile(latencies, 0.99)
        throughput = bs / (mean_ms / 1000.0)

        results.append(Result(
            model=args.model, device=device.type, dtype=args.dtype,
            batch_size=bs, iters=args.iters,
            latency_ms_mean=round(mean_ms, 3),
            latency_ms_p50=round(p50, 3),
            latency_ms_p99=round(p99, 3),
            throughput_sps=round(throughput, 1),
        ))
        print(f"batch={bs:<5} lat_mean={mean_ms:8.2f} ms  "
              f"p50={p50:8.2f}  p99={p99:8.2f}  "
              f"throughput={throughput:9.1f} samples/s")

    return results


def write_csv(results: list[Result], path: str) -> None:
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nWrote {len(results)} rows -> {path}")


def parse_args():
    p = argparse.ArgumentParser(description="PyTorch GPU inference benchmark")
    p.add_argument("--model", default="resnet50",
                   help="torchvision model name (e.g. resnet18, resnet50)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--batch-sizes", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--csv", default="results/benchmark.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run(args)
    write_csv(results, args.csv)
