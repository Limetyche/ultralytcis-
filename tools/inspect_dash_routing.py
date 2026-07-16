#!/usr/bin/env python3
"""Inspect DASH-M1 routing coverage, complexity, latency, and memory without training."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, get_num_params


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/dash/yolov8n-dash-m1.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DetectionModel(args.model, verbose=False).to(device).eval()
    head = model.model[-1]
    x = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device)
    shapes = [
        (64, args.imgsz // 8, args.imgsz // 8),
        (128, args.imgsz // 16, args.imgsz // 16),
        (256, args.imgsz // 32, args.imgsz // 32),
    ]
    routing_macs = head.routing.theoretical_macs(shapes, args.batch)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000 / args.runs

    stats = head.routing_stats
    print(f"device={device}")
    print(f"parameters={get_num_params(model):,}")
    print(f"ultralytics_gflops={get_flops(model, args.imgsz):.4f}")
    print(f"routing_theoretical_macs={routing_macs:,} ({routing_macs / 1e9:.4f} GMAC)")
    print(f"latency_ms={latency_ms:.3f}")
    print(f"anchor_tokens={stats['anchor_tokens']}")
    print(f"dense_locations={sum(h * w for _, h, w in shapes)}")
    print(f"cls_delta_abs={[float(v) for v in stats['cls_delta_abs']]}")
    print(f"reg_delta_abs={[float(v) for v in stats['reg_delta_abs']]}")
    for name in ("cls_readout", "reg_readout"):
        values = stats[name]
        if values is not None:
            coverage = [float((v.abs().sum(-1) > 0).float().mean()) for v in values]
            entropy = [float((-(v * v.clamp_min(1e-9).log()).sum(-1)).mean()) for v in values]
            print(f"{name}_coverage={coverage}")
            print(f"{name}_entropy={entropy}")
    if device.type == "cuda":
        print(f"peak_memory_mb={torch.cuda.max_memory_allocated() / 2**20:.2f}")
    else:
        print("peak_memory_mb=not_available_on_cpu")


if __name__ == "__main__":
    main()
