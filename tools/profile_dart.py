#!/usr/bin/env python3
"""Profile B0/A1/A2/M1 without training; CUDA metrics are reported only when CUDA is available."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from thop import profile

from ultralytics.nn.modules.dart import DARTDetect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

MODELS = {
    "B0": Path("ultralytics/cfg/models/v8/yolov8.yaml"),
    "A1": Path("ultralytics/cfg/models/dart/yolov8n-dart-feature-only.yaml"),
    "A2": Path("ultralytics/cfg/models/dart/yolov8n-dart-shared-scale.yaml"),
    "M1": Path("ultralytics/cfg/models/dart/yolov8n-dart-m1.yaml"),
}


def state_ops(reg_max: int, locations: int) -> int:
    """Count scalar softmax/statistic operations (not MACs) using an explicit documented convention."""
    # Per edge/bin: softmax exp+sum+divide=3R, mean multiply+sum=2R, variance subtract/square/multiply+sum=4R,
    # entropy clamp+log+multiply+sum=4R; plus three normalizations per edge.
    return locations * 4 * (13 * reg_max + 3)


def refiner_macs(head: DARTDetect, imgsz: int) -> list[int]:
    macs = []
    for level, refiner in enumerate(head.refiners):
        h = w = imgsz // int(head.stride[level])
        cin = refiner.feature_projection.in_channels
        hidden = refiner.feature_projection.out_channels
        fused = hidden + refiner.state_channels
        macs.append(h * w * (cin * hidden + 9 * fused + fused * hidden + hidden * 4 * head.reg_max))
    return macs


def cuda_metrics(model, image, warmup: int, runs: int):
    if not torch.cuda.is_available():
        return None, None
    model, image = model.cuda().eval(), image.cuda()
    with torch.inference_mode():
        for _ in range(warmup):
            model(image)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(runs):
            model(image)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / runs, torch.cuda.max_memory_allocated() / 2**20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    print("THOP reports multiply-add as 2 FLOPs; state statistics are separate scalar-op estimates.")
    for name, path in MODELS.items():
        model = DetectionModel(path, verbose=False).eval()
        image = torch.randn(1, 3, args.imgsz, args.imgsz)
        params = sum(p.numel() for p in model.parameters())
        summary = get_flops(model, args.imgsz)
        mac, _ = profile(model, inputs=(image,), verbose=False)
        real_gflops = mac * 2 / 1e9
        latency, memory = cuda_metrics(model, image, args.warmup, args.runs)
        head = model.model[-1]
        level_macs = refiner_macs(head, args.imgsz) if isinstance(head, DARTDetect) else [0, 0, 0]
        locations = sum((args.imgsz // int(s)) ** 2 for s in head.stride)
        statistics = state_ops(head.reg_max, locations) if getattr(head, "use_distribution_state", False) else 0
        refinement_ratio = (sum(level_macs) * 2 / 1e9) / real_gflops if real_gflops else 0.0
        print(
            f"{name}: params={params:,} summary={summary:.4f} GFLOPs real640={real_gflops:.4f} GFLOPs "
            f"latency={latency if latency is not None else 'N/A'} ms peak={memory if memory is not None else 'N/A'} MiB "
            f"refiner_MACs={level_macs} state_ops={statistics} refinement/base={refinement_ratio:.6f}"
        )


if __name__ == "__main__":
    main()
