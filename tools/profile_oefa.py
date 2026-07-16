#!/usr/bin/env python3
"""Profile standard YOLO11n and OEFA-M1 controls without training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from thop import profile

from ultralytics.nn.modules.oefa import BoundaryPreservingDownsample, OEFAM1Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

MODELS = {
    "B0": Path("ultralytics/cfg/models/11/yolo11.yaml"),
    "Identity": Path("ultralytics/cfg/models/oefa/yolo11n-oefa-identity.yaml"),
    "Center": Path("ultralytics/cfg/models/oefa/yolo11n-oefa-center-only.yaml"),
    "Boundary": Path("ultralytics/cfg/models/oefa/yolo11n-oefa-boundary-only.yaml"),
    "M1": Path("ultralytics/cfg/models/oefa/yolo11n-oefa-m1.yaml"),
}


def cuda_profile(model, image, warmup, runs):
    if not torch.cuda.is_available():
        return None, None, {}
    model, image = model.cuda().eval(), image.cuda()
    head = model.model[-1]
    categories = {}
    if isinstance(head, OEFAM1Detect):
        if head.evidence_predictor is not None:
            categories["evidence"] = [head.evidence_predictor]
        categories["top_down"] = [x for x in (head.sample5to4, head.sample4to3) if x is not None]
        categories["bottom_up"] = [x for x in (head.down3, head.down4) if isinstance(x, BoundaryPreservingDownsample)]
    with torch.inference_mode():
        for _ in range(warmup):
            model(image)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            model(image)
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end) / runs
        memory = torch.cuda.max_memory_allocated() / 2**20

        component = {}
        for name, modules in categories.items():
            pairs, handles = [], []
            for module in modules:

                def pre_hook(_module, _inputs, pairs=pairs):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    pairs.append([event, None])

                def post_hook(_module, _inputs, _output, pairs=pairs):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    pairs[-1][1] = event

                handles.extend((module.register_forward_pre_hook(pre_hook), module.register_forward_hook(post_hook)))
            for _ in range(runs):
                model(image)
            torch.cuda.synchronize()
            component[name] = sum(a.elapsed_time(b) for a, b in pairs) / runs
            for handle in handles:
                handle.remove()
    return latency, memory, component


def manual_macs(head: OEFAM1Detect, imgsz: int):
    sizes = (imgsz // 8, imgsz // 16, imgsz // 32)
    result = {"evidence": [0, 0, 0], "top_down": [0, 0], "bottom_up": [0, 0]}
    if head.evidence_predictor is not None:
        dim = head.evidence_predictor.projections[0].out_channels
        for i, (projection, size) in enumerate(zip(head.evidence_predictor.projections, sizes)):
            result["evidence"][i] = size * size * (projection.in_channels * dim + 9 * dim + 2 * dim)
    if head.enable_top_down:
        result["top_down"] = [
            sizes[1] ** 2 * head.sample5to4.offset_mixer.in_channels * 3 * head.sample5to4.k,
            sizes[0] ** 2 * head.sample4to3.offset_mixer.in_channels * 3 * head.sample4to3.k,
        ]
    if head.enable_bottom_up:
        for i, (module, size) in enumerate(zip((head.down3, head.down4), sizes[:2])):
            c = module.base.conv.in_channels
            hidden = module.geometry_projection.out_channels
            out_size = size // 2
            result["bottom_up"][i] = size**2 * c * hidden + out_size**2 * (
                9 * hidden + hidden * c + module.gate.in_channels
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    print("THOP counts MACs then reports 2 FLOPs/MAC; grid_sample/interpolation/softmax are not fully counted by THOP.")
    for name, path in MODELS.items():
        model = DetectionModel(path, verbose=False).eval()
        image = torch.randn(1, 3, args.imgsz, args.imgsz)
        params = sum(p.numel() for p in model.parameters())
        real = profile(model, inputs=(image,), verbose=False)[0] * 2 / 1e9
        summary = get_flops(model, args.imgsz) or real
        latency, memory, component = cuda_profile(model, image, args.warmup, args.runs)
        head = model.model[-1]
        macs = manual_macs(head, args.imgsz) if isinstance(head, OEFAM1Detect) else {}
        diagnostics = {}
        if isinstance(head, OEFAM1Detect) and head.last_evidence is not None:
            diagnostics["evidence"] = [
                (float(x.sigmoid().mean()), float(x.sigmoid().std())) for x in head.last_evidence["center_logits"]
            ]
            diagnostics["offset"] = [x.last_stats for x in (head.sample5to4, head.sample4to3) if x is not None]
            diagnostics["bottom_up"] = [
                x.last_stats for x in (head.down3, head.down4) if isinstance(x, BoundaryPreservingDownsample)
            ]
        print(
            f"{name}: params={params:,} summary={summary:.6f} real640={real:.6f} GFLOPs "
            f"latency_ms={latency if latency is not None else 'N/A'} peak_MiB={memory if memory is not None else 'N/A'} "
            f"component_ms={component} MACs={macs} diagnostics={diagnostics}"
        )


if __name__ == "__main__":
    main()
