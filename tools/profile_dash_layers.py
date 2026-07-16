#!/usr/bin/env python3
"""Profile B0/B1/M1/M1E per layer and audit DASH routing costs at a real input size."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from thop import profile

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, get_num_params


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "B0": ROOT / "ultralytics/cfg/models/v8/yolov8.yaml",
    "B1": ROOT / "ultralytics/cfg/models/dash/yolov8n-dash-m1e-neck.yaml",
    "M1": ROOT / "ultralytics/cfg/models/dash/yolov8n-dash-m1.yaml",
    "M1E": ROOT / "ultralytics/cfg/models/dash/yolov8n-dash-m1e.yaml",
    "H1": ROOT / "ultralytics/cfg/models/dash/yolov8n-hybrid-neck.yaml",
    "M1H": ROOT / "ultralytics/cfg/models/dash/yolov8n-dash-m1h.yaml",
}


def shape_summary(value):
    """Return a compact nested output-shape representation."""
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{shape_summary(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(shape_summary(v) for v in value) + "]"
    return type(value).__name__


def timed_forward(module, inputs, warmup: int, runs: int, device: torch.device) -> tuple[float, float | None]:
    """Measure latency and CUDA peak memory for a module call."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            module(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(runs):
            module(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    latency = (time.perf_counter() - start) * 1000 / runs
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    return latency, peak


def profile_model(name: str, path: Path, imgsz: int, warmup: int, runs: int, device: torch.device):
    """Profile one complete model and print its top-level layer table."""
    model = DetectionModel(path, verbose=False).to(device).eval()
    image = torch.zeros(1, 3, imgsz, imgsz, device=device)
    output_shapes = {}
    hooks = [
        layer.register_forward_hook(lambda _m, _i, out, index=i: output_shapes.__setitem__(index, shape_summary(out)))
        for i, layer in enumerate(model.model)
    ]
    macs, _, info = profile(model, inputs=(image,), verbose=False, ret_layer_info=True)
    for hook in hooks:
        hook.remove()

    layer_info = info["model"][2]
    rows = []
    for i, layer in enumerate(model.model):
        layer_macs = layer_info[str(i)][0]
        rows.append(
            {
                "index": i,
                "type": type(layer).__name__,
                "params": sum(p.numel() for p in layer.parameters()),
                "macs": layer_macs,
                "gflops": 2 * layer_macs / 1e9,
                "output": output_shapes.get(i, "?"),
            }
        )
    latency, peak = timed_forward(model, (image,), warmup, runs, device)
    print(f"\n## {name}: {path}")
    print("idx type                    params       MACs     GFLOPs output")
    for row in rows:
        print(
            f"{row['index']:>3} {row['type']:<22} {row['params']:>9,} "
            f"{row['macs'] / 1e6:>9.3f}M {row['gflops']:>8.4f} {row['output']}"
        )
    print("top10_by_real_640_MACs")
    for rank, row in enumerate(sorted(rows, key=lambda x: x["macs"], reverse=True)[:10], 1):
        print(
            f"{rank:>2}. layer={row['index']:<2} type={row['type']:<20} "
            f"params={row['params']:,} MACs={row['macs'] / 1e6:.3f}M GFLOPs={row['gflops']:.4f}"
        )

    head = model.model[-1]
    routing_macs = None
    routing_latency = routing_peak = None
    if hasattr(head, "routing"):
        shapes = [(64, imgsz // 8, imgsz // 8), (128, imgsz // 16, imgsz // 16), (256, imgsz // 32, imgsz // 32)]
        routing_macs = head.routing.theoretical_macs(shapes)
        routing_breakdown = head.routing.theoretical_mac_breakdown(shapes)
        routing_inputs = (
            [
                torch.zeros(1, channels, height, width, device=device)
                for channels, height, width in shapes
            ],
        )
        routing_latency, routing_peak = timed_forward(head.routing, routing_inputs, warmup, runs, device)
        routing_info = layer_info[str(len(model.model) - 1)][2]["routing"][2]
        print("routing_thop_submodules")
        for module_name, (module_macs, module_params, _) in sorted(
            routing_info.items(), key=lambda item: item[1][0], reverse=True
        ):
            print(
                f"  {module_name:<24} params={module_params:>8,.0f} "
                f"MACs={module_macs / 1e6:>9.3f}M GFLOPs={2 * module_macs / 1e9:.4f}"
            )
        print(f"routing_manual_MACs={routing_macs:,} ({routing_macs / 1e9:.6f} GMAC)")
        for operation, operation_macs in routing_breakdown.items():
            print(f"  theory/{operation:<25} {operation_macs:>12,} MACs")
        tokens = sum(h * w for h, w in head.routing.encoder.anchor_grid_sizes)
        dense_nodes = sum(h * w for _, h, w in shapes)
        edge_count = int(head.routing.enable_semantic) * head.routing.semantic_readout.num_edges + int(
            head.routing.enable_geometry
        ) * head.routing.geometry_readout.num_edges
        softmax_elements = (tokens + dense_nodes) * edge_count
        print(f"routing_softmax_elements={softmax_elements:,} (not MACs; FP32 exp/reduction/division)")
        print(f"routing_latency_ms={routing_latency:.4f} routing_peak_memory_mb={routing_peak}")

    result = {
        "name": name,
        "params": get_num_params(model),
        "summary_gflops": get_flops(model, imgsz),
        "real_640_gflops": 2 * macs / 1e9,
        "routing_macs": routing_macs,
        "latency_ms": latency,
        "peak_memory_mb": peak,
    }
    print("summary", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    results = [profile_model(n, MODELS[n], args.imgsz, args.warmup, args.runs, device) for n in args.models]
    print("\n## Comparison")
    print("name params summary_GFLOPs real640_GFLOPs routing_GMAC latency_ms peak_memory_mb")
    for result in results:
        routing_gmac = None if result["routing_macs"] is None else result["routing_macs"] / 1e9
        print(
            f"{result['name']} {result['params']} {result['summary_gflops']:.6f} "
            f"{result['real_640_gflops']:.6f} {routing_gmac} {result['latency_ms']:.4f} "
            f"{result['peak_memory_mb']}"
        )


if __name__ == "__main__":
    main()
