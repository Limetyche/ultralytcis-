#!/usr/bin/env python3
"""Visualize OEFA evidence targets, predictions, offsets and gates without requiring training."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules.oefa import BoundaryPreservingDownsample, EvidenceTargetGenerator
from ultralytics.nn.tasks import DetectionModel


def load_model(path: str):
    return YOLO(path).model if path.endswith(".pt") else DetectionModel(path, verbose=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ultralytics/cfg/models/oefa/yolo11n-oefa-m1.yaml")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/coco.yaml")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="runs/oefa_evidence/random_init")
    args = parser.parse_args()
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 640, "rect": False, "batch": 1, "data": args.data})
    data = check_det_dataset(cfg.data)
    dataset = build_yolo_dataset(cfg, data["val"], 1, data, mode="val", stride=32)
    batch = dataset.collate_fn([dataset[args.index]])
    model = load_model(args.model).eval()
    head = model.model[-1]
    with torch.no_grad():
        model(batch["img"].float() / 255)
    evidence = head.last_evidence
    if evidence is None:
        raise ValueError("The selected model has no evidence predictor")
    generator = EvidenceTargetGenerator()
    centers, boundaries, stats = generator(
        batch["batch_idx"], batch["bboxes"], 1, evidence["feature_shapes"], batch["img"].shape[-2:]
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for level, name in enumerate(("P3", "P4", "P5")):
        predicted_center = evidence["center_logits"][level][0, 0].sigmoid().cpu()
        predicted_boundary = evidence["boundary_logits"][level][0, 0].sigmoid().cpu()
        maps = [centers[level][0, 0].cpu(), predicted_center, boundaries[level][0, 0].cpu(), predicted_boundary]
        titles = ["center target", "center prediction", "boundary target", "boundary prediction"]
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        for axis, value, title in zip(axes, maps, titles):
            axis.imshow(value.numpy(), cmap="magma", vmin=0, vmax=1)
            axis.set_title(title)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output / f"{name}_evidence.png", dpi=160)
        plt.close(fig)
        p = predicted_center.numpy().clip(1e-6, 1 - 1e-6)
        diagnostics.append(
            {
                "level": name,
                "evidence_entropy": float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).mean()),
                "target_stats": stats[level],
                "center_prediction_mean": float(predicted_center.mean()),
                "boundary_prediction_mean": float(predicted_boundary.mean()),
                "center_positive_mean": float(predicted_center[centers[level][0, 0].cpu() > 0.1].mean())
                if (centers[level] > 0.1).any()
                else 0.0,
                "center_negative_mean": float(predicted_center[centers[level][0, 0].cpu() <= 0.1].mean()),
                "boundary_positive_mean": float(predicted_boundary[boundaries[level][0, 0].cpu() > 0.1].mean())
                if (boundaries[level] > 0.1).any()
                else 0.0,
                "boundary_negative_mean": float(predicted_boundary[boundaries[level][0, 0].cpu() <= 0.1].mean()),
            }
        )
    offset_modules = [x for x in (head.sample5to4, head.sample4to3) if x is not None]
    gate_modules = [x for x in (head.down3, head.down4) if isinstance(x, BoundaryPreservingDownsample)]
    np.savez_compressed(
        output / "sampling_diagnostics.npz",
        **{f"offset_{i}": x.last_offset.cpu().numpy() for i, x in enumerate(offset_modules)},
        **{f"gate_{i}": x.last_gate.cpu().numpy() for i, x in enumerate(gate_modules)},
    )
    for i, module in enumerate(offset_modules):
        magnitude = module.last_offset[0].square().sum(1).sqrt().mean(0).cpu().numpy()
        plt.imsave(output / f"top_down_offset_{i}.png", magnitude, cmap="viridis")
    for i, module in enumerate(gate_modules):
        plt.imsave(output / f"bottom_up_gate_{i}.png", module.last_gate[0, 0].cpu().numpy(), cmap="viridis")
    # Deterministic small/medium/large and overlapping-box target audit, independent of checkpoint quality.
    example_boxes = torch.tensor(
        [[0.18, 0.20, 0.04, 0.04], [0.45, 0.45, 0.15, 0.12], [0.72, 0.68, 0.45, 0.40], [0.50, 0.48, 0.18, 0.16]]
    )
    example_center, example_boundary, _ = generator(
        torch.zeros(4, dtype=torch.long), example_boxes, 1, evidence["feature_shapes"], (640, 640)
    )
    fig, axes = plt.subplots(3, 2, figsize=(7, 10))
    for level, name in enumerate(("P3", "P4", "P5")):
        axes[level, 0].imshow(example_center[level][0, 0].numpy(), cmap="magma", vmin=0, vmax=1)
        axes[level, 0].set_title(f"{name} center: small/medium/large/overlap")
        axes[level, 1].imshow(example_boundary[level][0, 0].numpy(), cmap="magma", vmin=0, vmax=1)
        axes[level, 1].set_title(f"{name} boundary: small/medium/large/overlap")
        axes[level, 0].axis("off")
        axes[level, 1].axis("off")
    fig.tight_layout()
    fig.savefig(output / "target_scale_overlap_audit.png", dpi=160)
    plt.close(fig)
    for row in diagnostics:
        print(row)
    print("top_down", [x.last_stats for x in offset_modules])
    print("bottom_up", [x.last_stats for x in gate_modules])


if __name__ == "__main__":
    main()
