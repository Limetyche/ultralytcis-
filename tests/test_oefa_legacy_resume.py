"""Legacy OEFA freeze checks; also executable against a real Identity last.pt."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules.oefa import OEFAM1Detect

IDENTITY = "ultralytics/cfg/models/oefa/yolo11n-oefa-identity.yaml"
M1 = "ultralytics/cfg/models/oefa/yolo11n-oefa-m1.yaml"


def test_legacy_yamls_build_and_forward():
    for yaml in (IDENTITY, M1):
        model = YOLO(yaml).model.eval()
        assert type(model.model[-1]) is OEFAM1Detect
        with torch.no_grad():
            model(torch.zeros(1, 3, 64, 64))


def check_checkpoint(weights: str, data: str | None = None, run_val: bool = False):
    raw = torch.load(weights, map_location="cpu", weights_only=False)
    checkpoint_model = raw.get("model") or raw.get("ema")
    checkpoint_keys = set(checkpoint_model.float().state_dict())
    loaded = YOLO(weights)
    assert type(loaded.model.model[-1]) is OEFAM1Detect, "Legacy checkpoint was routed to a non-legacy class"
    assert checkpoint_keys == set(loaded.model.state_dict()), "state-dict keys changed during checkpoint load"
    loaded.model.eval()(torch.zeros(1, 3, 64, 64, device=next(loaded.model.parameters()).device))
    if run_val:
        if not data:
            raise ValueError("--data is required with --val")
        loaded.val(data=data, imgsz=64, batch=1)
    print("legacy load, exact keys and forward: PASS")
    print("Resume with the unchanged run arguments: yolo detect train resume model=" + str(Path(weights).resolve()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data")
    p.add_argument("--val", action="store_true")
    a = p.parse_args()
    check_checkpoint(a.weights, a.data, a.val)
