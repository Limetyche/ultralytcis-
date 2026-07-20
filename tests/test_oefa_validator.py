"""End-to-end trainer validator smoke test for modular OEFA V2."""

from pathlib import Path

import numpy as np
from PIL import Image

from ultralytics import YOLO

M1 = "ultralytics/cfg/models/oefa/yolo11n-oefa-m1-v2.yaml"


def _tiny_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.full((64, 64, 3), 64 + i * 64, dtype=np.uint8)).save(
                root / "images" / split / f"{i}.jpg"
            )
            (root / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    data = root / "data.yaml"
    data.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames: [object]\n")
    return data


def test_oefa_v2_one_epoch_reaches_and_finishes_validator(tmp_path):
    result = YOLO(M1).train(
        data=_tiny_dataset(tmp_path / "data"),
        epochs=1,
        imgsz=64,
        batch=1,
        workers=0,
        device="cpu",
        pretrained=False,
        val=True,
        save=False,
        plots=False,
        project=tmp_path / "runs",
        name="validator-smoke",
        verbose=False,
    )
    assert result is not None
