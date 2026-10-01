# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Fair-neck, protocol, gradient, and compute tests for DASH-M1H."""

from pathlib import Path

import pytest
import torch
import yaml
from thop import profile

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.block import C2f, DSC3k2
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, get_num_params

ROOT = Path("ultralytics/cfg/models/dash")
H1 = ROOT / "yolov8n-hybrid-neck.yaml"
M1H = ROOT / "yolov8n-dash-m1h.yaml"
M2 = ROOT / "yolov8n-dash-m2lite.yaml"
B0 = Path("ultralytics/cfg/models/v8/yolov8.yaml")


def test_m1h_yaml_neck_pair_and_protocol():
    for path in (H1, M1H):
        config = yaml.safe_load(path.read_text())
        assert config["scale"] == "n" and config["detect_legacy"] is True
    h1, m1h = (DetectionModel(path, verbose=False) for path in (H1, M1H))
    for model in (h1, m1h):
        assert isinstance(model.model[12], C2f) and isinstance(model.model[15], C2f)
        assert isinstance(model.model[18], DSC3k2) and isinstance(model.model[21], DSC3k2)
        assert model.stride.tolist() == [8.0, 16.0, 32.0]
        assert type(model.model[-1].cv3[0][0]).__name__ == "Conv"

    model = m1h.train()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out["boxes"].shape == (1, 64, 336) and out["scores"].shape == (1, 80, 336)
    loss = out["boxes"].mean() + out["scores"].mean()
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        decoded, raw = model(x)
    assert decoded.shape == (1, 84, 336) and set(raw) == {"boxes", "scores", "feats"}


def test_m2lite_train_only_affinity_protocol_and_warmup():
    model = DetectionModel(M2, verbose=False)
    model.args = get_cfg()
    model.train()
    x = torch.randn(2, 3, 128, 128)
    batch = {
        "img": x,
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.4, 0.4, 0.2, 0.2]]),
    }
    preds = model(x)
    assert "dash_routing" in preds
    model.loss_epoch = 2
    loss0, _ = model.loss(batch, preds)
    assert model.criterion.last_relation_stats["warmup"] == 0.0
    model.loss_epoch = 15
    model.criterion.loss_epoch = 15
    loss1, _ = model.loss(batch, model(x))
    assert model.criterion.last_relation_stats["warmup"] == 1.0
    assert torch.isfinite(loss0) and torch.isfinite(loss1)
    loss1.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        decoded, raw = model(x)
    assert decoded.shape == (2, 84, 336) and "dash_routing" not in raw


@pytest.mark.slow
def test_m1h_640_compute_budget():
    baseline = DetectionModel(B0, verbose=False).eval()
    x = torch.randn(1, 3, 640, 640)
    for path in (H1, M1H):
        model = DetectionModel(path, verbose=False).eval()
        with torch.no_grad():
            decoded = model(x)[0]
        assert decoded.shape == (1, 84, 8400)
        assert get_num_params(model) <= get_num_params(baseline)
        assert get_flops(model, 640) <= get_flops(baseline, 640)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        base_macs, _ = profile(baseline, inputs=(x,), verbose=False)
        assert macs <= base_macs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_m1h_cuda_amp_backward():
    model = DetectionModel(M1H, verbose=False).cuda().train()
    x = torch.randn(1, 3, 640, 640, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x)
        loss = out["boxes"].float().mean() + out["scores"].float().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
