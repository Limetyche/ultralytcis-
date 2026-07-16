"""Focused correctness, stability and budget tests for DART-M1."""

from pathlib import Path

import pytest
import torch
import yaml

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.dart import DARTDetect, DistributionStateEncoder
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

ROOT = Path("ultralytics/cfg/models/dart")
PATHS = {
    "A1": ROOT / "yolov8n-dart-feature-only.yaml",
    "A2": ROOT / "yolov8n-dart-shared-scale.yaml",
    "M1": ROOT / "yolov8n-dart-m1.yaml",
}
BASE = Path("ultralytics/cfg/models/v8/yolov8.yaml")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _build(path, nc=80):
    model = DetectionModel(path, nc=nc, verbose=False)
    model.args = get_cfg()
    return model


def _detection_batch(size, batch_size=1, device="cpu"):
    return {
        "img": torch.rand(batch_size, 3, size, size, device=device),
        "batch_idx": torch.arange(batch_size, device=device),
        "cls": torch.zeros(batch_size, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], device=device).repeat(batch_size, 1),
    }


def test_yaml_builds_and_ablation_variables():
    for name, path in PATHS.items():
        cfg = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
        assert cfg["scale"] == "n" and cfg["detect_legacy"] is True
        assert cfg["head"][-1][2] == "DARTDetect"
        model = _build(path)
        head = model.model[-1]
        assert isinstance(head, DARTDetect)
        assert model.stride.tolist() == [8.0, 16.0, 32.0]
        assert head.use_distribution_state is (name != "A1")
        assert head.scale_adaptive is (name != "A2")
    # B0 remains independently constructible and untouched.
    assert _build(BASE).model[-1].__class__.__name__ == "Detect"


@pytest.mark.parametrize("batch_size,size", [(1, 128), (2, 160)])
def test_train_eval_dynamic_protocol_and_gradients(batch_size, size):
    model = _build(PATHS["M1"], nc=3).train()
    out = model(torch.randn(batch_size, 3, size, size))
    anchors = sum((size // stride) ** 2 for stride in (8, 16, 32))
    assert set(out) == {"boxes", "scores", "feats"}
    assert out["boxes"].shape == (batch_size, 64, anchors)
    assert out["scores"].shape == (batch_size, 3, anchors)
    out["boxes"].mean().backward()
    head = model.model[-1]
    assert head.refiners[0].output.weight.grad.abs().sum() > 0
    for tensor in (head.refiners[0].feature_projection.weight.grad, head.cv2[0][-1].weight.grad):
        assert tensor is not None and torch.isfinite(tensor).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    model.eval()
    with torch.no_grad():
        decoded, raw = model(torch.randn(batch_size, 3, size, size))
    assert decoded.shape == (batch_size, 7, anchors)
    assert set(raw) == {"boxes", "scores", "feats"}


def test_initial_identity_state_ranges_and_parameter_independence():
    model = _build(PATHS["M1"]).train()
    head = model.model[-1]
    feats = [torch.randn(2, c, s, s) for c, s in zip((64, 128, 256), (16, 8, 4))]
    refined, initial, states = head.refine_boxes(feats)
    for z1, z0, state in zip(refined, initial, states):
        assert torch.equal(z1, z0)
        assert torch.isfinite(state).all()
        assert state[:, 0::3].min() >= 0 and state[:, 0::3].max() <= 1
        assert state[:, 1::3].min() >= 0 and state[:, 1::3].max() <= 1 + 1e-5
        assert state[:, 2::3].min() >= 0 and state[:, 2::3].max() <= 1 + 1e-5
    assert len({r.output.weight.data_ptr() for r in head.refiners}) == 3
    assert len({head.refine_gain[i].data_ptr() for i in range(3)}) == 3
    assert torch.all(head.refine_gain != 0)


def test_zero_init_opens_refiner_after_first_update():
    model = _build(PATHS["M1"], nc=1).train()
    head = model.model[-1]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.randn(1, 3, 128, 128))["boxes"].mean().backward()
    assert head.refiners[0].output.weight.grad.abs().sum() > 0
    assert torch.isfinite(head.refiners[0].feature_projection.weight.grad).all()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(torch.randn(1, 3, 128, 128))["boxes"].mean().backward()
    assert head.refiners[0].feature_projection.weight.grad.abs().sum() > 0
    assert head.refine_gain.grad.abs().sum() > 0


def test_feature_only_never_calls_state_encoder(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("feature-only path read distribution state")

    monkeypatch.setattr(DistributionStateEncoder, "forward", fail)
    model = _build(PATHS["A1"]).train()
    out = model(torch.randn(1, 3, 128, 128))
    assert torch.isfinite(out["boxes"]).all()


def test_standard_detection_loss_backward():
    model = _build(PATHS["M1"], nc=1).train()
    total, items = model(_detection_batch(128))
    total.sum().backward()
    assert torch.isfinite(total).all() and torch.isfinite(items).all()
    head = model.model[-1]
    assert head.refiners[0].output.weight.grad.abs().sum() > 0
    assert torch.isfinite(head.cv2[0][-1].weight.grad).all()


@pytest.mark.slow
def test_640_fp32_forward_backward():
    model = _build(PATHS["M1"], nc=1).train()
    out = model(torch.randn(1, 3, 640, 640))
    assert out["boxes"].shape == (1, 64, 8400)
    (out["boxes"].mean() + out["scores"].mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_parameter_and_summary_flop_budgets():
    baseline = _build(BASE)
    base_params = sum(p.numel() for p in baseline.parameters())
    base_flops = get_flops(baseline, 640)
    for path in PATHS.values():
        model = _build(path)
        assert sum(p.numel() for p in model.parameters()) / base_params <= 1.08
        flops = get_flops(model, 640)
        if base_flops and flops:
            assert flops / base_flops <= 1.10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_640_loss_backward():
    model = _build(PATHS["M1"], nc=1).cuda().train()
    batch = _detection_batch(640, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        total, items = model(batch)
    total.sum().backward()
    assert torch.isfinite(total).all() and torch.isfinite(items).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
