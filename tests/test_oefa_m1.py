"""Correctness, isolation, stability and budget tests for OEFA-M1."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from thop import profile

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.oefa import (
    BoundaryPreservingDownsample,
    EvidenceGuidedSampler,
    EvidenceTargetGenerator,
    OEFAM1Detect,
)
from ultralytics.nn.tasks import DetectionModel

BASE = Path("ultralytics/cfg/models/11/yolo11.yaml")
ROOT = Path("ultralytics/cfg/models/oefa")
PATHS = {
    "identity": ROOT / "yolo11n-oefa-identity.yaml",
    "center": ROOT / "yolo11n-oefa-center-only.yaml",
    "boundary": ROOT / "yolo11n-oefa-boundary-only.yaml",
    "m1": ROOT / "yolo11n-oefa-m1.yaml",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that fails on duplicate keys."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def build(path, nc=3):
    model = DetectionModel(path, nc=nc, verbose=False)
    model.args = get_cfg()
    return model


def batch(size=128, batch_size=2, empty=False, device="cpu"):
    if empty:
        idx = torch.empty(0, device=device)
        cls = torch.empty(0, 1, device=device)
        boxes = torch.empty(0, 4, device=device)
    else:
        idx = torch.arange(batch_size, device=device)
        cls = torch.arange(batch_size, device=device).remainder(3).view(-1, 1).float()
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=device).repeat(batch_size, 1)
    return {
        "img": torch.rand(batch_size, 3, size, size, device=device),
        "batch_idx": idx,
        "cls": cls,
        "bboxes": boxes,
    }


def test_strict_yaml_builds_protocol_and_switches():
    baseline = build(BASE)
    assert baseline.stride.tolist() == [8.0, 16.0, 32.0]
    for name, path in PATHS.items():
        cfg = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
        assert cfg["scale"] == "n" and cfg["detect_legacy"] is False
        model = build(path)
        head = model.model[-1]
        assert isinstance(head, OEFAM1Detect) and head.legacy is False
        assert model.stride.tolist() == [8.0, 16.0, 32.0]
        assert head.enable_top_down is (name in {"center", "m1"})
        assert head.enable_bottom_up is (name in {"boundary", "m1"})
        model.train()
        output = model(torch.randn(1, 3, 128, 128))
        assert set(output) == ({"boxes", "scores", "feats"} if name == "identity" else {"boxes", "scores", "feats", "evidence"})
        model.eval()
        with torch.no_grad():
            decoded, raw = model(torch.randn(1, 3, 160, 160))
        assert decoded.shape == (1, 7, 525)
        assert set(raw) == {"boxes", "scores", "feats"}


@pytest.mark.parametrize("batch_size,size", [(1, 128), (2, 160)])
def test_dynamic_fp32_forward_backward(batch_size, size):
    model = build(PATHS["m1"]).train()
    output = model(torch.randn(batch_size, 3, size, size))
    loss = output["boxes"].mean() + output["scores"].mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_standard_and_evidence_loss_gradients_and_empty_gt():
    for empty in (False, True):
        model = build(PATHS["m1"]).train()
        total, items = model(batch(empty=empty))
        total.backward()
        assert torch.isfinite(total) and torch.isfinite(items).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if not empty:
            head = model.model[-1]
            assert head.sample5to4.offset_mixer.weight.grad.abs().sum() > 0
            assert head.evidence_predictor.projections[0].weight.grad.abs().sum() > 0
            assert head.down3.geometry[-1].weight.grad.abs().sum() > 0
            assert model.model[0].conv.weight.grad.abs().sum() > 0
            assert head.cv2[0][-1].weight.grad.abs().sum() > 0
            assert model.criterion.last_evidence_stats["evidence/total_loss"] > 0


def test_identity_has_standard_loss_and_exact_parameter_budget():
    baseline, identity = build(BASE), build(PATHS["identity"])
    assert type(identity.init_criterion()).__name__ == "v8DetectionLoss"
    assert identity.model[-1].evidence_predictor is None
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in identity.parameters())
    x = torch.randn(1, 3, 128, 128)
    output = identity.train()(x)
    output["boxes"].mean().backward()
    assert identity.model[-1].cv2[0][-1].weight.grad is not None


def test_identity_can_be_weight_mapped_to_exact_standard_outputs():
    baseline, identity = build(BASE), build(PATHS["identity"])
    for index in range(11):
        identity.model[index].load_state_dict(baseline.model[index].state_dict())
    head, standard_head = identity.model[-1], baseline.model[23]
    mapping = (
        (head.fuse_td4, baseline.model[13]),
        (head.fuse_td3, baseline.model[16]),
        (head.down3, baseline.model[17]),
        (head.fuse_bu4, baseline.model[19]),
        (head.down4, baseline.model[20]),
        (head.fuse_bu5, baseline.model[22]),
    )
    for target, source in mapping:
        target.load_state_dict(source.state_dict())
    head.cv2.load_state_dict(standard_head.cv2.state_dict())
    head.cv3.load_state_dict(standard_head.cv3.state_dict())
    head.dfl.load_state_dict(standard_head.dfl.state_dict())
    baseline.eval()
    identity.eval()
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        decoded_base, raw_base = baseline(image)
        decoded_identity, raw_identity = identity(image)
    assert torch.equal(decoded_base, decoded_identity)
    assert torch.equal(raw_base["boxes"], raw_identity["boxes"])
    assert torch.equal(raw_base["scores"], raw_identity["scores"])


def test_evidence_targets_ranges_overlap_and_scale_weights():
    generator = EvidenceTargetGenerator()
    boxes = torch.tensor([[0.3, 0.3, 0.08, 0.08], [0.32, 0.32, 0.4, 0.3], [0.7, 0.7, 0.8, 0.8]])
    centers, boundaries, _ = generator(torch.tensor([0, 0, 0]), boxes, 1, ((80, 80), (40, 40), (20, 20)), (640, 640))
    for target in centers + boundaries:
        assert torch.isfinite(target).all() and target.min() >= 0 and target.max() <= 1
    weights = generator.level_weights(torch.tensor([16.0, 96.0, 300.0]))
    assert weights.argmax(1).tolist() == [0, 1, 2]
    assert not torch.allclose(weights[:, 0], weights[:, 1])


def test_sampling_initialization_is_baseline_equivalent():
    source = torch.randn(2, 32, 8, 8)
    lateral = torch.randn(2, 16, 16, 16)
    sampler = EvidenceGuidedSampler(32, 16)
    dynamic = sampler(source, lateral, torch.randn(2, 1, 16, 16))
    assert torch.allclose(dynamic, F.interpolate(source, (16, 16), mode="nearest"), atol=1e-6)
    down = BoundaryPreservingDownsample(16, use_center=True).eval()
    x = torch.randn(2, 16, 16, 16)
    boundary = torch.randn(2, 1, 16, 16)
    center = torch.randn_like(boundary)
    with torch.no_grad():
        assert torch.equal(down(x, boundary, center), down.base(x))
    assert float(sampler.alpha.detach()) != 0 and float(down.alpha.detach()) != 0


@pytest.mark.slow
def test_640_fp32_forward_backward():
    model = build(PATHS["m1"], nc=1).train()
    output = model(torch.randn(1, 3, 640, 640))
    assert output["boxes"].shape == (1, 64, 8400)
    (output["boxes"].mean() + output["scores"].mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_parameter_and_real_thop_budgets():
    baseline = build(BASE).eval()
    image = torch.randn(1, 3, 640, 640)
    base_params = sum(p.numel() for p in baseline.parameters())
    base_flops = profile(baseline, inputs=(image,), verbose=False)[0]
    for path in PATHS.values():
        model = build(path).eval()
        assert sum(p.numel() for p in model.parameters()) / base_params <= 1.10
        assert profile(model, inputs=(image,), verbose=False)[0] / base_flops <= 1.12


def test_standard_sources_are_not_modified_for_oefa():
    assert "OEFA" not in Path("ultralytics/nn/modules/head.py").read_text()
    assert "OEFA" not in Path("ultralytics/utils/tal.py").read_text()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_640_standard_and_evidence_loss_backward():
    model = build(PATHS["m1"], nc=1).cuda().train()
    data = batch(640, 1, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        total, items = model(data)
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(items).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
