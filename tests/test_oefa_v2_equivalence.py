"""OEFA V2 structural and numerical regression tests."""

from copy import deepcopy

import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.oefa import EvidenceGuidedSampler, EvidenceTargetGenerator
from ultralytics.nn.modules.oefa_v2 import EvidenceTargetGeneratorV2, OEFAGuidedSamplerV2

IDENTITY = "ultralytics/cfg/models/oefa/yolo11n-oefa-identity-v2.yaml"
M1 = "ultralytics/cfg/models/oefa/yolo11n-oefa-m1-v2.yaml"


def make_small_detection_batch(batch_size=2):
    return {
        "img": torch.rand(batch_size, 3, 64, 64),
        "batch_idx": torch.arange(batch_size),
        "cls": torch.arange(batch_size).remainder(3).view(-1, 1).float(),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]).repeat(batch_size, 1),
    }


def test_v2_native_detect_and_identity_size():
    identity = YOLO(IDENTITY).model
    official = YOLO("ultralytics/cfg/models/11/yolo11.yaml").model
    m1 = YOLO(M1).model
    assert type(identity.model[-1]) is Detect and type(m1.model[-1]) is Detect
    assert identity.stride.tolist() == m1.stride.tolist() == [8.0, 16.0, 32.0]
    assert sum(p.numel() for p in identity.parameters()) == sum(p.numel() for p in official.parameters()) == 2_624_080
    assert not any("OEFA" in type(x).__name__ for x in identity.modules())


def test_identity_v2_exact_official_output():
    torch.manual_seed(7)
    a = YOLO(IDENTITY).model.eval()
    b = YOLO("ultralytics/cfg/models/11/yolo11.yaml").model.eval()
    b.load_state_dict(deepcopy(a.state_dict()), strict=True)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        ya, yb = a(x)[0], b(x)[0]
    torch.testing.assert_close(ya, yb, atol=0, rtol=0)


def test_sampler_v2_fp32_equivalence():
    torch.manual_seed(11)
    old = EvidenceGuidedSampler(16, 8, 4, 1.5, False).eval()
    new = OEFAGuidedSamplerV2(16, 8, 0, 4, 1.5, False).eval()
    new.offset_mixer.load_state_dict(old.offset_mixer.state_dict())
    new.alpha.data.copy_(old.alpha.data)
    source, lateral, logits = torch.randn(2, 16, 5, 7), torch.randn(2, 8, 10, 14), torch.randn(2, 1, 10, 14)
    expected = old(source, lateral, logits)
    actual = new([source, lateral, {"center_logits": [logits], "boundary_logits": [logits]}])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_vector_target_generator_fp32_equivalence():
    boxes = torch.tensor([[0.3, 0.4, 0.2, 0.15], [0.32, 0.41, 0.1, 0.2], [0.7, 0.7, 0.25, 0.3]], dtype=torch.float32)
    batch_idx = torch.tensor([0, 0, 1])
    shapes = [(2, 1, 16, 20), (2, 1, 8, 10), (2, 1, 4, 5)]
    old = EvidenceTargetGenerator()(batch_idx, boxes, 2, shapes, (128, 160))[:2]
    new = EvidenceTargetGeneratorV2(chunk_size=2)(batch_idx, boxes, 2, shapes, (128, 160))
    for old_group, new_group in zip(old, new):
        for a, b in zip(old_group, new_group):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_m1_v2_loss_and_finite_gradients():
    model = YOLO(M1).model.train()
    model.args = get_cfg()
    batch = make_small_detection_batch()
    loss, items = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(items).all()
    parameters = dict(model.named_parameters())
    required = (
        "model.11.predictor.3.weight",
        "model.12.offset_mixer.weight",
        "model.12.alpha",
        "model.18.geometry.3.weight",
        "model.18.gate.weight",
    )
    assert all(parameters[name].grad is not None and torch.isfinite(parameters[name].grad).all() for name in required)


def test_m1_v2_training_requires_evidence():
    model = YOLO(M1).model.train()
    model.args = get_cfg()
    batch = make_small_detection_batch()
    preds = model(batch["img"])
    preds.pop("evidence")
    with pytest.raises(RuntimeError, match="evidence is missing"):
        model.loss(batch, preds)


def test_m1_v2_eval_forward_and_loss_without_evidence():
    model = YOLO(M1).model
    model.args = get_cfg()
    batch = make_small_detection_batch()
    model.eval()
    with torch.no_grad():
        preds = model(batch["img"])
        assert isinstance(preds, tuple) and "evidence" not in preds[1]
        loss, items = model.loss(batch, preds)
    assert torch.isfinite(loss) and torch.isfinite(items).all() and items.numel() == 3


def test_m1_v2_train_to_validator_loss_lifecycle():
    model = YOLO(M1).model
    model.args = get_cfg()
    batch = make_small_detection_batch()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    model(batch)[0].backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        loss, items = model.loss(batch, model(batch["img"]))
    assert torch.isfinite(loss) and torch.isfinite(items).all() and items.shape == (3,)
