# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Protocol, independence, coverage, and compute-budget tests for DASH-M1E."""

from pathlib import Path

import pytest
import torch
import yaml
from thop import profile

from ultralytics.nn.modules.dash import DASHRoutingBlockEfficient
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, get_num_params


ROOT = Path("ultralytics/cfg/models")
B0 = ROOT / "v8/yolov8.yaml"
B1 = ROOT / "dash/yolov8n-dash-m1e-neck.yaml"
M1 = ROOT / "dash/yolov8n-dash-m1.yaml"
M1E = ROOT / "dash/yolov8n-dash-m1e.yaml"


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate keys."""


def _unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def test_dash_m1e_yaml_build_protocol_and_legacy_lock():
    for path in (B1, M1, M1E):
        config = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
        assert config["scale"] == "n"
        assert config["detect_legacy"] is True
    models = [DetectionModel(path, verbose=False) for path in (B0, B1, M1, M1E)]
    assert all(model.stride.tolist() == [8.0, 16.0, 32.0] for model in models)
    # A legacy head begins with a standard Conv; modern heads begin with a depthwise/pointwise Sequential.
    assert all(type(model.model[-1].cv3[0][0]).__name__ == "Conv" for model in models)

    model = models[-1].train()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out["boxes"].shape == (1, 64, 336)
    assert out["scores"].shape == (1, 80, 336)
    loss = out["boxes"].float().mean() + out["scores"].float().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        decoded, raw = model(x)
    assert decoded.shape == (1, 84, 336)
    assert set(raw) == {"boxes", "scores", "feats"}


def test_dash_m1e_router_independence_and_dense_coverage():
    block = DASHRoutingBlockEfficient((64, 128, 256))
    feats = [torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8), torch.randn(1, 256, 4, 4)]
    cls, reg = block(feats)
    assert block.semantic_router.query.weight.data_ptr() != block.geometry_router.query.weight.data_ptr()
    assert block.semantic_router.prototypes.data_ptr() != block.geometry_router.prototypes.data_ptr()
    assert block.semantic_readout.query.weight.data_ptr() != block.geometry_readout.query.weight.data_ptr()
    assert block.semantic_router.value.weight.data_ptr() != block.geometry_router.value.weight.data_ptr()
    assert len({p.data_ptr() for p in block.cls_gamma}) == 3
    assert len({p.data_ptr() for p in block.reg_gamma}) == 3
    assert block.last_stats["anchor_tokens"] == 152
    for name in ("cls_readout", "reg_readout"):
        assert all(float((weights.sum(-1) > 0).float().mean()) == 1.0 for weights in block.last_stats[name])
    assert all(float(value) > 0 for value in block.last_stats["cls_delta_abs"])
    assert all(float(value) > 0 for value in block.last_stats["reg_delta_abs"])
    assert all(a.shape == b.shape for a, b in zip(cls, feats))
    assert all(a.shape == b.shape for a, b in zip(reg, feats))


@pytest.mark.slow
def test_dash_m1e_640_forward_backward_and_compute_budget():
    baseline = DetectionModel(B0, verbose=False).eval()
    model = DetectionModel(M1E, verbose=False).train()
    x = torch.randn(1, 3, 640, 640)
    out = model(x)
    assert out["boxes"].shape == (1, 64, 8400)
    assert out["scores"].shape == (1, 80, 8400)
    loss = out["boxes"].float().mean() + out["scores"].float().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    model.eval()
    with torch.no_grad():
        decoded, _ = model(x)
    assert decoded.shape == (1, 84, 8400)
    assert get_num_params(model) <= int(get_num_params(baseline) * 1.01)
    assert get_flops(model, 640) <= 8.86
    real_macs, _ = profile(model, inputs=(x,), verbose=False)
    assert 2 * real_macs / 1e9 <= 8.86


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dash_m1e_cuda_amp_backward():
    model = DetectionModel(M1E, verbose=False).cuda().train()
    x = torch.randn(1, 3, 640, 640, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x)
        loss = out["boxes"].float().mean() + out["scores"].float().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
