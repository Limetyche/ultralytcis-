# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Focused tests for DASH-M1."""

from pathlib import Path

import pytest
import torch
import yaml

from ultralytics.nn.modules.dash import DASHRoutingBlock
from ultralytics.nn.tasks import DetectionModel


YAML_PATH = Path("ultralytics/cfg/models/dash/yolov8n-dash-m1.yaml")


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def test_dash_yaml_and_protocol():
    config = yaml.load(YAML_PATH.read_text(), Loader=_UniqueKeyLoader)
    assert config["head"][-1][2] == "DASHDetect"
    model = DetectionModel(YAML_PATH, verbose=False)
    assert model.stride.tolist() == [8.0, 16.0, 32.0]
    model.train()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out["boxes"].shape == (1, 64, 336)
    assert out["scores"].shape == (1, 80, 336)
    loss = out["boxes"].float().mean() + out["scores"].float().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        decoded, raw = model(x)
    assert decoded.shape == (1, 84, 336)
    assert set(raw) == {"boxes", "scores", "feats"}


def test_dash_independence_coverage_and_switches():
    shapes = ((64, 16, 16), (128, 8, 8), (256, 4, 4))
    feats = [torch.randn(1, c, h, w) for c, h, w in shapes]
    block = DASHRoutingBlock((64, 128, 256), latent_dim=32, anchor_grid_sizes=((6, 6), (4, 4), (2, 2)))
    cls, reg = block(feats)
    assert block.semantic_router.query.weight.data_ptr() != block.geometry_router.query.weight.data_ptr()
    assert block.semantic_readout.query.weight.data_ptr() != block.geometry_readout.query.weight.data_ptr()
    assert len({x.data_ptr() for x in block.cls_gamma}) == 3
    assert len({x.data_ptr() for x in block.reg_gamma}) == 3
    assert all(float(x) > 0 for x in block.last_stats["cls_delta_abs"])
    assert all(float(x) > 0 for x in block.last_stats["reg_delta_abs"])
    assert all(a.shape == b.shape for a, b in zip(cls, feats))
    assert all(a.shape == b.shape for a, b in zip(reg, feats))

    semantic_off = DASHRoutingBlock((64, 128, 256), latent_dim=32, enable_semantic=False)
    cls_off, _ = semantic_off(feats)
    assert all(torch.equal(a, b) for a, b in zip(cls_off, feats))
    geometry_off = DASHRoutingBlock((64, 128, 256), latent_dim=32, enable_geometry=False)
    _, reg_off = geometry_off(feats)
    assert all(torch.equal(a, b) for a, b in zip(reg_off, feats))

    sparse = DASHRoutingBlock((64, 128, 256), latent_dim=32, dense_readout=False)
    sparse_cls, sparse_reg = sparse(feats)
    assert all(not torch.equal(a, b) for a, b in zip(sparse_cls, feats))
    assert all(not torch.equal(a, b) for a, b in zip(sparse_reg, feats))

    shared = DASHRoutingBlock(
        (64, 128, 256), latent_dim=32, enable_semantic=False, enable_geometry=True, share_hypergraph=True
    )
    _, shared_reg = shared(feats)
    assert all(not torch.equal(a, b) for a, b in zip(shared_reg, feats))


@pytest.mark.slow
def test_dash_640_dense_output_shape():
    model = DetectionModel(YAML_PATH, verbose=False).eval()
    with torch.no_grad():
        decoded, raw = model(torch.randn(1, 3, 640, 640))
    assert decoded.shape == (1, 84, 8400)
    assert raw["boxes"].shape == (1, 64, 8400)
    assert raw["scores"].shape == (1, 80, 8400)
    assert all(float(x) > 0 for x in model.model[-1].routing_stats["cls_delta_abs"])
    assert all(float(x) > 0 for x in model.model[-1].routing_stats["reg_delta_abs"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dash_cuda_amp_backward():
    model = DetectionModel(YAML_PATH, verbose=False).cuda().train()
    x = torch.randn(1, 3, 128, 128, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x)
        loss = out["boxes"].float().mean() + out["scores"].float().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
