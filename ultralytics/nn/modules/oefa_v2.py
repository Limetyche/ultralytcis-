# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Modular, compile-friendly OEFA-M1 V2 components (never a Detect subclass)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = (
    "OEFAEvidencePredictorV2",
    "OEFAGuidedSamplerV2",
    "OEFABoundaryDownsampleV2",
    "EvidenceTargetGeneratorV2",
)


class _GridCache:
    """Small non-persistent LRU cache keyed by device, dtype and spatial shape."""

    def __init__(self, limit: int = 8):
        self.limit, self.data = limit, OrderedDict()

    def get(self, ref: torch.Tensor, h: int, w: int, centers: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        key = (ref.device.type, ref.device.index, ref.dtype, h, w, centers)
        value = self.data.pop(key, None)
        if value is None:
            shift = 0.5 if centers else 0.0
            yy, xx = torch.meshgrid(
                torch.arange(h, device=ref.device, dtype=ref.dtype) + shift,
                torch.arange(w, device=ref.device, dtype=ref.dtype) + shift,
                indexing="ij",
            )
            value = (yy, xx)
        self.data[key] = value
        while len(self.data) > self.limit:
            self.data.popitem(last=False)
        return value


class OEFAEvidencePredictorV2(nn.Module):
    """Predict center/boundary logits from P3/P4/P5 and return an explicit graph bundle."""

    def __init__(self, channels: Sequence[int], evidence_dim: int = 24, shared: bool = True, debug_stats: bool = False):
        super().__init__()
        self.shared, self.debug_stats = bool(shared), bool(debug_stats)
        self.projections = nn.ModuleList(nn.Conv2d(c, evidence_dim, 1, bias=False) for c in channels)
        make_tower = lambda: nn.Sequential(
            nn.Conv2d(evidence_dim, evidence_dim, 3, 1, 1, groups=evidence_dim, bias=False),
            nn.BatchNorm2d(evidence_dim), nn.SiLU(inplace=True), nn.Conv2d(evidence_dim, 2, 1)
        )
        self.predictor = make_tower() if self.shared else nn.ModuleList(make_tower() for _ in channels)

    def forward(self, features: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        logits = [
            (self.predictor if self.shared else self.predictor[i])(projection(feature))
            for i, (projection, feature) in enumerate(zip(self.projections, features))
        ]
        return {"center_logits": [z[:, :1] for z in logits], "boundary_logits": [z[:, 1:] for z in logits]}


class OEFAGuidedSamplerV2(nn.Module):
    """Mathematically equivalent K-point residual sampler with cached normalized grids."""

    def __init__(self, source_channels: int, lateral_channels: int, evidence_level: int, k: int = 4,
                 max_range: float = 1.5, detach_evidence: bool = False, debug_stats: bool = False):
        super().__init__()
        self.evidence_level, self.k = int(evidence_level), int(k)
        self.max_range, self.detach_evidence, self.debug_stats = float(max_range), bool(detach_evidence), bool(debug_stats)
        self.offset_mixer = nn.Conv2d(source_channels + lateral_channels + 1, 3 * self.k, 1)
        nn.init.zeros_(self.offset_mixer.weight)
        nn.init.zeros_(self.offset_mixer.bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self._grid_cache = _GridCache()
        self.last_stats: dict[str, float] = {}

    def forward(self, inputs: list) -> torch.Tensor:
        source, lateral, evidence = inputs
        center = evidence["center_logits"][self.evidence_level].sigmoid()
        control = center.detach() if self.detach_evidence else center
        baseline = F.interpolate(source, size=lateral.shape[-2:], mode="nearest")
        raw = self.offset_mixer(torch.cat((lateral, baseline, center), 1))
        raw_offset, raw_weight = raw.split((2 * self.k, self.k), 1)
        b, _, h, w = raw_offset.shape
        offset = raw_offset.view(b, self.k, 2, h, w).tanh() * self.max_range * control.unsqueeze(1)
        weight = raw_weight.softmax(1)
        yy, xx = self._grid_cache.get(source, h, w)
        base = torch.stack((2.0 * (xx + 0.5) / w - 1.0, 2.0 * (yy + 0.5) / h - 1.0), -1)
        delta = torch.stack((offset[:, :, 0] * (2.0 / w), offset[:, :, 1] * (2.0 / h)), -1)
        grid = (base.view(1, h, w, 1, 2) + delta.permute(0, 2, 3, 1, 4)).reshape(b, h, w * self.k, 2)
        sampled = F.grid_sample(baseline, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.view(b, baseline.shape[1], h, w, self.k).permute(0, 4, 1, 2, 3)
        dynamic = torch.sum(sampled * weight.unsqueeze(2), dim=1)
        if self.debug_stats:
            self.last_stats = {"offset_mean": float(offset.detach().abs().mean()), "offset_max": float(offset.detach().abs().max())}
        return baseline + self.alpha * (dynamic - baseline)


class OEFABoundaryDownsampleV2(nn.Module):
    """Boundary-preserving stride-2 branch, independent of the detector."""

    def __init__(self, channels: int, evidence_level: int, use_center: bool = True,
                 geometry_ratio: float = 0.25, debug_stats: bool = False):
        super().__init__()
        self.evidence_level, self.use_center, self.debug_stats = int(evidence_level), bool(use_center), bool(debug_stats)
        self.base = Conv(channels, channels, 3, 2)
        hidden = max(8, int(channels * geometry_ratio))
        self.geometry_projection = nn.Conv2d(channels, hidden, 1, bias=False)
        self.geometry = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 2, 1, groups=hidden, bias=False),
                                      nn.BatchNorm2d(hidden), nn.SiLU(inplace=True), nn.Conv2d(hidden, channels, 1))
        nn.init.zeros_(self.geometry[-1].weight); nn.init.zeros_(self.geometry[-1].bias)
        self.gate = nn.Conv2d(2 if self.use_center else 1, 1, 1)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.last_stats: dict[str, float] = {}

    def forward(self, inputs: list) -> torch.Tensor:
        x, evidence = inputs
        boundary = evidence["boundary_logits"][self.evidence_level]
        projected = self.geometry_projection(x)
        geometry = self.geometry((projected - F.avg_pool2d(projected, 3, 1, 1)) * boundary.sigmoid())
        gate_inputs = [F.avg_pool2d(boundary, 2, 2)]
        if self.use_center:
            gate_inputs.insert(0, F.avg_pool2d(evidence["center_logits"][self.evidence_level], 2, 2))
        gate = self.gate(torch.cat(gate_inputs, 1)).sigmoid()
        base = self.base(x)
        if self.debug_stats:
            self.last_stats = {"gate_mean": float(gate.detach().mean()), "gate_std": float(gate.detach().std(unbiased=False))}
        return base + self.alpha * gate * geometry


class EvidenceTargetGeneratorV2(nn.Module):
    """Chunked GPU-vectorized target generator preserving per-image maximum overlap semantics."""

    def __init__(self, canonical_scales=(32.0, 96.0, 224.0), tau=1.0, center_sigma_ratio=0.25,
                 boundary_sigma=1.5, min_sigma=1.0, boundary_expand=0.1, chunk_size=32):
        super().__init__()
        self.canonical_scales = tuple(float(x) for x in canonical_scales)
        self.tau, self.center_sigma_ratio = float(tau), float(center_sigma_ratio)
        self.boundary_sigma, self.min_sigma = float(boundary_sigma), float(min_sigma)
        self.boundary_expand, self.chunk_size = float(boundary_expand), int(chunk_size)
        self._grid_cache = _GridCache()

    def level_weights(self, sizes: torch.Tensor) -> torch.Tensor:
        distance = torch.log2(sizes.clamp_min(1e-6).unsqueeze(-1) / sizes.new_tensor(self.canonical_scales))
        return torch.exp(-distance.square() / (2 * self.tau**2))

    @torch.no_grad()
    def forward(self, batch_idx, boxes, batch_size, feature_shapes, image_size):
        ih, iw = image_size[-2], image_size[-1]
        weights = self.level_weights((boxes[:, 2] * iw * boxes[:, 3] * ih).clamp_min(0).sqrt())
        centers, boundaries = [], []
        for level, shape in enumerate(feature_shapes):
            h, w = shape[-2], shape[-1]
            center = boxes.new_zeros((batch_size, h * w)); boundary = boxes.new_zeros((batch_size, h * w))
            yy, xx = self._grid_cache.get(boxes, h, w, centers=True)
            flat_index = batch_idx.long().view(-1, 1) * (h * w) + torch.arange(h * w, device=boxes.device).view(1, -1)
            for start in range(0, boxes.shape[0], self.chunk_size):
                box = boxes[start:start + self.chunk_size]
                if box.numel() == 0: continue
                cx, cy, bw, bh = (box[:, 0] * w, box[:, 1] * h, box[:, 2] * w, box[:, 3] * h)
                sx = (bw * self.center_sigma_ratio).clamp_min(self.min_sigma); sy = (bh * self.center_sigma_ratio).clamp_min(self.min_sigma)
                cm = weights[start:start + len(box), level, None, None] * torch.exp(-0.5 * (((xx-cx[:,None,None])/sx[:,None,None]).square() + ((yy-cy[:,None,None])/sy[:,None,None]).square()))
                left, right, top, bottom = cx-bw/2, cx+bw/2, cy-bh/2, cy+bh/2
                edge = torch.minimum(torch.minimum((xx-left[:,None,None]).abs(), (xx-right[:,None,None]).abs()), torch.minimum((yy-top[:,None,None]).abs(), (yy-bottom[:,None,None]).abs()))
                region = (xx >= (left-bw*self.boundary_expand)[:,None,None]) & (xx <= (right+bw*self.boundary_expand)[:,None,None]) & (yy >= (top-bh*self.boundary_expand)[:,None,None]) & (yy <= (bottom+bh*self.boundary_expand)[:,None,None])
                bm = weights[start:start + len(box), level, None, None] * torch.exp(-edge.square()/(2*self.boundary_sigma**2)) * region
                idx = flat_index[start:start + len(box)]
                center.view(-1).scatter_reduce_(0, idx.reshape(-1), cm.reshape(-1), reduce="amax", include_self=True)
                boundary.view(-1).scatter_reduce_(0, idx.reshape(-1), bm.reshape(-1), reduce="amax", include_self=True)
            centers.append(center.view(batch_size,1,h,w).clamp_(0,1)); boundaries.append(boundary.view(batch_size,1,h,w).clamp_(0,1))
        return centers, boundaries
