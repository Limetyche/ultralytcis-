# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Object-Evidence Flow Alignment M1 modules for YOLO11."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3k2
from .conv import Conv
from .head import Detect

__all__ = (
    "BoundaryPreservingDownsample",
    "EvidenceGuidedSampler",
    "EvidenceTargetGenerator",
    "OEFAM1Detect",
    "ObjectEvidencePredictor",
)


class ObjectEvidencePredictor(nn.Module):
    """Predict center and boundary evidence once per lateral pyramid level."""

    def __init__(self, channels: Sequence[int], evidence_dim: int = 24, shared: bool = True):
        super().__init__()
        self.shared = bool(shared)
        self.projections = nn.ModuleList(nn.Conv2d(c, evidence_dim, 1, bias=False) for c in channels)
        if self.shared:
            self.predictor = nn.Sequential(
                nn.Conv2d(evidence_dim, evidence_dim, 3, 1, 1, groups=evidence_dim, bias=False),
                nn.BatchNorm2d(evidence_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(evidence_dim, 2, 1),
            )
        else:
            self.predictor = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(evidence_dim, evidence_dim, 3, 1, 1, groups=evidence_dim, bias=False),
                    nn.BatchNorm2d(evidence_dim),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(evidence_dim, 2, 1),
                )
                for _ in channels
            )

    def forward(self, features: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        logits = []
        for i, (projection, feature) in enumerate(zip(self.projections, features)):
            tower = self.predictor if self.shared else self.predictor[i]
            logits.append(tower(projection(feature)))
        return [x[:, :1] for x in logits], [x[:, 1:] for x in logits]


class EvidenceGuidedSampler(nn.Module):
    """Center-evidence-guided K-point residual sampling on a standard nearest-upsampled feature."""

    def __init__(self, source_channels: int, lateral_channels: int, k: int = 4, max_range: float = 1.5, detach=False):
        super().__init__()
        self.k = int(k)
        self.max_range = float(max_range)
        self.detach_evidence = bool(detach)
        self.offset_mixer = nn.Conv2d(source_channels + lateral_channels + 1, 3 * self.k, 1)
        nn.init.zeros_(self.offset_mixer.weight)
        nn.init.zeros_(self.offset_mixer.bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.last_stats: dict[str, float] = {}
        self.last_offset: torch.Tensor | None = None

    def forward(self, source: torch.Tensor, lateral: torch.Tensor, center_logits: torch.Tensor) -> torch.Tensor:
        baseline = F.interpolate(source, size=lateral.shape[-2:], mode="nearest")
        center = center_logits.sigmoid()
        control = center.detach() if self.detach_evidence else center
        raw = self.offset_mixer(torch.cat((lateral, baseline, center), 1))
        raw_offset, raw_weight = raw[:, : 2 * self.k], raw[:, 2 * self.k :]
        b, _, h, w = raw_offset.shape
        offset = raw_offset.view(b, self.k, 2, h, w).tanh() * self.max_range * control.unsqueeze(1)
        weight = raw_weight.softmax(1)

        yy, xx = torch.meshgrid(
            torch.arange(h, device=source.device, dtype=source.dtype),
            torch.arange(w, device=source.device, dtype=source.dtype),
            indexing="ij",
        )
        base_x = 2.0 * (xx + 0.5) / w - 1.0
        base_y = 2.0 * (yy + 0.5) / h - 1.0
        base_grid = torch.stack((base_x, base_y), -1).view(1, h, w, 1, 2)
        normalized_offset = torch.stack((offset[:, :, 0] * (2.0 / w), offset[:, :, 1] * (2.0 / h)), -1)
        grid = base_grid + normalized_offset.permute(0, 2, 3, 1, 4)
        # Pack K samples along the grid width so one grid_sample call handles all points without repeating features.
        sampled = F.grid_sample(
            baseline, grid.reshape(b, h, w * self.k, 2), mode="bilinear", padding_mode="border", align_corners=False
        )
        sampled = sampled.view(b, baseline.shape[1], h, w, self.k).permute(0, 4, 1, 2, 3)
        dynamic = (sampled * weight.unsqueeze(2)).sum(1)
        self.last_offset = offset.detach()
        self.last_stats = {
            "offset_mean": float(offset.detach().abs().mean()),
            "offset_max": float(offset.detach().abs().max()),
            "alpha": float(self.alpha.detach()),
        }
        return baseline + self.alpha * (dynamic - baseline)


class BoundaryPreservingDownsample(nn.Module):
    """Standard stride-2 Conv plus a boundary-gated high-frequency geometry residual."""

    def __init__(self, channels: int, use_center: bool = True, geometry_ratio: float = 0.25):
        super().__init__()
        self.base = Conv(channels, channels, 3, 2)
        hidden = max(8, int(channels * geometry_ratio))
        self.geometry_projection = nn.Conv2d(channels, hidden, 1, bias=False)
        self.geometry = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 2, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.geometry[-1].weight)
        nn.init.zeros_(self.geometry[-1].bias)
        self.use_center = bool(use_center)
        self.gate = nn.Conv2d(2 if self.use_center else 1, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.last_gate: torch.Tensor | None = None
        self.last_stats: dict[str, float] = {}

    def forward(
        self, x: torch.Tensor, boundary_logits: torch.Tensor, center_logits: torch.Tensor | None = None
    ) -> torch.Tensor:
        base = self.base(x)
        projected = self.geometry_projection(x)
        high = projected - F.avg_pool2d(projected, 3, 1, 1)
        geometry = self.geometry(high * boundary_logits.sigmoid())
        evidence = [F.avg_pool2d(boundary_logits, 2, 2)]
        if self.use_center:
            if center_logits is None:
                raise ValueError("Center logits are required by the full OEFA bottom-up gate")
            evidence.insert(0, F.avg_pool2d(center_logits, 2, 2))
        gate = self.gate(torch.cat(evidence, 1)).sigmoid()
        self.last_gate = gate.detach()
        self.last_stats = {
            "gate_mean": float(gate.detach().mean()),
            "gate_std": float(gate.detach().std(unbiased=False)),
            "alpha": float(self.alpha.detach()),
            "residual_base_ratio": float(
                (self.alpha * gate * geometry).detach().abs().mean() / base.detach().abs().mean().clamp_min(1e-9)
            ),
        }
        return base + self.alpha * gate * geometry


class EvidenceTargetGenerator(nn.Module):
    """Generate scale-weighted soft center and boundary evidence targets from normalized xywh boxes."""

    def __init__(
        self,
        canonical_scales: Sequence[float] = (32.0, 96.0, 224.0),
        tau: float = 1.0,
        center_sigma_ratio: float = 0.25,
        boundary_sigma: float = 1.5,
        min_sigma: float = 1.0,
        boundary_expand: float = 0.1,
    ):
        super().__init__()
        self.canonical_scales = tuple(float(x) for x in canonical_scales)
        self.tau = float(tau)
        self.center_sigma_ratio = float(center_sigma_ratio)
        self.boundary_sigma = float(boundary_sigma)
        self.min_sigma = float(min_sigma)
        self.boundary_expand = float(boundary_expand)

    def level_weights(self, sizes: torch.Tensor) -> torch.Tensor:
        canonical = sizes.new_tensor(self.canonical_scales)
        distance = torch.log2(sizes.clamp_min(1e-6).unsqueeze(-1) / canonical)
        return torch.exp(-distance.square() / (2 * self.tau**2))

    @torch.no_grad()
    def forward(
        self,
        batch_idx: torch.Tensor,
        boxes: torch.Tensor,
        batch_size: int,
        feature_shapes: Sequence[Sequence[int]],
        image_size: Sequence[int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, float]]]:
        device, dtype = boxes.device, boxes.dtype
        ih, iw = int(image_size[0]), int(image_size[1])
        center_targets, boundary_targets, stats = [], [], []
        sizes = (boxes[:, 2] * iw * boxes[:, 3] * ih).clamp_min(0).sqrt()
        weights = self.level_weights(sizes)
        for level, shape in enumerate(feature_shapes):
            h, w = int(shape[-2]), int(shape[-1])
            center = torch.zeros(batch_size, 1, h, w, device=device, dtype=dtype)
            boundary = torch.zeros_like(center)
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype) + 0.5,
                torch.arange(w, device=device, dtype=dtype) + 0.5,
                indexing="ij",
            )
            for j, box in enumerate(boxes):
                bi = int(batch_idx[j])
                cx, cy, bw, bh = box
                cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                scale_weight = weights[j, level]
                sx = torch.clamp(bw * self.center_sigma_ratio, min=self.min_sigma)
                sy = torch.clamp(bh * self.center_sigma_ratio, min=self.min_sigma)
                center_map = scale_weight * torch.exp(-0.5 * (((xx - cx) / sx).square() + ((yy - cy) / sy).square()))

                left, right, top, bottom = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2
                edge_distance = torch.minimum(
                    torch.minimum((xx - left).abs(), (xx - right).abs()),
                    torch.minimum((yy - top).abs(), (yy - bottom).abs()),
                )
                ex, ey = bw * self.boundary_expand, bh * self.boundary_expand
                region = (xx >= left - ex) & (xx <= right + ex) & (yy >= top - ey) & (yy <= bottom + ey)
                boundary_map = scale_weight * torch.exp(-edge_distance.square() / (2 * self.boundary_sigma**2)) * region
                center[bi, 0] = torch.maximum(center[bi, 0], center_map)
                boundary[bi, 0] = torch.maximum(boundary[bi, 0], boundary_map)
            center_targets.append(center.clamp_(0, 1))
            boundary_targets.append(boundary.clamp_(0, 1))
            stats.append(
                {
                    "center_positive_ratio": float((center > 0.1).float().mean()),
                    "boundary_positive_ratio": float((boundary > 0.1).float().mean()),
                }
            )
        return center_targets, boundary_targets, stats


class OEFAM1Detect(Detect):
    """Self-contained YOLO11 PAN and Detect head with optional evidence-guided sampling."""

    def __init__(
        self,
        nc: int = 80,
        evidence_dim: int = 24,
        enable_center: bool = True,
        enable_boundary: bool = True,
        enable_top_down: bool = True,
        enable_bottom_up: bool = True,
        shared_predictor: bool = True,
        detach_evidence: bool = False,
        lambda_evidence: float = 0.25,
        sampling_k: int = 4,
        geometry_ratio: float = 0.25,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        if end2end:
            raise ValueError("OEFA-M1 supports the standard one-to-many YOLO11 Detect path only")
        if len(ch) != 3 or ch[0] != ch[1] or ch[2] != 2 * ch[1]:
            raise ValueError(f"OEFA-M1 expects YOLO11 P3/P4/P5 channel ratio C/C/2C, got {ch}")
        lateral = int(ch[0])
        detect_channels = (lateral // 2, lateral, lateral * 2)
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=detect_channels)
        self.enable_center = bool(enable_center)
        self.enable_boundary = bool(enable_boundary)
        self.enable_top_down = bool(enable_top_down)
        self.enable_bottom_up = bool(enable_bottom_up)
        self.lambda_evidence = float(lambda_evidence)
        self.has_evidence = self.enable_center or self.enable_boundary

        # Exact YOLO11 PAN fusion blocks for the scaled C/C/2C lateral channels.
        self.fuse_td4 = C3k2(lateral * 3, lateral, 1, False)
        self.fuse_td3 = C3k2(lateral * 2, lateral // 2, 1, False)
        self.down3 = (
            BoundaryPreservingDownsample(lateral // 2, use_center=self.enable_center, geometry_ratio=geometry_ratio)
            if self.enable_bottom_up
            else Conv(lateral // 2, lateral // 2, 3, 2)
        )
        self.fuse_bu4 = C3k2(lateral + lateral // 2, lateral, 1, False)
        self.down4 = (
            BoundaryPreservingDownsample(lateral, use_center=self.enable_center, geometry_ratio=geometry_ratio)
            if self.enable_bottom_up
            else Conv(lateral, lateral, 3, 2)
        )
        self.fuse_bu5 = C3k2(lateral * 3, lateral * 2, 1, True)

        if self.has_evidence:
            self.evidence_predictor = ObjectEvidencePredictor(ch, evidence_dim, shared_predictor)
        else:
            self.evidence_predictor = None
        if self.enable_top_down:
            self.sample5to4 = EvidenceGuidedSampler(ch[2], ch[1], sampling_k, 1.0, detach_evidence)
            self.sample4to3 = EvidenceGuidedSampler(lateral, ch[0], sampling_k, 1.5, detach_evidence)
        else:
            self.sample5to4 = self.sample4to3 = None
        self.last_evidence: dict | None = None

    def _neck(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], dict | None]:
        p3, p4, p5 = x
        evidence = None
        if self.evidence_predictor is not None:
            center, boundary = self.evidence_predictor(x)
            evidence = {"center_logits": center, "boundary_logits": boundary, "feature_shapes": [z.shape for z in x]}
        if self.enable_top_down:
            u4 = self.sample5to4(p5, p4, center[1])
        else:
            u4 = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        n4_td = self.fuse_td4(torch.cat((u4, p4), 1))
        if self.enable_top_down:
            u3 = self.sample4to3(n4_td, p3, center[0])
        else:
            u3 = F.interpolate(n4_td, size=p3.shape[-2:], mode="nearest")
        n3 = self.fuse_td3(torch.cat((u3, p3), 1))

        if self.enable_bottom_up:
            d4 = self.down3(n3, boundary[0], center[0] if self.enable_center else None)
        else:
            d4 = self.down3(n3)
        n4 = self.fuse_bu4(torch.cat((d4, n4_td), 1))
        if self.enable_bottom_up:
            d5 = self.down4(n4, boundary[1], center[1] if self.enable_center else None)
        else:
            d5 = self.down4(n4)
        n5 = self.fuse_bu5(torch.cat((d5, p5), 1))
        return [n3, n4, n5], evidence

    def forward(self, x: list[torch.Tensor]):
        neck, evidence = self._neck(x)
        self.last_evidence = (
            {
                "center_logits": [z.detach() for z in evidence["center_logits"]],
                "boundary_logits": [z.detach() for z in evidence["boundary_logits"]],
                "feature_shapes": evidence["feature_shapes"],
            }
            if evidence is not None
            else None
        )
        output = super().forward(neck)
        if self.training and evidence is not None:
            output["evidence"] = evidence
        return output
