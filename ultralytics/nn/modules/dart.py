# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Distribution-aware residual refinement for YOLO dense detection heads."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from .head import Detect

__all__ = ("DARTDetect", "DistributionResidualRefiner", "DistributionStateEncoder")


class DistributionStateEncoder(nn.Module):
    """Encode per-edge expectation, variance and entropy from DFL logits."""

    def __init__(self, reg_max: int, detach: bool = True, eps: float = 1e-6):
        super().__init__()
        if reg_max < 2:
            raise ValueError(f"Distribution states require reg_max >= 2, got {reg_max}")
        self.reg_max = int(reg_max)
        self.detach = bool(detach)
        self.eps = float(eps)
        self.register_buffer("bins", torch.arange(self.reg_max, dtype=torch.float32), persistent=False)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return 12 BCHW channels ordered as [mean, variance, entropy] for each of four edges."""
        b, c, h, w = logits.shape
        if c != 4 * self.reg_max:
            raise ValueError(f"Expected {4 * self.reg_max} regression channels, got {c}")
        source = logits.detach() if self.detach else logits
        # Compute statistics in FP32 for stable entropy/variance under autocast, then match feature dtype.
        p = source.float().view(b, 4, self.reg_max, h, w).softmax(2)
        bins = self.bins.view(1, 1, self.reg_max, 1, 1)
        mean_bins = (p * bins).sum(2)
        mean = mean_bins / (self.reg_max - 1)
        # The maximum variance of a variable bounded by [0, reg_max-1] is range^2 / 4.
        variance = (p * (bins - mean_bins.unsqueeze(2)).square()).sum(2)
        variance = variance * (4.0 / ((self.reg_max - 1) ** 2))
        entropy = -(p * p.clamp_min(self.eps).log()).sum(2) / math.log(self.reg_max)
        state = torch.stack((mean, variance, entropy), dim=2).flatten(1, 2)
        return state.to(dtype=logits.dtype)


class DistributionResidualRefiner(nn.Module):
    """Predict lightweight local residuals for one pyramid level's DFL logits."""

    def __init__(
        self,
        channels: int,
        reg_max: int,
        hidden_dim: int = 32,
        state_channels: int = 12,
        dilation: int = 1,
    ):
        super().__init__()
        hidden_dim, state_channels = int(hidden_dim), int(state_channels)
        fused_channels = hidden_dim + state_channels
        self.state_channels = state_channels
        self.feature_projection = nn.Conv2d(channels, hidden_dim, 1, bias=False)
        self.local = nn.Sequential(
            nn.Conv2d(
                fused_channels,
                fused_channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=fused_channels,
                bias=False,
            ),
            nn.BatchNorm2d(fused_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(fused_channels, hidden_dim, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Conv2d(hidden_dim, 4 * reg_max, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.feature_projection(feature)
        if self.state_channels:
            if state is None:
                raise ValueError("Distribution state is required by this refiner")
            projected = torch.cat((projected, state), 1)
        elif state is not None:
            raise ValueError("Feature-only refiner must not receive distribution state")
        return self.output(self.local(projected))


class DARTDetect(Detect):
    """Detect head that applies one distribution-state-conditioned residual refinement step."""

    def __init__(
        self,
        nc: int = 80,
        hidden_dim: int = 32,
        use_distribution_state: bool = True,
        scale_adaptive: bool = True,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        if end2end:
            raise ValueError("DARTDetect M1 currently supports the standard one-to-many Detect path only")
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        if len(ch) != 3:
            raise ValueError(f"DARTDetect M1 expects P3/P4/P5 features, got {len(ch)} levels")
        self.hidden_dim = int(hidden_dim)
        self.use_distribution_state = bool(use_distribution_state)
        self.scale_adaptive = bool(scale_adaptive)
        self.state_encoder = DistributionStateEncoder(self.reg_max, detach=True) if self.use_distribution_state else None
        # P3/P4 retain dense nearest-neighbour sampling. P5 uses dilation 2 to expose uncertain large-object
        # edges to wider context at negligible parameter cost; the shared-scale control uses dilation 1 everywhere.
        dilations: Sequence[int] = (1, 1, 2) if self.scale_adaptive else (1, 1, 1)
        state_channels = 12 if self.use_distribution_state else 0
        self.refiners = nn.ModuleList(
            DistributionResidualRefiner(c, self.reg_max, self.hidden_dim, state_channels, dilation)
            for c, dilation in zip(ch, dilations)
        )
        # One learnable gain per pyramid level and box edge; non-zero initialization keeps output-conv gradients open.
        self.refine_gain = nn.Parameter(torch.full((self.nl, 4), 0.1))
        self.return_diagnostics = False
        self.last_diagnostics: tuple[dict[str, torch.Tensor], ...] | None = None

    def refine_boxes(
        self, x: list[torch.Tensor], box_head: nn.ModuleList | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor | None]]:
        """Return refined logits, initial logits and detached states for tests/profiling."""
        box_head = self.cv2 if box_head is None else box_head
        initial = [box_head[i](x[i]) for i in range(self.nl)]
        refined, states = [], []
        for i, (z0, feature, refiner) in enumerate(zip(initial, x, self.refiners)):
            state = self.state_encoder(z0) if self.state_encoder is not None else None
            delta = refiner(feature, state)
            gain = self.refine_gain[i].repeat_interleave(self.reg_max).view(1, -1, 1, 1)
            if torch.is_grad_enabled() or self.return_diagnostics:
                refined.append(z0 + gain * delta)
            else:
                # In inference there is no graph that needs z0 or delta, so reuse both buffers to reduce peak memory.
                refined.append(z0.add_(delta.mul_(gain)))
            states.append(state)
        return refined, initial, states

    def forward_head(
        self, x: list[torch.Tensor], box_head: nn.Module = None, cls_head: nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Preserve Detect's boxes/scores/feats protocol while replacing boxes with refined logits."""
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        refined, initial, states = self.refine_boxes(x, box_head)
        boxes = torch.cat([z.view(bs, 4 * self.reg_max, -1) for z in refined], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        if self.return_diagnostics:
            self.last_diagnostics = tuple(
                {
                    "initial": z0.detach(),
                    "refined": z1.detach(),
                    "delta": (z1 - z0).detach(),
                    **({"state": state.detach()} if state is not None else {}),
                }
                for z0, z1, state in zip(initial, refined, states)
            )
        else:
            self.last_diagnostics = None
        return dict(boxes=boxes, scores=scores, feats=x)
