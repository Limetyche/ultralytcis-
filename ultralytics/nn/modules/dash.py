# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DASH-M1: dual-path anchor hypergraphs with dense multi-scale readout."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import Detect


def _coordinate_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return cell-centre coordinates in [0, 1] with shape (H*W, 2)."""
    y = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
    x = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), -1).reshape(-1, 2)


class MultiScaleAnchorEncoder(nn.Module):
    """Encode independent pyramid levels and extract spatially covering anchor-token grids."""

    def __init__(self, ch: Sequence[int], latent_dim: int, anchor_grid_sizes: Sequence[Sequence[int]]):
        super().__init__()
        if len(ch) != 3 or len(anchor_grid_sizes) != 3:
            raise ValueError("DASH-M1 requires exactly three feature levels and three anchor grids")
        self.latent_dim = int(latent_dim)
        self.anchor_grid_sizes = tuple((int(g[0]), int(g[1])) for g in anchor_grid_sizes)
        self.input_proj = nn.ModuleList(nn.Conv2d(c, self.latent_dim, 1, bias=False) for c in ch)
        self.input_norm = nn.ModuleList(nn.GroupNorm(1, self.latent_dim) for _ in ch)
        self.scale_embedding = nn.Parameter(torch.zeros(3, self.latent_dim))
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(2, self.latent_dim), nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)
        )
        nn.init.normal_(self.scale_embedding, std=0.02)

    def forward(self, feats: list[torch.Tensor]) -> tuple:
        dense, anchor_tokens, anchor_boundary, anchor_xy, anchor_levels, actual_grid_sizes = [], [], [], [], [], []
        dense_xy, dense_levels = [], []
        for level, (feat, proj, norm, grid_size) in enumerate(
            zip(feats, self.input_proj, self.input_norm, self.anchor_grid_sizes)
        ):
            z = norm(proj(feat))
            b, _, h, w = z.shape
            xy = _coordinate_grid(h, w, z.device, z.dtype)
            pos = self.coordinate_mlp(xy).transpose(0, 1).reshape(1, self.latent_dim, h, w)
            z = z + pos + self.scale_embedding[level].view(1, -1, 1, 1)
            boundary = (z - F.avg_pool2d(z, 3, 1, 1)).abs()
            # Avoid artificial upsampling for tiny profiling/test inputs; this also keeps FLOP extrapolation sane.
            gh, gw = min(grid_size[0], h), min(grid_size[1], w)
            actual_grid = (gh, gw)
            pooled = F.adaptive_avg_pool2d(z, actual_grid)
            pooled_boundary = F.adaptive_avg_pool2d(boundary, actual_grid)
            axy = _coordinate_grid(gh, gw, z.device, z.dtype)

            dense.append(z)
            anchor_tokens.append(pooled.flatten(2).transpose(1, 2))
            anchor_boundary.append(pooled_boundary.flatten(2).transpose(1, 2))
            anchor_xy.append(axy)
            anchor_levels.append(torch.full((gh * gw,), level, device=z.device, dtype=torch.long))
            dense_xy.append(xy)
            dense_levels.append(torch.full((h * w,), level, device=z.device, dtype=torch.long))
            actual_grid_sizes.append(actual_grid)

        return (
            dense,
            torch.cat(anchor_tokens, 1),
            torch.cat(anchor_boundary, 1),
            torch.cat(anchor_xy, 0),
            torch.cat(anchor_levels, 0),
            dense_xy,
            dense_levels,
            actual_grid_sizes,
        )


class SemanticHypergraphRouter(nn.Module):
    """Build semantic/context hyperedges and aggregate anchor-token messages."""

    def __init__(self, dim: int, num_edges: int):
        super().__init__()
        self.num_edges = int(num_edges)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.context = nn.Linear(dim, dim, bias=False)
        self.prototypes = nn.Parameter(torch.empty(self.num_edges, dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.message = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.level_bias = nn.Parameter(torch.zeros(3, self.num_edges))
        nn.init.normal_(self.prototypes, std=dim**-0.5)

    def forward(self, tokens: torch.Tensor, levels: torch.Tensor, enable_cross_scale: bool = True) -> tuple:
        x = self.norm(tokens)
        global_context = x.mean(1, keepdim=True)
        q = self.query(x) + self.context(global_context)
        k = self.key(self.prototypes).unsqueeze(0)
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        logits = logits + self.level_bias[levels].float().unsqueeze(0)
        if not enable_cross_scale:
            edge_levels = torch.arange(self.num_edges, device=levels.device) % 3
            logits = logits.masked_fill(levels.view(1, -1, 1) != edge_levels.view(1, 1, -1), -20.0)
        incidence = torch.softmax(logits.clamp(-20.0, 20.0), -1)
        degree = incidence.sum(1).clamp_min(1e-4)
        edge_states = torch.bmm(incidence.transpose(1, 2), self.value(x).float()) / degree.unsqueeze(-1)
        edge_states = self.message(edge_states.to(tokens.dtype))
        return incidence, edge_states


class GeometryHypergraphRouter(nn.Module):
    """Build geometry-specific hyperedges from visual, coordinate, stride, and boundary descriptors."""

    def __init__(self, dim: int, num_edges: int):
        super().__init__()
        self.num_edges = int(num_edges)
        self.norm = nn.LayerNorm(dim)
        self.boundary_proj = nn.Linear(dim, dim, bias=False)
        self.coordinate_proj = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.query = nn.Linear(dim, dim, bias=False)
        self.prototypes = nn.Parameter(torch.empty(self.num_edges, dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.message = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.edge_centres = nn.Parameter(torch.rand(self.num_edges, 2))
        self.level_bias = nn.Parameter(torch.zeros(3, self.num_edges))
        self.log_distance_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.prototypes, std=dim**-0.5)

    def forward(
        self,
        tokens: torch.Tensor,
        boundary: torch.Tensor,
        xy: torch.Tensor,
        levels: torch.Tensor,
        enable_cross_scale: bool = True,
    ) -> tuple:
        descriptor = self.norm(tokens) + self.boundary_proj(boundary) + self.coordinate_proj(xy).unsqueeze(0)
        q = self.query(descriptor)
        k = self.key(self.prototypes).unsqueeze(0)
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        distance = torch.cdist(xy.float(), self.edge_centres.sigmoid().float(), p=1)
        logits = logits - self.log_distance_scale.float().exp().clamp(max=10.0) * distance.unsqueeze(0)
        logits = logits + self.level_bias[levels].float().unsqueeze(0)
        if not enable_cross_scale:
            edge_levels = torch.arange(self.num_edges, device=levels.device) % 3
            logits = logits.masked_fill(levels.view(1, -1, 1) != edge_levels.view(1, 1, -1), -20.0)
        incidence = torch.softmax(logits.clamp(-20.0, 20.0), -1)
        degree = incidence.sum(1).clamp_min(1e-4)
        edge_states = torch.bmm(incidence.transpose(1, 2), self.value(descriptor).float()) / degree.unsqueeze(-1)
        edge_states = self.message(edge_states.to(tokens.dtype))
        return incidence, edge_states, self.edge_centres.sigmoid()


class DenseHyperedgeReadout(nn.Module):
    """Read a small edge set back into every dense pyramid location."""

    def __init__(self, dim: int, num_edges: int, geometry: bool = False):
        super().__init__()
        self.num_edges = int(num_edges)
        self.geometry = geometry
        self.query = nn.Conv2d(dim, dim, 1, bias=False)
        self.edge_key = nn.Linear(dim, dim, bias=False)
        self.edge_value = nn.Linear(dim, dim, bias=False)
        self.level_bias = nn.Parameter(torch.zeros(3, self.num_edges))
        if geometry:
            self.log_distance_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        dense: list[torch.Tensor],
        edge_states: torch.Tensor,
        dense_xy: list[torch.Tensor],
        edge_xy: torch.Tensor | None = None,
        enable_cross_scale: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        keys = self.edge_key(edge_states).float()
        values = self.edge_value(edge_states).float()
        outputs, weights = [], []
        for level, (feat, xy) in enumerate(zip(dense, dense_xy)):
            b, c, h, w = feat.shape
            q = self.query(feat).flatten(2).transpose(1, 2).float()
            logits = torch.bmm(q, keys.transpose(1, 2)) / math.sqrt(c)
            logits = logits + self.level_bias[level].float().view(1, 1, -1)
            if not enable_cross_scale:
                edge_levels = torch.arange(self.num_edges, device=feat.device) % 3
                logits = logits.masked_fill(edge_levels.view(1, 1, -1) != level, -20.0)
            if self.geometry and edge_xy is not None:
                if edge_xy.ndim == 2:
                    distance = torch.cdist(xy.float(), edge_xy.float(), p=1).unsqueeze(0)
                else:
                    dense_coordinates = xy.float().unsqueeze(0).expand(edge_xy.shape[0], -1, -1)
                    distance = torch.cdist(dense_coordinates, edge_xy.float(), p=1)
                logits = logits - self.log_distance_scale.float().exp().clamp(max=10.0) * distance
            read_weights = torch.softmax(logits.clamp(-20.0, 20.0), -1)
            delta = torch.bmm(read_weights, values).to(feat.dtype)
            outputs.append(delta.transpose(1, 2).reshape(b, c, h, w).contiguous())
            weights.append(read_weights)
        return outputs, weights


class DASHRoutingBlock(nn.Module):
    """Coordinate anchor encoding, independent dual hypergraphs, and dense or anchor-grid readout."""

    def __init__(
        self,
        ch: Sequence[int],
        latent_dim: int = 48,
        semantic_edges: int = 12,
        geometry_edges: int = 8,
        anchor_grid_sizes: Sequence[Sequence[int]] = ((12, 12), (8, 8), (4, 4)),
        enable_semantic: bool = True,
        enable_geometry: bool = True,
        enable_cross_scale: bool = True,
        dense_readout: bool = True,
        gamma_init: float = 0.05,
        share_hypergraph: bool = False,
    ):
        super().__init__()
        self.enable_semantic = bool(enable_semantic)
        self.enable_geometry = bool(enable_geometry)
        self.enable_cross_scale = bool(enable_cross_scale)
        self.dense_readout = bool(dense_readout)
        self.share_hypergraph = bool(share_hypergraph)
        self.encoder = MultiScaleAnchorEncoder(ch, latent_dim, anchor_grid_sizes)
        self.semantic_router = (
            SemanticHypergraphRouter(latent_dim, semantic_edges)
            if self.enable_semantic or (self.share_hypergraph and self.enable_geometry)
            else None
        )
        geometry_router_edges = semantic_edges if self.share_hypergraph else geometry_edges
        self.geometry_router = (
            None
            if not self.enable_geometry or self.share_hypergraph
            else GeometryHypergraphRouter(latent_dim, geometry_router_edges)
        )
        self.semantic_readout = (
            DenseHyperedgeReadout(latent_dim, semantic_edges, geometry=False) if self.enable_semantic else None
        )
        self.geometry_readout = (
            DenseHyperedgeReadout(latent_dim, geometry_router_edges, geometry=True) if self.enable_geometry else None
        )
        self.cls_adapters = nn.ModuleList(nn.Conv2d(latent_dim, c, 1, bias=False) for c in ch)
        self.reg_adapters = nn.ModuleList(nn.Conv2d(latent_dim, c, 1, bias=False) for c in ch)
        self.cls_gamma = nn.ParameterList(nn.Parameter(torch.full((c,), float(gamma_init))) for c in ch)
        self.reg_gamma = nn.ParameterList(nn.Parameter(torch.full((c,), float(gamma_init))) for c in ch)
        self.collect_stats = True
        self.collect_loss_routing = False
        self.last_stats: dict[str, object] = {}
        self.loss_routing: dict[str, object] = {}

    @staticmethod
    def _anchor_broadcast(
        anchor_messages: torch.Tensor, grid_sizes: Sequence[Sequence[int]], dense: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        outputs, start = [], 0
        for message_target, grid_size in zip(dense, grid_sizes):
            gh, gw = int(grid_size[0]), int(grid_size[1])
            count = gh * gw
            msg = anchor_messages[:, start : start + count].transpose(1, 2).reshape(
                anchor_messages.shape[0], anchor_messages.shape[-1], gh, gw
            )
            outputs.append(F.interpolate(msg, message_target.shape[-2:], mode="bilinear", align_corners=False))
            start += count
        return outputs

    def forward(self, feats: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        dense, tokens, boundary, anchor_xy, levels, dense_xy, _, actual_grid_sizes = self.encoder(feats)
        cls_delta = [torch.zeros_like(x) for x in dense]
        reg_delta = [torch.zeros_like(x) for x in dense]
        cls_incidence = reg_incidence = cls_weights = reg_weights = None
        semantic_edges = semantic_anchor_messages = None

        if self.enable_semantic or self.share_hypergraph:
            cls_incidence, semantic_edges = self.semantic_router(tokens, levels, self.enable_cross_scale)
            if not self.dense_readout:
                semantic_anchor_messages = torch.bmm(cls_incidence, semantic_edges.float()).to(tokens.dtype)
        if self.enable_semantic:
            if self.dense_readout:
                cls_delta, cls_weights = self.semantic_readout(
                    dense, semantic_edges, dense_xy, enable_cross_scale=self.enable_cross_scale
                )
            else:
                cls_delta = self._anchor_broadcast(
                    semantic_anchor_messages, actual_grid_sizes, dense
                )

        edge_xy = None
        if self.enable_geometry:
            if self.share_hypergraph:
                reg_incidence, geometry_edges = cls_incidence, semantic_edges
                degree = reg_incidence.sum(1).clamp_min(1e-4)
                batched_anchor_xy = anchor_xy.float().unsqueeze(0).expand(tokens.shape[0], -1, -1)
                edge_xy = torch.matmul(reg_incidence.transpose(1, 2), batched_anchor_xy)
                edge_xy = edge_xy / degree.unsqueeze(-1)
            else:
                reg_incidence, geometry_edges, learned_edge_xy = self.geometry_router(
                    tokens, boundary, anchor_xy, levels, self.enable_cross_scale
                )
                edge_xy = learned_edge_xy.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            geometry_anchor_messages = (
                torch.bmm(reg_incidence, geometry_edges.float()).to(tokens.dtype)
                if not self.dense_readout
                else None
            )
            if self.dense_readout:
                reg_delta, reg_weights = self.geometry_readout(
                    dense,
                    geometry_edges,
                    dense_xy,
                    edge_xy,
                    enable_cross_scale=self.enable_cross_scale,
                )
            else:
                reg_delta = self._anchor_broadcast(
                    geometry_anchor_messages, actual_grid_sizes, dense
                )

        cls_feats, reg_feats = self._apply_residuals(feats, cls_delta, reg_delta)

        self.last_stats = (
            {
                "anchor_tokens": tokens.shape[1],
                "cls_incidence": None if cls_incidence is None else cls_incidence.detach(),
                "reg_incidence": None if reg_incidence is None else reg_incidence.detach(),
                "cls_readout": None if cls_weights is None else [x.detach() for x in cls_weights],
                "reg_readout": None if reg_weights is None else [x.detach() for x in reg_weights],
                "cls_delta_abs": [x.detach().float().abs().mean() for x in cls_delta],
                "reg_delta_abs": [x.detach().float().abs().mean() for x in reg_delta],
            }
            if self.collect_stats
            else {}
        )
        self.loss_routing = (
            {
                "cls_incidence": cls_incidence,
                "reg_incidence": reg_incidence,
                "anchor_grid_sizes": actual_grid_sizes,
            }
            if getattr(self, "collect_loss_routing", False)
            else {}
        )
        return cls_feats, reg_feats

    def _apply_residuals(
        self, feats: list[torch.Tensor], cls_delta: list[torch.Tensor], reg_delta: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Apply task-specific adapters and independent per-level LayerScale parameters."""
        cls_feats, reg_feats = [], []
        for i, feat in enumerate(feats):
            cls_residual = self.cls_adapters[i](cls_delta[i]) if self.enable_semantic else torch.zeros_like(feat)
            reg_residual = self.reg_adapters[i](reg_delta[i]) if self.enable_geometry else torch.zeros_like(feat)
            cls_feats.append(feat + self.cls_gamma[i].view(1, -1, 1, 1) * cls_residual)
            reg_feats.append(feat + self.reg_gamma[i].view(1, -1, 1, 1) * reg_residual)
        return cls_feats, reg_feats

    def theoretical_mac_breakdown(self, shapes: Sequence[Sequence[int]], batch_size: int = 1) -> dict[str, int]:
        """Return routing MACs that THOP misses or merges, grouped by operation family."""
        dim = self.encoder.latent_dim
        tokens = sum(h * w for h, w in self.encoder.anchor_grid_sizes)
        dense_nodes = sum(h * w for _, h, w in shapes)
        semantic_edges = self.semantic_readout.num_edges if self.enable_semantic else 0
        geometry_edges = self.geometry_readout.num_edges if self.enable_geometry else 0
        breakdown = {
            "shared_stem": sum(h * w * c * dim for c, h, w in shapes),
            "coordinate_mlp": dense_nodes * (2 * dim + dim * dim),
            "semantic_router_linear": int(self.enable_semantic)
            * (2 * tokens * dim * dim + dim * dim + 3 * semantic_edges * dim * dim),
            "geometry_router_linear": int(self.enable_geometry)
            * (4 * tokens * dim * dim + 3 * geometry_edges * dim * dim + 2 * tokens * dim),
            "anchor_incidence_bmm": 2 * tokens * (semantic_edges + geometry_edges) * dim,
            "dense_query_projection": dense_nodes
            * dim
            * dim
            * (int(self.enable_semantic) + int(self.enable_geometry)),
            "dense_edge_key_value": 2 * (semantic_edges + geometry_edges) * dim * dim,
            "dense_readout_bmm": 2 * dense_nodes * (semantic_edges + geometry_edges) * dim,
            "residual_adapters": 2 * sum(h * w * c * dim for c, h, w in shapes),
        }
        return {key: int(batch_size * value) for key, value in breakdown.items()}

    def theoretical_macs(self, shapes: Sequence[Sequence[int]], batch_size: int = 1) -> int:
        """Return total routing MACs, including matmul operations omitted by THOP."""
        return sum(self.theoretical_mac_breakdown(shapes, batch_size).values())


class DASHDetect(Detect):
    """YOLO Detect head with independent semantic/geometry hypergraphs and dense readout."""

    def __init__(
        self,
        nc: int = 80,
        latent_dim: int = 48,
        semantic_edges: int = 12,
        geometry_edges: int = 8,
        anchor_grid_sizes: Sequence[Sequence[int]] = ((12, 12), (8, 8), (4, 4)),
        enable_semantic: bool = True,
        enable_geometry: bool = True,
        enable_cross_scale: bool = True,
        dense_readout: bool = True,
        gamma_init: float = 0.05,
        share_hypergraph: bool = False,
        return_routing_stats: bool = True,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        if end2end:
            raise ValueError("DASH-M1 supports the standard one-to-many Detect protocol only")
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=ch)
        self.routing = DASHRoutingBlock(
            ch=ch,
            latent_dim=latent_dim,
            semantic_edges=semantic_edges,
            geometry_edges=geometry_edges,
            anchor_grid_sizes=anchor_grid_sizes,
            enable_semantic=enable_semantic,
            enable_geometry=enable_geometry,
            enable_cross_scale=enable_cross_scale,
            dense_readout=dense_readout,
            gamma_init=gamma_init,
            share_hypergraph=share_hypergraph,
        )
        self.return_routing_stats = bool(return_routing_stats)
        self.routing.collect_stats = self.return_routing_stats
        self.routing_stats: dict[str, object] = {}

    def forward(self, x: list[torch.Tensor]):
        cls_feats, reg_feats = self.routing(x)
        bs = x[0].shape[0]
        preds = {
            "boxes": torch.cat(
                [self.cv2[i](reg_feats[i]).reshape(bs, 4 * self.reg_max, -1) for i in range(self.nl)], -1
            ),
            "scores": torch.cat([self.cv3[i](cls_feats[i]).reshape(bs, self.nc, -1) for i in range(self.nl)], -1),
            "feats": x,
        }
        self.routing_stats = self.routing.last_stats if self.return_routing_stats else {}
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)


class DASHRoutingBlockEfficient(DASHRoutingBlock):
    """Compute-efficient DASH routing with shared stems and low-rank task-specific residual adapters."""

    def __init__(
        self,
        ch: Sequence[int],
        latent_dim: int = 32,
        semantic_edges: int = 8,
        geometry_edges: int = 6,
        anchor_grid_sizes: Sequence[Sequence[int]] = ((10, 10), (6, 6), (4, 4)),
        adapter_rank: int = 16,
        enable_semantic: bool = True,
        enable_geometry: bool = True,
        enable_cross_scale: bool = True,
        dense_readout: bool = True,
        gamma_init: float = 0.05,
        share_hypergraph: bool = False,
    ):
        super().__init__(
            ch=ch,
            latent_dim=latent_dim,
            semantic_edges=semantic_edges,
            geometry_edges=geometry_edges,
            anchor_grid_sizes=anchor_grid_sizes,
            enable_semantic=enable_semantic,
            enable_geometry=enable_geometry,
            enable_cross_scale=enable_cross_scale,
            dense_readout=dense_readout,
            gamma_init=gamma_init,
            share_hypergraph=share_hypergraph,
        )
        self.adapter_rank = int(adapter_rank)
        if self.adapter_rank <= 0:
            raise ValueError("adapter_rank must be positive")
        self.adapter_reductions = nn.ModuleList(
            nn.Sequential(nn.Conv2d(latent_dim, self.adapter_rank, 1, bias=False), nn.SiLU()) for _ in ch
        )
        self.cls_adapters = nn.ModuleList(nn.Conv2d(self.adapter_rank, c, 1, bias=False) for c in ch)
        self.reg_adapters = nn.ModuleList(nn.Conv2d(self.adapter_rank, c, 1, bias=False) for c in ch)

    def _apply_residuals(
        self, feats: list[torch.Tensor], cls_delta: list[torch.Tensor], reg_delta: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Share only low-rank reduction stems; keep task outputs and LayerScale fully independent."""
        cls_feats, reg_feats = [], []
        for i, feat in enumerate(feats):
            if self.enable_semantic and self.enable_geometry:
                reduced = self.adapter_reductions[i](torch.cat((cls_delta[i], reg_delta[i]), 0))
                cls_reduced, reg_reduced = reduced.chunk(2, 0)
            else:
                cls_reduced = self.adapter_reductions[i](cls_delta[i]) if self.enable_semantic else None
                reg_reduced = self.adapter_reductions[i](reg_delta[i]) if self.enable_geometry else None
            cls_residual = self.cls_adapters[i](cls_reduced) if cls_reduced is not None else torch.zeros_like(feat)
            reg_residual = self.reg_adapters[i](reg_reduced) if reg_reduced is not None else torch.zeros_like(feat)
            cls_feats.append(feat + self.cls_gamma[i].view(1, -1, 1, 1) * cls_residual)
            reg_feats.append(feat + self.reg_gamma[i].view(1, -1, 1, 1) * reg_residual)
        return cls_feats, reg_feats

    def theoretical_mac_breakdown(self, shapes: Sequence[Sequence[int]], batch_size: int = 1) -> dict[str, int]:
        """Return Efficient DASH MACs with low-rank adapters separated from graph/readout operations."""
        breakdown = super().theoretical_mac_breakdown(shapes, batch_size=1)
        dim = self.encoder.latent_dim
        breakdown["residual_adapters"] = sum(
            h * w * (dim * self.adapter_rank + 2 * self.adapter_rank * c) for c, h, w in shapes
        )
        return {key: int(batch_size * value) for key, value in breakdown.items()}


class DASHDetectEfficient(DASHDetect):
    """DASH-M1E Detect head retaining independent dual hypergraphs and dense readout."""

    def __init__(
        self,
        nc: int = 80,
        latent_dim: int = 32,
        semantic_edges: int = 8,
        geometry_edges: int = 6,
        anchor_grid_sizes: Sequence[Sequence[int]] = ((10, 10), (6, 6), (4, 4)),
        adapter_rank: int = 16,
        enable_semantic: bool = True,
        enable_geometry: bool = True,
        enable_cross_scale: bool = True,
        dense_readout: bool = True,
        gamma_init: float = 0.05,
        share_hypergraph: bool = False,
        return_routing_stats: bool = True,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        super().__init__(
            nc=nc,
            latent_dim=latent_dim,
            semantic_edges=semantic_edges,
            geometry_edges=geometry_edges,
            anchor_grid_sizes=anchor_grid_sizes,
            enable_semantic=enable_semantic,
            enable_geometry=enable_geometry,
            enable_cross_scale=enable_cross_scale,
            dense_readout=dense_readout,
            gamma_init=gamma_init,
            share_hypergraph=share_hypergraph,
            return_routing_stats=return_routing_stats,
            reg_max=reg_max,
            end2end=end2end,
            ch=ch,
        )
        self.routing = DASHRoutingBlockEfficient(
            ch=ch,
            latent_dim=latent_dim,
            semantic_edges=semantic_edges,
            geometry_edges=geometry_edges,
            anchor_grid_sizes=anchor_grid_sizes,
            adapter_rank=adapter_rank,
            enable_semantic=enable_semantic,
            enable_geometry=enable_geometry,
            enable_cross_scale=enable_cross_scale,
            dense_readout=dense_readout,
            gamma_init=gamma_init,
            share_hypergraph=share_hypergraph,
        )
        self.routing.collect_stats = self.return_routing_stats


class DASHDetectM2Lite(DASHDetectEfficient):
    """M1H inference graph with train-only task-aligned incidence tensors for M2-Lite loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.routing.collect_loss_routing = True
        self.relation_loss_weights = (0.05, 0.05)

    def forward(self, x: list[torch.Tensor]):
        output = super().forward(x)
        if self.training:
            output["dash_routing"] = self.routing.loss_routing
        return output
