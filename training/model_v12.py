#!/usr/bin/env python3
"""
Hanfire Shizi Clean V12 Architecture (方案 3C):
  - Modern ConvNeXt backbones (`convnext_base` 88.6M Teacher / `convnext_tiny` 28.6M Student, pure torchvision!)
  - Sub-center ArcFace (K=3 sub-centers per class) + PartialFC negative sampling (5x faster, 4x less VRAM!)
  - IDS Component/Radical multi-label structural supervision head (1,217 CJKVI components + 12 layout operators)
  - Pure opset-17 per-channel Weight-Only U8 ONNX export (Conv + MatMul/Gemm) -> ~28.5 MB for ConvNeXt-Tiny!
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SubCenterPartialFCHead(nn.Module):
    """
    Sub-center ArcFace (K=3) with PartialFC negative sampling.
    Handles multi-variant ancient/cursive glyphs (e.g., oracle bones / Chu slips / Kuangcao)
    without collapsing distinct allographs into a single centroid, while keeping T4 VRAM low!
    """
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_subcenters: int = 3,
        s: float = 64.0,
        m: float = 0.45,
        neg_ratio: float = 0.20,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.K = num_subcenters
        self.s = s
        self.m = m
        self.neg_ratio = neg_ratio
        # Shape: (num_classes * K, in_dim)
        self.weight = nn.Parameter(torch.empty(num_classes * num_subcenters, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor, margin: float | None = None) -> torch.Tensor:
        """Return cross-entropy loss directly (uses PartialFC class subset when neg_ratio < 1.0)."""
        emb = F.normalize(emb.float(), dim=1)
        m_cur = self.m if margin is None else float(margin)
        device = emb.device

        if self.training and 0.0 < self.neg_ratio < 1.0:
            # Gather all positive classes present in this local batch + random negative subset
            pos_cls = torch.unique(labels)
            n_neg = max(1024, int(self.num_classes * self.neg_ratio))
            neg_cls = torch.randint(0, self.num_classes, (n_neg,), device=device)
            subset = torch.unique(torch.cat([pos_cls, neg_cls], dim=0))  # sorted ascending!
            # Map labels -> local index in `subset`
            local_labels = torch.searchsorted(subset, labels)
            # Expand subset class indices -> K sub-center row indices
            k_offs = torch.arange(self.K, device=device)
            row_idx = (subset[:, None] * self.K + k_offs[None, :]).reshape(-1)
            w_sub = F.normalize(self.weight[row_idx].float(), dim=1)  # (M*K, D)
            M = subset.shape[0]
        else:
            local_labels = labels
            w_sub = F.normalize(self.weight.float(), dim=1)
            M = self.num_classes

        # Cosine against all M*K sub-centers on BF16/FP16 Tensor Cores (2x faster GEMM on A100/L4),
        # keeping F.normalize & ArcFace angular margin math in exact FP32!
        if emb.is_cuda and torch.is_autocast_enabled():
            gd = torch.get_autocast_dtype("cuda")
            cos_all = torch.matmul(emb.to(gd), w_sub.to(gd).t()).float().view(emb.shape[0], M, self.K)
        else:
            cos_all = torch.matmul(emb, w_sub.t()).view(emb.shape[0], M, self.K)
        cos = cos_all.max(dim=2).values.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        if m_cur > 0.0:
            # Stable cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
            sin_t = torch.sqrt((1.0 - cos * cos).clamp(min=1e-7))
            cos_m = math.cos(m_cur)
            sin_m = math.sin(m_cur)
            phi = cos * cos_m - sin_t * sin_m
            # Easy-margin fallback for stability when theta + m > pi
            th = math.cos(math.pi - m_cur)
            mm = math.sin(math.pi - m_cur) * m_cur
            phi = torch.where(cos > th, phi, cos - mm)
            # Only apply margin to target class
            idx = torch.arange(emb.shape[0], device=device)
            logits = cos.clone()
            logits[idx, local_labels] = phi[idx, local_labels]
        else:
            logits = cos

        return F.cross_entropy(logits * self.s, local_labels)


def build_backbone(arch: str, emb_dim: int = 512, pretrained: bool = True) -> nn.Module:
    arch = arch.lower()
    if arch == "convnext_base":
        w = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.convnext_base(weights=w)
        in_f = net.classifier[2].in_features
        net.classifier[2] = nn.Linear(in_f, emb_dim)
        return net
    if arch == "convnext_small":
        w = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.convnext_small(weights=w)
        in_f = net.classifier[2].in_features
        net.classifier[2] = nn.Linear(in_f, emb_dim)
        return net
    if arch == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.convnext_tiny(weights=w)
        in_f = net.classifier[2].in_features
        net.classifier[2] = nn.Linear(in_f, emb_dim)
        return net
    if arch == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=w)
        net.fc = nn.Linear(net.fc.in_features, emb_dim)
        return net
    raise ValueError(f"Unsupported arch: {arch}")


class GlyphModelV12(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        num_styles: int,
        num_dynasties: int,
        num_ids: int = 0,
        emb_dim: int = 512,
        subcenters: int = 3,
        neg_ratio: float = 0.20,
        pretrained: bool = True,
    ):
        super().__init__()
        self.arch = arch
        self.emb_dim = emb_dim
        self.backbone = build_backbone(arch, emb_dim=emb_dim, pretrained=pretrained)
        self.head = SubCenterPartialFCHead(
            emb_dim, num_classes, num_subcenters=subcenters, s=64.0, m=0.45, neg_ratio=neg_ratio
        )
        self.style_head = nn.Linear(emb_dim, num_styles)
        self.dyn_head = nn.Linear(emb_dim, num_dynasties)
        self.ids_head = nn.Linear(emb_dim, num_ids) if num_ids > 0 else None

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x).float(), p=2, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        margin: float | None = None,
    ):
        emb = self.embed(x)
        if labels is None:
            return emb
        l_arc = self.head(emb, labels, margin=margin)
        sty_logits = self.style_head(emb)
        dyn_logits = self.dyn_head(emb)
        ids_logits = self.ids_head(emb) if self.ids_head is not None else None
        return emb, l_arc, sty_logits, dyn_logits, ids_logits


def distillation_loss(e_student: torch.Tensor, e_teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combined Feature Cosine + Relational Structure Distillation (RKD):
      1. Direct unit-sphere alignment: mean(1 - cos(e_s, e_t))
      2. Pairwise relational similarity preservation: SmoothL1(E_s @ E_s.T, E_t @ E_t.T)
    """
    es = F.normalize(e_student.float(), dim=1)
    et = F.normalize(e_teacher.float(), dim=1)
    loss_cos = (1.0 - (es * et).sum(dim=1)).mean()
    sim_s = es @ es.t()
    sim_t = et @ et.t()
    loss_rkd = F.smooth_l1_loss(sim_s, sim_t, beta=0.1)
    return loss_cos, loss_rkd


class _EmbedExportWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), p=2, dim=1)


def export_v12_wo_u8_opset17(model: GlyphModelV12, out_onnx_u8: Path) -> float:
    """
    Export ConvNeXt / ResNet V12 backbone to pure opset-17 per-channel Weight-Only U8 ONNX.
    Quantizes both Conv AND MatMul/Gemm weight matrices (>=16,384 params) so ConvNeXt-Tiny fits ~28.5 MB!
    Safely unwraps torch.compile (_orig_mod) and restores contiguous FP32 on CPU.
    """
    tmp_fp32 = out_onnx_u8.with_suffix(".fp32.onnx")
    bb = getattr(model.backbone, "_orig_mod", model.backbone)
    bb = bb.cpu().float().to(memory_format=torch.contiguous_format).eval()
    wrapper = _EmbedExportWrapper(bb).cpu().eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        wrapper,
        dummy,
        str(tmp_fp32),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    m = onnx.load(str(tmp_fp32), load_external_data=True)
    del m.graph.value_info[:]
    init_map = {t.name: t for t in m.graph.initializer}

    new_inits, deq_nodes, replaced = [], [], set()
    for node in m.graph.node:
        if node.op_type in ("Conv", "Gemm", "MatMul") and len(node.input) >= 2 and node.input[1] in init_map:
            wname = node.input[1]
            if wname in replaced:
                node.input[1] = wname + "_deq"
                continue
            arr = numpy_helper.to_array(init_map[wname])
            if arr.dtype != np.float32 or arr.size < 16384:
                continue
            # Per-channel quantization along axis=0 for Conv/Gemm, axis=1 for MatMul (K, N)
            axis = 1 if node.op_type == "MatMul" and arr.ndim == 2 else 0
            reduce_axes = tuple(i for i in range(arr.ndim) if i != axis)
            wmin = arr.min(axis=reduce_axes)
            wmax = arr.max(axis=reduce_axes)
            scales = np.maximum((wmax - wmin) / 255.0, 1e-8).astype(np.float32)
            zps = np.clip(np.round(-wmin / scales), 0, 255).astype(np.uint8)
            b = [1] * arr.ndim
            b[axis] = arr.shape[axis]
            q = np.clip(np.round(arr / scales.reshape(b)) + zps.reshape(b), 0, 255).astype(np.uint8)
            q_name, s_name, z_name, dq_out = wname + "_q8", wname + "_s8", wname + "_z8", wname + "_deq"
            new_inits.extend([
                numpy_helper.from_array(q, name=q_name),
                numpy_helper.from_array(scales, name=s_name),
                numpy_helper.from_array(zps, name=z_name),
            ])
            deq_nodes.append(helper.make_node("DequantizeLinear", [q_name, s_name, z_name], [dq_out], axis=axis))
            node.input[1] = dq_out
            replaced.add(wname)

    kept = [t for t in m.graph.initializer if t.name not in replaced]
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept + new_inits)
    nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(deq_nodes + nodes)
    onnx.save_model(m, str(out_onnx_u8), save_as_external_data=False)
    if tmp_fp32.exists():
        tmp_fp32.unlink()

    xs = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        e_pt = wrapper(xs).numpy()
    sess = ort.InferenceSession(str(out_onnx_u8), providers=["CPUExecutionProvider"])
    e_ox = sess.run(["embedding"], {"image": xs.numpy()})[0]
    e_ox /= np.maximum(np.linalg.norm(e_ox, axis=1, keepdims=True), 1e-8)
    min_cos = float((e_pt * e_ox).sum(axis=1).min())
    print(f"[ONNX V12] {out_onnx_u8.name} ({out_onnx_u8.stat().st_size/1048576:.2f} MB) | min_cos={min_cos:.5f}", flush=True)
    return min_cos
