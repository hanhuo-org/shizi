#!/usr/bin/env python3
"""
汉火识字 Clean V12 (方案 3C) 全平台高性能训练与离线蒸馏引擎（原生兼容 Colab L4/A100/H100 与 Kaggle 2×T4 DDP）：
  1. `--mode teacher`       : ConvNeXt-Base (88.6M) + Sub-center ArcFace (K=3 PartialFC) + 1,217 汉字 IDS 偏旁监督
  2. `--mode cache_teacher` : 离线抽取 123.5 万清洗图 512 维 fp16 教师特征（~1.26 GB），L4 约 7 分钟完成
  3. `--mode student`       : ConvNeXt-Tiny (28.6M -> 27.5 MB WO-U8 ONNX) 离线余弦 + RKD 关系蒸馏

极限显卡利用率优化栈（面向 Ada L4 / Ampere A100 自动拉满 95%+ 利用率，并无缝向下兼容 Turing T4）：
  - Auto-BF16 + TF32 (SM>=8.0 自动 BF16 零 GradScaler 开销；T4 SM7.5 自动回退 FP16 + GradScaler)
  - NHWC `channels_last` 内存布局（ConvNeXt 7×7 DWConv + 1×1 PWConv Tensor Core 提速 15~25%）
  - `torch.compile(backbone)` 选择性编译静态主干（融合 LayerNorm2d+GELU+LayerScale，避开动态 PartialFC 零重编译！）
  - `GpuGlyphAugment` 全 GPU 批量数据增强 + 1-ch `uint8` 传输（PCIe 减 12×，CPU 吞吐提 5×，杜绝 GPU 饥饿）
  - `GpuIdsTable` 全 GPU 向量化偏旁查表 + GPU 端 Loss 累计（零 `.tolist()` / `.item()` 每步 CUDA 同步气泡）
  - Fused CUDA `AdamW(fused=True)` + `CUDAPrefetcher` 双流异步预取 + 零解压 PAX `.tar` 字节偏移流式索引
  - 每 N 步与每 Epoch 后台异步 `rclone` 备份至 Google Drive (`wylggww:shizi_v12/...`)，支持跨平台无缝断点续训
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import math
import os
import random
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

SHIZI_ROOT = Path(__file__).resolve().parent.parent
if str(SHIZI_ROOT) not in sys.path:
    sys.path.insert(0, str(SHIZI_ROOT))

from training.model import RandomMorphology, load_glyph, make_transforms, simp
from training.model_v12 import GlyphModelV12, distillation_loss, export_v12_wo_u8_opset17
from training.sync_gdrive import CheckpointSyncer, ensure_rclone_conf


# ---------- 1. Ultra-Fast C++ OpenCV Decode + GPU Batched Augmentation (Feeds A100 at 3,000+ img/s!) ----------

def _decode_glyph_bytes_cv2(raw: bytes) -> np.ndarray:
    """C++ libpng/libjpeg-turbo direct decode -> (H, W) uint8 black-ink-on-white numpy array (~3x faster than PIL!)."""
    arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return np.full((224, 224), 255, dtype=np.uint8)
    if arr.ndim == 3:
        c = arr.shape[2]
        if c == 4:
            # Composite BGRA onto white background; fallback black background if glyph is white-on-transparent
            alpha = arr[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
            rgb = arr[:, :, :3].astype(np.float32)
            comp = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
            gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
            if int(gray.max()) - int(gray.min()) < 4:
                comp_b = (rgb * alpha).astype(np.uint8)
                gray = cv2.cvtColor(comp_b, cv2.COLOR_BGR2GRAY)
        elif c == 2:
            alpha = arr[:, :, 1].astype(np.float32) * (1.0 / 255.0)
            gray = (arr[:, :, 0].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        else:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr
    # Fast strided polarity check (zero resize allocation)
    if gray[::max(1, gray.shape[0] // 32), ::max(1, gray.shape[1] // 32)].mean() < 110.0:
        gray = 255 - gray
    return gray


def _fast_rrc_224_u8(gray: np.ndarray) -> torch.Tensor:
    """Fast C++ OpenCV RandomResizedCrop(224, scale=(0.8, 1.0)) -> (1, 224, 224) uint8 tensor."""
    h, w = gray.shape[:2]
    if h >= 32 and w >= 32:
        area = h * w
        target_area = random.uniform(0.80, 1.0) * area
        ratio = random.uniform(0.75, 1.3333)
        cw = min(w, max(16, int(round(math.sqrt(target_area * ratio)))))
        ch = min(h, max(16, int(round(math.sqrt(target_area / ratio)))))
        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        crop = gray[y0 : y0 + ch, x0 : x0 + cw]
    else:
        crop = gray
    out = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(out).unsqueeze(0)


def _fast_val_224_u8(gray: np.ndarray) -> torch.Tensor:
    """Fast C++ OpenCV Resize(256 short side) + CenterCrop(224) -> (1, 224, 224) uint8 tensor."""
    h, w = gray.shape[:2]
    if h <= w:
        nh, nw = 256, max(256, int(round(w * 256.0 / max(1, h))))
    else:
        nw, nh = 256, max(256, int(round(h * 256.0 / max(1, w))))
    res = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0 = (nh - 224) // 2
    x0 = (nw - 224) // 2
    crop = np.ascontiguousarray(res[y0 : y0 + 224, x0 : x0 + 224])
    return torch.from_numpy(crop).unsqueeze(0)


def make_fast_u8_transforms():
    """Ultra-fast C++ OpenCV 1-ch uint8 transforms; ALL heavy augs (incl. morphology!) run on GPU!"""
    return _fast_rrc_224_u8, _fast_val_224_u8


class GpuGlyphAugment(nn.Module):
    """
    Batched GPU-native domain augmentation matching `make_transforms()` semantics at <0.4ms / batch:
      - Random morphology dilation/erosion (p=0.25, GPU max_pool2d — 0.03ms!)
      - Random affine (rotation ±12°, scale 0.90~1.10, translate ±8%, shear ±6° for perspective-like warp)
      - Random Gaussian & salt-pepper noise (p=0.30)
      - Random brightness & contrast jitter (±0.35)
      - Random separable 5×5 Gaussian blur (p=0.45)
      - Random white-fill erasing (p=0.25)
      - Normalize to [-1, 1] and expand to 3-ch `channels_last`!
    """
    def __init__(self, device: torch.device, channels_last: bool = True):
        super().__init__()
        self.channels_last = channels_last
        ks = []
        coords = torch.arange(-2, 3, dtype=torch.float32)
        for sig in (0.5, 1.0, 1.4):
            g = torch.exp(-(coords ** 2) / (2.0 * sig * sig))
            g = g / g.sum()
            ks.append(g)
        self.register_buffer("blur_k", torch.stack(ks, dim=0).to(device))  # (3, 5)

    @torch.no_grad()
    def forward(self, x_u8: torch.Tensor, training: bool = True) -> torch.Tensor:
        # x_u8: (B, 1, 224, 224) uint8 on GPU -> float32 in [0, 1]
        x = x_u8.to(dtype=torch.float32).mul_(1.0 / 255.0)
        if training:
            B, _, H, W = x.shape
            dev = x.device

            # 0) Batched GPU Morphology (thicken/thin strokes via 3x3 min/max pool, p=0.25)
            m_morph = torch.rand(B, 1, 1, 1, device=dev) < 0.25
            if m_morph.any():
                thicken = (torch.rand(B, 1, 1, 1, device=dev) < 0.50) & m_morph
                thin = (~thicken) & m_morph
                x_thk = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
                x_thn = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
                x = torch.where(thicken, x_thk, torch.where(thin, x_thn, x))

            # 1) Batched affine warp (rotation ±12°, scale 0.9~1.1, translate ±8%, shear ±6°)
            ang = (torch.rand(B, device=dev) * 24.0 - 12.0) * (math.pi / 180.0)
            shx = (torch.rand(B, device=dev) * 12.0 - 6.0) * (math.pi / 180.0)
            sc = 1.0 / (0.90 + torch.rand(B, device=dev) * 0.20)
            tx = (torch.rand(B, device=dev) * 0.16 - 0.08)
            ty = (torch.rand(B, device=dev) * 0.16 - 0.08)
            ca, sa = torch.cos(ang), torch.sin(ang)
            tsh = torch.tan(shx)
            theta = torch.zeros((B, 2, 3), device=dev, dtype=torch.float32)
            theta[:, 0, 0] = sc * (ca + sa * tsh)
            theta[:, 0, 1] = sc * (-sa + ca * tsh)
            theta[:, 0, 2] = tx
            theta[:, 1, 0] = sc * sa
            theta[:, 1, 1] = sc * ca
            theta[:, 1, 2] = ty
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            # Invert-pad hack: grid_sample pads with 0, so invert to pad white (1.0) cleanly!
            x = 1.0 - F.grid_sample(1.0 - x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

            # 2) Random Gaussian + Salt/Pepper noise (p=0.30)
            m_noise = (torch.rand(B, 1, 1, 1, device=dev) < 0.30)
            if m_noise.any():
                is_gauss = (torch.rand(B, 1, 1, 1, device=dev) < 0.50) & m_noise
                sigma = (0.02 + torch.rand(B, 1, 1, 1, device=dev) * 0.04) * is_gauss
                x = x + torch.randn_like(x) * sigma
                is_sp = (~is_gauss) & m_noise
                if is_sp.any():
                    sp_mask = (torch.rand_like(x) < 0.006) & is_sp
                    sp_val = (torch.rand_like(x) > 0.5).float()
                    x = torch.where(sp_mask, sp_val, x)

            # 3) Random brightness & contrast jitter (±0.35)
            bri = 0.65 + torch.rand(B, 1, 1, 1, device=dev) * 0.70
            con = 0.65 + torch.rand(B, 1, 1, 1, device=dev) * 0.70
            mean = x.mean(dim=(2, 3), keepdim=True)
            x = ((x - mean) * con + mean) * bri
            x.clamp_(0.0, 1.0)

            # 4) Random separable 5x5 Gaussian blur (p=0.45)
            m_blur = (torch.rand(B, 1, 1, 1, device=dev) < 0.45)
            if m_blur.any():
                k = self.blur_k[random.randint(0, 2)]
                xb = F.conv2d(x, k.view(1, 1, 1, 5), padding=(0, 2))
                xb = F.conv2d(xb, k.view(1, 1, 5, 1), padding=(2, 0))
                x = torch.where(m_blur, xb, x)

            # 5) Random white-fill erasing (p=0.25, area 2%~12%)
            m_er = torch.rand(B, device=dev) < 0.25
            if m_er.any():
                er_idx = torch.nonzero(m_er, as_tuple=False).squeeze(1)
                eh = torch.randint(24, 68, (len(er_idx),), device=dev)
                ew = torch.randint(24, 68, (len(er_idx),), device=dev)
                ey = (torch.rand(len(er_idx), device=dev) * (H - eh).float()).long()
                ex = (torch.rand(len(er_idx), device=dev) * (W - ew).float()).long()
                yy = torch.arange(H, device=dev).view(1, 1, H, 1)
                xx = torch.arange(W, device=dev).view(1, 1, 1, W)
                mask = (
                    (yy >= ey.view(-1, 1, 1, 1)) & (yy < (ey + eh).view(-1, 1, 1, 1)) &
                    (xx >= ex.view(-1, 1, 1, 1)) & (xx < (ex + ew).view(-1, 1, 1, 1))
                )
                x_sel = x[er_idx].masked_fill(mask, 1.0)
                x[er_idx] = x_sel

        # Normalize [0, 1] -> [-1, 1] and expand 1-ch -> 3-ch channels_last
        x.mul_(2.0).sub_(1.0)
        x3 = x.expand(-1, 3, -1, -1)
        if self.channels_last:
            return x3.contiguous(memory_format=torch.channels_last)
        return x3.contiguous()


class CUDAPrefetcher:
    """Overlaps H2D transfer on a dedicated CUDA stream with current step's compute."""
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.use_cuda = (device.type == "cuda")
        self.stream = torch.cuda.Stream(device=device) if self.use_cuda else None

    def __len__(self):
        return len(self.loader)

    def _to_gpu(self, batch):
        return tuple(
            t.to(self.device, non_blocking=True) if torch.is_tensor(t) else t
            for t in batch
        )

    def __iter__(self):
        if not self.use_cuda:
            for batch in self.loader:
                yield batch
            return
        it = iter(self.loader)
        try:
            nxt = next(it)
        except StopIteration:
            return
        with torch.cuda.stream(self.stream):
            nxt = self._to_gpu(nxt)
        while True:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            cur = nxt
            try:
                raw_nxt = next(it)
                with torch.cuda.stream(self.stream):
                    nxt = self._to_gpu(raw_nxt)
            except StopIteration:
                nxt = None
            yield cur
            if nxt is None:
                break


class GpuIdsTable:
    """Zero-sync GPU lookup table mapping class_id -> multi-hot IDS radical target vector."""
    def __init__(self, cls2ids: list[list[int]], num_ids: int, device: torch.device):
        self.num_ids = num_ids
        self.device = device
        n_cls = len(cls2ids)
        max_k = max((len(c) for c in cls2ids), default=1)
        max_k = max(1, max_k)
        idx_tbl = np.full((n_cls, max_k), num_ids, dtype=np.int64)
        valid_tbl = np.zeros(n_cls, dtype=bool)
        for ci, comps in enumerate(cls2ids):
            if comps:
                k = min(len(comps), max_k)
                idx_tbl[ci, :k] = comps[:k]
                valid_tbl[ci] = True
        self.idx_tbl = torch.from_numpy(idx_tbl).to(device)      # (C, max_k)
        self.valid_tbl = torch.from_numpy(valid_tbl).to(device)  # (C,)

    @torch.no_grad()
    def lookup(self, cids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = cids.shape[0]
        valid = self.valid_tbl[cids]
        tgt_ext = torch.zeros((B, self.num_ids + 1), dtype=torch.float32, device=self.device)
        tgt_ext.scatter_(1, self.idx_tbl[cids], 1.0)
        return tgt_ext[:, :self.num_ids], valid


# ---------- 2. Zero-Unpack Tar Indexer & Dual-Backend Dataset ----------

def build_or_load_tar_index(tar_dir: Path, cache_npz: Path, is_rank0: bool = True):
    """Index all clean_base_00..15.tar byte offsets once (~8s) -> 0-unpack random-access Dataset!"""
    csd = json.loads((tar_dir / "classes_styles_dynasties.json").read_text(encoding="utf-8"))
    classes, styles, dynasties = csd["classes"], csd["styles"], csd["dynasties"]
    cls2id = {c: i for i, c in enumerate(classes)}
    sty2id = {s: i for i, s in enumerate(styles)}
    dyn2id = {d: i for i, d in enumerate(dynasties)}

    if cache_npz.exists():
        z = np.load(cache_npz)
        return classes, styles, dynasties, z["shard_id"], z["offset"], z["size"], z["cls_id"], z["sty_id"], z["dyn_id"]

    if not is_rank0:
        for _ in range(300):
            if cache_npz.exists():
                time.sleep(0.5)
                z = np.load(cache_npz)
                return classes, styles, dynasties, z["shard_id"], z["offset"], z["size"], z["cls_id"], z["sty_id"], z["dyn_id"]
            time.sleep(1.0)
        raise RuntimeError(f"Timeout waiting for {cache_npz}")

    t0 = time.time()
    print(f"[TarIndex] Scanning clean_base_*.tar in {tar_dir} (one-time ~8s)...", flush=True)
    meta_map: dict[str, tuple[int, int, int]] = {}
    with gzip.open(tar_dir / "clean_base_meta.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rel, cls_name, st, dyn = json.loads(line)
            if cls_name in cls2id:
                meta_map[rel] = (cls2id[cls_name], sty2id.get(st, 0), dyn2id.get(dyn, 0))

    shards = sorted(tar_dir.glob("clean_base_*.tar"))
    assert shards, f"No clean_base_*.tar found in {tar_dir}"
    sh_ids, offs, szs, cids, sids, dids = [], [], [], [], [], []
    for si, sp in enumerate(shards):
        with tarfile.open(sp, "r:") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                nm = m.name.lstrip("./")
                rec = meta_map.get(nm) or meta_map.get(f"all_data/{nm}")
                if rec is None:
                    continue
                sh_ids.append(si)
                offs.append(m.offset_data)
                szs.append(m.size)
                cids.append(rec[0])
                sids.append(rec[1])
                dids.append(rec[2])
    assert len(sh_ids) > 0, f"Matched 0 glyphs across {len(shards)} tars in {tar_dir}!"

    arrs = (
        np.asarray(sh_ids, dtype=np.uint8),
        np.asarray(offs, dtype=np.int64),
        np.asarray(szs, dtype=np.int32),
        np.asarray(cids, dtype=np.int32),
        np.asarray(sids, dtype=np.int16),
        np.asarray(dids, dtype=np.int16),
    )
    tmp_npz = cache_npz.with_suffix(".tmp.npz")
    np.savez(
        tmp_npz,
        shard_id=arrs[0], offset=arrs[1], size=arrs[2],
        cls_id=arrs[3], sty_id=arrs[4], dyn_id=arrs[5],
    )
    tmp_npz.replace(cache_npz)
    print(f"[TarIndex] Indexed {len(arrs[0]):,} glyphs across {len(shards)} tars in {time.time()-t0:.1f}s -> {cache_npz}", flush=True)
    return classes, styles, dynasties, *arrs


def make_deterministic_split(cls_ids: np.ndarray, smoke_limit: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """85/15 deterministic stratified split per class (matches clean_base protocol)."""
    by_c = defaultdict(list)
    for idx, cid in enumerate(cls_ids.tolist()):
        by_c[cid].append(idx)
    rng = random.Random(42)
    keys = sorted(by_c.keys())
    if smoke_limit > 0:
        keys = random.Random(0).sample(keys, min(smoke_limit, len(keys)))
    tr, va = [], []
    for k in keys:
        lst = by_c[k]
        if len(lst) < 2:
            tr.extend(lst)
            continue
        rng.shuffle(lst)
        nv = max(1, int(len(lst) * 0.15))
        va.extend(lst[:nv])
        tr.extend(lst[nv:])
    return np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64)


def build_class_to_ids_map(classes: list[str], ids_json: Path | None) -> tuple[int, list[list[int]]]:
    if not ids_json or not ids_json.exists():
        return 0, [[] for _ in classes]
    d = json.loads(ids_json.read_text(encoding="utf-8"))
    num_ids = len(d["vocab"])
    c2i = d["char2ids"]
    out = []
    for cls_name in classes:
        ch = cls_name.split("__", 1)[0]
        comps = c2i.get(ch) or c2i.get(simp(ch)) or []
        out.append(comps)
    return num_ids, out


def _decode_glyph_bytes(raw: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(raw))
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        im = bg.convert("L")
        lo, hi = im.getextrema()
        if hi - lo < 4:
            bg = Image.new("RGB", rgba.size, (0, 0, 0))
            bg.paste(rgba, mask=rgba.split()[-1])
            im = bg.convert("L")
    else:
        im = im.convert("L")
    if np.asarray(im.resize((32, 32))).mean() < 110:
        im = Image.eval(im, lambda v: 255 - v)
    return im


class TarIndexedGlyphDataset(Dataset):
    def __init__(
        self,
        tar_dir: Path,
        indices: np.ndarray,
        shard_ids: np.ndarray,
        offsets: np.ndarray,
        sizes: np.ndarray,
        cls_ids: np.ndarray,
        sty_ids: np.ndarray,
        dyn_ids: np.ndarray,
        tf,
        u8_mode: bool = True,
        teacher_npy: Path | None = None,
    ):
        self.shards = [str(p) for p in sorted(tar_dir.glob("clean_base_*.tar"))]
        self.indices = indices
        self.shard_ids = shard_ids
        self.offsets = offsets
        self.sizes = sizes
        self.cls_ids = cls_ids
        self.sty_ids = sty_ids
        self.dyn_ids = dyn_ids
        self.tf = tf
        self.u8_mode = u8_mode
        self.teacher_npy = str(teacher_npy) if teacher_npy else ""
        self._fhs: list | None = None
        self._t_mmap = None

    def _ensure_handles(self):
        if self._fhs is None:
            self._fhs = [open(p, "rb") for p in self.shards]
        if self.teacher_npy and self._t_mmap is None:
            self._t_mmap = np.load(self.teacher_npy, mmap_mode="r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        self._ensure_handles()
        gidx = int(self.indices[i])
        fh = self._fhs[int(self.shard_ids[gidx])]
        fh.seek(int(self.offsets[gidx]))
        raw = fh.read(int(self.sizes[gidx]))
        try:
            x = self.tf(_decode_glyph_bytes_cv2(raw)) if self.u8_mode else self.tf(_decode_glyph_bytes(raw))
        except Exception:
            x = torch.full((1, 224, 224), 255, dtype=torch.uint8) if self.u8_mode else torch.zeros(3, 224, 224)
        cid = int(self.cls_ids[gidx])
        sid = int(self.sty_ids[gidx])
        did = int(self.dyn_ids[gidx])
        if self._t_mmap is not None:
            t_emb = torch.from_numpy(np.array(self._t_mmap[gidx], dtype=np.float32))
            return x, cid, sid, did, gidx, t_emb
        return x, cid, sid, did, gidx


class LocalPathGlyphDataset(Dataset):
    """Fallback local-directory dataset for fast local smoke tests without tar files."""
    def __init__(self, items: list[tuple[str, int, int, int]], tf, u8_mode: bool = True, teacher_npy: Path | None = None):
        self.items = items
        self.tf = tf
        self.u8_mode = u8_mode
        self.teacher_npy = str(teacher_npy) if teacher_npy else ""
        self._t_mmap = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        if self.teacher_npy and self._t_mmap is None:
            self._t_mmap = np.load(self.teacher_npy, mmap_mode="r")
        p, cid, sid, did = self.items[i]
        try:
            if self.u8_mode:
                x = self.tf(_decode_glyph_bytes_cv2(Path(p).read_bytes()))
            else:
                x = self.tf(load_glyph(p))
        except Exception:
            x = torch.full((1, 224, 224), 255, dtype=torch.uint8) if self.u8_mode else torch.zeros(3, 224, 224)
        if self._t_mmap is not None:
            t_emb = torch.from_numpy(np.array(self._t_mmap[i], dtype=np.float32))
            return x, cid, sid, did, i, t_emb
        return x, cid, sid, did, i


# ---------- 3. DDP, State Dict Unwrap & Training / Caching Helpers ----------

def setup_ddp() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world, local_rank, torch.device(f"cuda:{local_rank}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, dev


def clean_state_dict(model_raw: GlyphModelV12) -> dict[str, torch.Tensor]:
    """Strip `_orig_mod.` prefix from compiled backbone so checkpoints are 100% portable across L4/A100/T4/CPU!"""
    out = {}
    for k, v in model_raw.state_dict().items():
        ck = k.replace("backbone._orig_mod.", "backbone.")
        out[ck] = v.detach().cpu().contiguous()
    return out


def load_clean_state_dict(model_raw: GlyphModelV12, sd: dict[str, torch.Tensor]) -> None:
    """Load portable state_dict cleanly whether `model_raw.backbone` is compiled or not."""
    is_compiled = hasattr(model_raw.backbone, "_orig_mod")
    mapped = {}
    for k, v in sd.items():
        ck = k.replace("backbone._orig_mod.", "backbone.")
        if is_compiled and ck.startswith("backbone."):
            ck = "backbone._orig_mod." + ck[len("backbone."):]
        mapped[ck] = v
    model_raw.load_state_dict(mapped, strict=True)


def save_checkpoint(
    path: Path,
    model_raw: GlyphModelV12,
    opt: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    step_in_ep: int,
    global_step: int,
    meta_cfg: dict,
    weights_only_ckpt: bool = False,
):
    tmp = path.with_suffix(".tmp.pt")
    payload = {
        "arch": model_raw.arch,
        "model": clean_state_dict(model_raw),
        "epoch": epoch,
        "step_in_ep": step_in_ep,
        "global_step": global_step,
        "meta_cfg": meta_cfg,
    }
    if not weights_only_ckpt and opt is not None:
        payload["opt"] = opt.state_dict()
        if scaler is not None and scaler.is_enabled():
            payload["scaler"] = scaler.state_dict()
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.no_grad()
def run_cache_teacher(
    args,
    model_raw: GlyphModelV12,
    ds_all: Dataset,
    gpu_aug: GpuGlyphAugment | None,
    amp_dtype: torch.dtype,
    channels_last: bool,
    rank: int,
    world: int,
    device: torch.device,
    syncer: CheckpointSyncer | None,
):
    N = len(ds_all)
    if rank == 0:
        print(f"[CacheTeacher] Extracting 512-d fp16 teacher embeddings for {N:,} glyphs (world={world}, dtype={amp_dtype})...", flush=True)
    sampler = DistributedSampler(ds_all, num_replicas=world, rank=rank, shuffle=False, drop_last=False) if world > 1 else None
    dl_kw = {"prefetch_factor": 4} if args.workers > 0 else {}
    loader = DataLoader(
        ds_all, batch_size=args.batch_size * 2, shuffle=False, sampler=sampler,
        num_workers=args.workers, pin_memory=(device.type == "cuda"), **dl_kw,
    )
    prefetcher = CUDAPrefetcher(loader, device)

    local_idxs, local_embs = [], []
    model_raw.eval()
    t0 = time.time()
    for bi, batch in enumerate(prefetcher):
        x, _, _, _, gidx = batch[:5]
        if gpu_aug is not None:
            x = gpu_aug(x, training=False)
        elif channels_last and device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            e = model_raw.embed(x)
        local_idxs.append(gidx.cpu().numpy().astype(np.int64))
        local_embs.append(e.half().cpu().numpy())
        if rank == 0 and (bi + 1) % 100 == 0:
            done = (bi + 1) * args.batch_size * 2 * world
            ips = min(done, N) / max(1.0, time.time() - t0)
            print(f"  cached ~{min(done, N):,}/{N:,} | {ips:.0f} img/s ({time.time()-t0:.1f}s)", flush=True)

    part_p = args.out_dir / f"_teacher_part_r{rank}.npz"
    np.savez(part_p, idx=np.concatenate(local_idxs), emb=np.concatenate(local_embs, axis=0))
    if world > 1:
        dist.barrier()

    if rank == 0:
        out_npy = args.out_dir / "teacher_embs_fp16.npy"
        full = np.zeros((N, 512), dtype=np.float16)
        for r in range(world):
            pp = args.out_dir / f"_teacher_part_r{r}.npz"
            z = np.load(pp)
            full[z["idx"]] = z["emb"]
            pp.unlink()
        np.save(out_npy, full)
        print(f"[CacheTeacher] Wrote {out_npy} ({out_npy.stat().st_size/1e9:.2f} GB) in {(time.time()-t0)/60:.1f}m", flush=True)
        if syncer is not None:
            ok = syncer.upload_sync(out_npy, "teacher_embs_fp16.npy")
            print(f"[CacheTeacher] GDrive sync ok={ok}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="汉火识字 Clean V12 (方案 3C) Colab L4/A100 & Kaggle 2×T4 通用高性能训练引擎")
    ap.add_argument("--mode", choices=["teacher", "cache_teacher", "student"], required=True)
    ap.add_argument("--arch", default="", help="留空自动: teacher=convnext_base, student=convnext_tiny")
    ap.add_argument("--tar-dir", type=Path, default=None, help="含 clean_base_00..15.tar 与 meta 的目录")
    ap.add_argument("--local-all-data", type=Path, default=None, help="本地目录模式（冒烟测试用）")
    ap.add_argument("--ids-json", type=Path, default=SHIZI_ROOT / "checkpoints/classes_meta/ids_components.json")
    ap.add_argument("--out-dir", type=Path, default=Path("/content/v12_run"))
    ap.add_argument("--teacher-ckpt", type=Path, default=None, help="cache_teacher 模式加载的教师权重")
    ap.add_argument("--teacher-npy", type=Path, default=None, help="student 模式读取的离线教师特征 .npy")
    ap.add_argument("--gdrive-remote", type=str, default="wylggww:shizi_v12", help="rclone 远程目录（空串禁用）")
    ap.add_argument("--auto-resume", action="store_true", help="启动时自动从本地或 GDrive 拉取 ckpt_latest.pt 续训")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=128, help="每卡微批大小（L4 Teacher 128×2=256 / Student 256×1；T4 Teacher 64×2）")
    ap.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数")
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--subcenters", type=int, default=3)
    ap.add_argument("--neg-ratio", type=float, default=0.20, help="PartialFC 负类采样比例")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers（0=自动按 CPU 核数优化）")
    ap.add_argument("--save-every-steps", type=int, default=600, help="每隔 N 个优化步异步备份 ckpt_latest.pt 至 GDrive")
    ap.add_argument("--max-hours", type=float, default=11.2, help="会话安全停机门禁（小时）")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto", help="auto: SM>=8.0(L4/A100)用bf16, T4用fp16")
    ap.add_argument("--compile", choices=["auto", "on", "off"], default="auto", help="auto: SM>=8.0 自动 torch.compile(backbone)")
    ap.add_argument("--no-gpu-aug", action="store_true", help="关闭 GPU 端批量增强（回退经典 CPU PIL 增强）")
    ap.add_argument("--no-channels-last", action="store_true", help="关闭 NHWC channels_last 内存布局")
    ap.add_argument("--smoke-classes", type=int, default=0, help=">0 时仅取 N 类做快速冒烟测试")
    ap.add_argument("--max-steps", type=int, default=0, help=">0 时最多跑 N 步即退出（冒烟测试用）")
    args = ap.parse_args()

    t_start = time.time()
    rank, world, local_rank, device = setup_ddp()
    is_rank0 = rank == 0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Hardware-aware kernel tuning (Ada L4 / Ampere A100 / Turing T4)
    sm_major, sm_minor = (0, 0)
    gpu_name = "CPU"
    if device.type == "cuda":
        sm_major, sm_minor = torch.cuda.get_device_capability(device)
        gpu_name = torch.cuda.get_device_name(device)
        torch.backends.cudnn.benchmark = True
        if sm_major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    if args.precision == "bf16" or (args.precision == "auto" and sm_major >= 8 and torch.cuda.is_bf16_supported()):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    use_scaler = (device.type == "cuda" and amp_dtype == torch.float16)
    channels_last = (device.type == "cuda" and not args.no_channels_last)
    use_gpu_aug = (device.type == "cuda" and not args.no_gpu_aug)
    use_compile = (
        device.type == "cuda"
        and args.mode != "cache_teacher"
        and (args.compile == "on" or (args.compile == "auto" and sm_major >= 8))
    )
    if args.workers <= 0:
        cpu_cnt = os.cpu_count() or 4
        args.workers = min(12, max(2, cpu_cnt // max(1, world))) if device.type == "cuda" else 0

    arch = args.arch or ("convnext_base" if args.mode in ("teacher", "cache_teacher") else "convnext_tiny")
    mode_sub = "teacher" if args.mode in ("teacher", "cache_teacher") else "student"

    if is_rank0:
        print(
            f"[Hardware] {gpu_name} (SM {sm_major}.{sm_minor}, world={world}) | "
            f"dtype={str(amp_dtype).split('.')[-1]} tf32={sm_major >= 8} channels_last={channels_last} "
            f"gpu_aug={use_gpu_aug} compile={use_compile} workers={args.workers}",
            flush=True,
        )

    syncer = None
    if is_rank0 and args.gdrive_remote:
        ensure_rclone_conf()
        syncer = CheckpointSyncer(remote=f"{args.gdrive_remote.rstrip('/')}/{mode_sub}", log_path=args.out_dir / "sync.log")
        if not syncer.healthy():
            print("[WARN] rclone remote unreachable; disabling GDrive sync!", flush=True)
            syncer = None
        else:
            print(f"[GDrive] Async CheckpointSyncer active -> {syncer.remote}", flush=True)

    # Dataset loading (Tar zero-unpack OR local directory smoke mode)
    train_tf, val_tf = make_fast_u8_transforms() if use_gpu_aug else make_transforms()
    gpu_aug = GpuGlyphAugment(device, channels_last=channels_last) if use_gpu_aug else None

    if args.tar_dir is not None:
        cache_npz = args.tar_dir / "tar_index.npz"
        classes, styles, dynasties, sh_ids, offs, szs, cids, sids, dids = build_or_load_tar_index(args.tar_dir, cache_npz, is_rank0=is_rank0)
        if world > 1:
            dist.barrier()
        tr_idx, va_idx = make_deterministic_split(cids, smoke_limit=args.smoke_classes)
        all_idx = np.arange(len(cids), dtype=np.int64)
        t_npy = args.teacher_npy
        if args.mode == "student" and t_npy is not None and not t_npy.exists() and is_rank0 and args.gdrive_remote:
            t_sync = CheckpointSyncer(remote=f"{args.gdrive_remote.rstrip('/')}/teacher", log_path=args.out_dir / "sync.log")
            print(f"[GDrive] Pulling teacher_embs_fp16.npy -> {t_npy} ...", flush=True)
            t_sync.download_if_exists("teacher_embs_fp16.npy", t_npy, timeout=1800.0)
        if world > 1:
            dist.barrier()
        ds_train = TarIndexedGlyphDataset(
            args.tar_dir, tr_idx, sh_ids, offs, szs, cids, sids, dids, train_tf,
            u8_mode=use_gpu_aug, teacher_npy=t_npy if args.mode == "student" else None,
        )
        ds_all = TarIndexedGlyphDataset(
            args.tar_dir, all_idx, sh_ids, offs, szs, cids, sids, dids, val_tf,
            u8_mode=use_gpu_aug,
        )
    elif args.local_all_data is not None:
        work_dir = args.local_all_data.parent
        classes = json.loads((work_dir / "classes.json").read_text(encoding="utf-8"))
        styles = json.loads((work_dir / "styles.json").read_text(encoding="utf-8"))
        dynasties = json.loads((work_dir / "dynasties.json").read_text(encoding="utf-8"))
        cls2id = {c: i for i, c in enumerate(classes)}
        sty2id = {s: i for i, s in enumerate(styles)}
        dyn2id = {d: i for i, d in enumerate(dynasties)}
        metas = json.loads((work_dir / "metas.json").read_text(encoding="utf-8"))
        rng = random.Random(42)
        if args.smoke_classes > 0:
            sel = set(rng.sample(classes, min(args.smoke_classes, len(classes))))
            metas = [m for m in metas if m[1] in sel]
        items = [(m[0], cls2id[m[1]], sty2id.get(m[2], 0), dyn2id.get(m[3], 0)) for m in metas if m[1] in cls2id]
        ds_train = LocalPathGlyphDataset(items, train_tf, u8_mode=use_gpu_aug, teacher_npy=args.teacher_npy if args.mode == "student" else None)
        ds_all = LocalPathGlyphDataset(items, val_tf, u8_mode=use_gpu_aug)
    else:
        raise ValueError("必须指定 --tar-dir 或 --local-all-data")

    num_ids, cls2ids = build_class_to_ids_map(classes, args.ids_json)
    ids_table = GpuIdsTable(cls2ids, num_ids, device) if num_ids > 0 else None
    if is_rank0:
        print(
            f"[Model] mode={args.mode} arch={arch} | classes={len(classes):,} K={args.subcenters} neg={args.neg_ratio} "
            f"| IDS={num_ids} | train={len(ds_train):,} | eff_bs={args.batch_size * max(1, args.grad_accum) * world}",
            flush=True,
        )

    model_raw = GlyphModelV12(
        arch=arch, num_classes=len(classes), num_styles=len(styles),
        num_dynasties=len(dynasties), num_ids=num_ids,
        subcenters=args.subcenters, neg_ratio=args.neg_ratio,
        pretrained=(args.mode != "cache_teacher"),
    ).to(device)
    if channels_last:
        model_raw.backbone.to(memory_format=torch.channels_last)

    # Mode: cache_teacher
    if args.mode == "cache_teacher":
        ckpt_p = args.teacher_ckpt or (args.out_dir / "ckpt_latest.pt")
        if not ckpt_p.exists() and is_rank0 and syncer is not None:
            syncer.download_if_exists("ckpt_latest.pt", ckpt_p)
        if world > 1:
            dist.barrier()
        ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        load_clean_state_dict(model_raw, ck["model"])
        run_cache_teacher(args, model_raw, ds_all, gpu_aug, amp_dtype, channels_last, rank, world, device, syncer)
        if world > 1:
            dist.destroy_process_group()
        return

    # Mode: teacher OR student training (resume BEFORE torch.compile for clean state_dict!)
    bb_params, head_params = [], []
    for name, p in model_raw.named_parameters():
        (bb_params if name.startswith("backbone.") else head_params).append(p)
    opt_kw = {"fused": True} if device.type == "cuda" else {}
    try:
        opt = torch.optim.AdamW([
            {"params": bb_params, "lr": args.lr, "weight_decay": 1e-4},
            {"params": head_params, "lr": args.lr * 3.0, "weight_decay": 1e-4},
        ], **opt_kw)
    except Exception:
        opt = torch.optim.AdamW([
            {"params": bb_params, "lr": args.lr, "weight_decay": 1e-4},
            {"params": head_params, "lr": args.lr * 3.0, "weight_decay": 1e-4},
        ])
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_ep, resume_step_in_ep, global_step = 0, 0, 0
    latest_p = args.out_dir / "ckpt_latest.pt"
    if args.auto_resume:
        if is_rank0 and not latest_p.exists() and syncer is not None:
            if syncer.download_if_exists("ckpt_latest.pt", latest_p):
                print("[Resume] Pulled ckpt_latest.pt from GDrive!", flush=True)
        if world > 1:
            dist.barrier()
        if latest_p.exists():
            ck = torch.load(latest_p, map_location="cpu", weights_only=False)
            load_clean_state_dict(model_raw, ck["model"])
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
            if use_scaler and "scaler" in ck:
                scaler.load_state_dict(ck["scaler"])
            start_ep = int(ck.get("epoch", 0))
            resume_step_in_ep = int(ck.get("step_in_ep", 0))
            global_step = int(ck.get("global_step", 0))
            if is_rank0:
                print(f"[Resume] Restored ep={start_ep} step_in_ep={resume_step_in_ep} global_step={global_step}", flush=True)

    # Selectively compile ONLY static-shape ConvNeXt backbone (zero graph breaks from dynamic PartialFC!)
    if use_compile:
        try:
            model_raw.backbone = torch.compile(model_raw.backbone, mode="default", dynamic=False)
            if is_rank0:
                print("[Compile] torch.compile(backbone, mode='default') enabled!", flush=True)
        except Exception as e:
            if is_rank0:
                print(f"[Compile] Fallback to eager mode: {e}", flush=True)

    model = DDP(model_raw, device_ids=[local_rank], find_unused_parameters=False) if world > 1 else model_raw

    grad_accum = max(1, int(args.grad_accum))
    # Instant zero-waste epoch resume sampler (skips already-completed optimizer steps in 0.001s!)
    def make_epoch_loader(ep_idx: int, skip_opt_steps: int) -> DataLoader:
        g = torch.Generator()
        g.manual_seed(42 + ep_idx)
        perm = torch.randperm(len(ds_train), generator=g).tolist()
        if world > 1:
            per_rank = (len(perm) // world) * world
            perm = perm[rank:per_rank:world]
        skip_samples = skip_opt_steps * args.batch_size * grad_accum
        if skip_samples > 0:
            perm = perm[skip_samples:]
        dl_kw = {"prefetch_factor": 6, "persistent_workers": False} if args.workers > 0 else {}
        return DataLoader(
            ds_train, batch_size=args.batch_size, sampler=perm,
            num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True, **dl_kw,
        )

    eff_per_rank_samples = len(ds_train) // max(1, world)
    steps_per_ep = max(1, (eff_per_rank_samples // args.batch_size) // grad_accum)
    total_steps = args.epochs * steps_per_ep
    warmup_steps = min(steps_per_ep, 1500)

    def set_lr(gstep: int):
        if gstep < warmup_steps:
            mult = (gstep + 1) / max(1, warmup_steps)
        else:
            prog = (gstep - warmup_steps) / max(1, total_steps - warmup_steps)
            mult = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        opt.param_groups[0]["lr"] = args.lr * mult
        opt.param_groups[1]["lr"] = args.lr * 3.0 * mult

    ids_pos_w = torch.full((num_ids,), 12.0, device=device) if num_ids > 0 else None
    meta_cfg = {
        "arch": arch, "mode": args.mode, "num_classes": len(classes),
        "num_styles": len(styles), "num_dynasties": len(dynasties),
        "subcenters": args.subcenters, "num_ids": num_ids,
    }

    zero_t = torch.zeros((), device=device, dtype=torch.float32)
    stop_early = False
    for ep in range(start_ep, args.epochs):
        skip_steps = resume_step_in_ep if ep == start_ep else 0
        loader = make_epoch_loader(ep, skip_steps)
        model.train()
        margin_cur = 0.20 + 0.25 * min(1.0, ep / 3.0)
        acc_tot = torch.zeros((), device=device)
        acc_arc = torch.zeros((), device=device)
        acc_ids = torch.zeros((), device=device)
        acc_cos = torch.zeros((), device=device)
        acc_rkd = torch.zeros((), device=device)
        n_log = 0
        t_ep = time.time()
        opt.zero_grad(set_to_none=True)
        opt_step_in_ep = skip_steps

        prefetcher = CUDAPrefetcher(loader, device)
        for micro_i, batch in enumerate(prefetcher, start=1):
            if args.mode == "student":
                x, cid, sid, did, _, t_emb = batch
            else:
                x, cid, sid, did, _ = batch
                t_emb = None

            if gpu_aug is not None:
                x = gpu_aug(x, training=True)
            elif channels_last and device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)

            is_sync_step = (micro_i % grad_accum == 0) or (micro_i == len(loader))
            ctx = model.no_sync() if (world > 1 and not is_sync_step) else contextlib.nullcontext()
            with ctx:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    emb, l_arc, sty_logits, dyn_logits, ids_logits = model(x, cid, margin_cur)
                    l_sty = F.cross_entropy(sty_logits, sid)
                    l_dyn = F.cross_entropy(dyn_logits, did)
                    loss = l_arc + 0.15 * l_sty + 0.08 * l_dyn

                    l_ids = zero_t
                    if ids_logits is not None and ids_table is not None:
                        ids_tgt, ids_val = ids_table.lookup(cid)
                        # Masked per-sample BCE without data-dependent control flow sync!
                        bce_per = F.binary_cross_entropy_with_logits(
                            ids_logits, ids_tgt, pos_weight=ids_pos_w, reduction="none"
                        ).mean(dim=1)
                        v_f = ids_val.float()
                        l_ids = (bce_per * v_f).sum() / v_f.sum().clamp(min=1.0)
                        loss = loss + 0.35 * l_ids

                    l_cos = zero_t
                    l_rkd = zero_t
                    if t_emb is not None:
                        l_cos, l_rkd = distillation_loss(emb, t_emb)
                        loss = loss + 4.0 * l_cos + 2.0 * l_rkd

                if use_scaler:
                    scaler.scale(loss / grad_accum).backward()
                else:
                    (loss / grad_accum).backward()

            # Zero-sync GPU tensor accumulation!
            acc_tot.add_(loss.detach())
            acc_arc.add_(l_arc.detach())
            acc_ids.add_(l_ids.detach())
            acc_cos.add_(l_cos.detach())
            acc_rkd.add_(l_rkd.detach())
            n_log += 1

            if not is_sync_step:
                continue

            set_lr(global_step)
            if use_scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            opt.zero_grad(set_to_none=True)

            global_step += 1
            opt_step_in_ep = skip_steps + (micro_i // grad_accum)

            if is_rank0 and (opt_step_in_ep == skip_steps + 1 or opt_step_in_ep % 100 == 0 or opt_step_in_ep >= steps_per_ep):
                elap = max(1.0, time.time() - t_ep)
                ips = (micro_i * args.batch_size * world) / elap
                vram_gb = (torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0
                inv_n = 1.0 / max(1, n_log)
                dist_str = (
                    f" | cos={acc_cos.item()*inv_n:.4f} rkd={acc_rkd.item()*inv_n:.4f}"
                    if args.mode == "student" else ""
                )
                print(
                    f"[Ep {ep+1:02d}/{args.epochs} | {opt_step_in_ep:4d}/{steps_per_ep}] "
                    f"tot={acc_tot.item()*inv_n:.3f} arc={acc_arc.item()*inv_n:.3f} ids={acc_ids.item()*inv_n:.3f}{dist_str} "
                    f"| lr={opt.param_groups[0]['lr']:.2e} | {ips:.0f} img/s | vram={vram_gb:.1f}G | {(time.time()-t_start)/3600:.2f}h",
                    flush=True,
                )
                acc_tot.zero_()
                acc_arc.zero_()
                acc_ids.zero_()
                acc_cos.zero_()
                acc_rkd.zero_()
                n_log = 0

            # Periodic async backup to Google Drive (zero training stall!)
            if is_rank0 and args.save_every_steps > 0 and (global_step % args.save_every_steps == 0):
                save_checkpoint(latest_p, model_raw, opt, scaler, ep, opt_step_in_ep, global_step, meta_cfg)
                if syncer is not None:
                    syncer.upload_async(latest_p, "ckpt_latest.pt")

            # Wall-clock guard
            should_stop = 0
            if is_rank0:
                if (time.time() - t_start) >= args.max_hours * 3600.0:
                    should_stop = 1
                if args.max_steps > 0 and global_step >= args.max_steps:
                    should_stop = 2
            if world > 1:
                flag = torch.tensor([should_stop], device=device)
                dist.broadcast(flag, src=0)
                should_stop = int(flag.item())
            if should_stop:
                if is_rank0:
                    print(f"[Guard] Clean early stop (reason={should_stop}) at ep={ep+1} step={opt_step_in_ep}!", flush=True)
                    save_checkpoint(latest_p, model_raw, opt, scaler, ep, opt_step_in_ep, global_step, meta_cfg)
                    if syncer is not None:
                        syncer.upload_sync(latest_p, "ckpt_latest.pt")
                stop_early = True
                break

        if stop_early:
            break

        # End of epoch: save full latest + lightweight milestone weights asynchronously to GDrive
        resume_step_in_ep = 0
        if is_rank0:
            ep_p = args.out_dir / f"ckpt_ep{ep+1:02d}.pt"
            save_checkpoint(latest_p, model_raw, opt, scaler, ep + 1, 0, global_step, meta_cfg)
            save_checkpoint(ep_p, model_raw, None, None, ep + 1, 0, global_step, meta_cfg, weights_only_ckpt=True)
            print(f"[Epoch {ep+1:02d} Done] {(time.time()-t_ep)/60:.1f}m -> saved {ep_p.name}", flush=True)
            if syncer is not None:
                syncer.upload_async(latest_p, "ckpt_latest.pt")
                syncer.upload_async(ep_p, ep_p.name)
            old_eps = sorted(args.out_dir.glob("ckpt_ep*.pt"))[:-2]
            for old in old_eps:
                try:
                    old.unlink()
                except OSError:
                    pass

    # Finalize: wait for all background GDrive uploads + export ONNX if student finished!
    if is_rank0:
        if args.mode == "student" and not stop_early:
            onnx_p = args.out_dir / f"{arch}_v12_op17_wo_u8.onnx"
            export_v12_wo_u8_opset17(model_raw, onnx_p)
            if syncer is not None:
                syncer.upload_sync(onnx_p, onnx_p.name)
        if syncer is not None:
            syncer.wait_all()
            syncer.upload_sync(args.out_dir / "sync.log", "sync.log")
        print(f"=== V12 ({args.mode}/{arch}) finished cleanly in {(time.time()-t_start)/60:.1f}m ===", flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
