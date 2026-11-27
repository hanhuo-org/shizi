#!/usr/bin/env python3
"""
Core model architecture, augmentations, polarity normalization, and WiSE-FT utilities
for Hanfire Shizi (汉火识字 Clean V10).

100% state_dict-compatible with released checkpoints (`wiseft_v10_a070.pt`).
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from opencc import OpenCC
    _cc = OpenCC("t2s")
    _cc_cache: dict[str, str] = {}

    def simp(ch: str) -> str:
        if ch not in _cc_cache:
            s = _cc.convert(ch)
            _cc_cache[ch] = s if len(s) == 1 else ch
        return _cc_cache[ch]
except ImportError:
    def simp(ch: str) -> str:
        return ch


def is_valid_cjk(ch: str) -> bool:
    if not ch or len(ch) != 1:
        return False
    cp = ord(ch)
    return (0x3400 <= cp <= 0x9FFF) or (0x20000 <= cp <= 0x2FA1F)


# ---------- Image Loading & Preprocessing ----------

def load_glyph(path: str | Path) -> Image.Image:
    """Load glyph image with alpha compositing onto white background and polarity normalization (black ink on white)."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg.convert("L")
        lo, hi = img.getextrema()
        if hi - lo < 4:
            bg = Image.new("RGB", rgba.size, (0, 0, 0))
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg.convert("L")
    else:
        img = img.convert("L")
    if np.asarray(img.resize((32, 32))).mean() < 110:
        img = Image.eval(img, lambda v: 255 - v)
    return img


def load_rubbing_autoinv(path: str | Path) -> Image.Image:
    """Auto-inverted & contrast-stretched grayscale rubbing (matches browser Hybrid 0.65*bw + 0.35*gray TTA)."""
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32)
    if arr.mean() < 127.0:
        im = ImageOps.invert(im)
        arr = np.asarray(im, dtype=np.float32)
    lo, hi = np.percentile(arr, 4), np.percentile(arr, 96)
    if hi > lo + 10:
        arr = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        im = Image.fromarray(arr, mode="L")
    return im


def preprocess_upload(pil_img: Image.Image, size: int = 256) -> Image.Image:
    """Standard single-character upload preprocessing:
    CIE-Lab a* red-seal extraction (朱文/白文) + uneven-shadow flattening + Hybrid Sauvola + border strip."""
    if pil_img.mode in ("RGBA", "LA", "P"):
        rgba = pil_img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        pil_img = bg
    rgb_u8 = np.asarray(pil_img.convert("RGB"))
    H0, W0 = rgb_u8.shape[:2]
    cx0, cx1 = int(W0 * 0.15), max(int(W0 * 0.15) + 1, int(W0 * 0.85))
    cy0, cy1 = int(H0 * 0.15), max(int(H0 * 0.15) + 1, int(H0 * 0.85))

    # 1) CIE-Lab a* red-seal ink channel (immune to yellowed paper/bamboo; supports both 朱文 and 白文)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_ch = lab[:, :, 1] - 128.0
    rf, gf, bf = rgb_u8[:, :, 0].astype(np.float32), rgb_u8[:, :, 1].astype(np.float32), rgb_u8[:, :, 2].astype(np.float32)
    red_m = (a_ch > 15.0) & ((rf - gf) > 14.0) & ((rf - bf) > 10.0)
    red_frac = float(red_m.mean())
    core_red_frac = float(red_m[cy0:cy1, cx0:cx1].mean())
    red_in_core = float(red_m[cy0:cy1, cx0:cx1].sum()) / max(1.0, float(red_m.sum()))

    bw = None
    if red_frac > 0.025 and core_red_frac >= 0.03 and red_in_core >= 0.42:
        a_u8 = np.clip(a_ch + 128.0, 0, 255).astype(np.uint8)
        thr_raw, _ = cv2.threshold(a_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo_m = a_u8 <= thr_raw
        hi_m = a_u8 > thr_raw
        if lo_m.any() and hi_m.any():
            m_lo, m_hi = float(a_u8[lo_m].mean()), float(a_u8[hi_m].mean())
            thr_a = 0.5 * (m_lo + m_hi)
            if thr_a > 136.0 and (m_hi - m_lo) >= 12.0:
                is_r = a_u8 >= thr_a
                c_frac = float(is_r[cy0:cy1, cx0:cx1].mean())
                if 0.025 <= c_frac <= 0.88:
                    if c_frac > 0.52:  # 白文印 (white strokes in red seal)
                        bw = np.where(is_r, 255, 0).astype(np.uint8)
                    else:  # 朱文印 (red strokes on paper)
                        bw = np.where(is_r, 0, 255).astype(np.uint8)
                        k_cl = 5 if min(H0, W0) >= 96 else 3
                        bw = 255 - cv2.morphologyEx(255 - bw, cv2.MORPH_CLOSE, np.ones((k_cl, k_cl), np.uint8))

    if bw is None:
        rgb = rgb_u8.astype(np.int16)
        diff = np.stack([
            np.abs(rgb[:, :, 0] - rgb[:, :, 1]),
            np.abs(rgb[:, :, 0] - rgb[:, :, 2]),
            np.abs(rgb[:, :, 1] - rgb[:, :, 2]),
        ], -1).max(-1)
        colored = (diff > 30).mean()
        if 0.02 < colored < 0.60 and float((diff[cy0:cy1, cx0:cx1] > 30).mean()) >= 0.025:
            bw = np.where(diff > 30, 0, 255).astype(np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        else:
            g = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
            int_g = g[int(H0 * 0.1):max(int(H0 * 0.1) + 1, int(H0 * 0.9)),
                      int(W0 * 0.1):max(int(W0 * 0.1) + 1, int(W0 * 0.9))]
            is_wob = float(np.median(int_g)) < 126.0
            if is_wob:
                g = 255 - g
            g_f = g.astype(np.float32)
            # Gentle background illumination flattening only on large uneven-shadow photos
            if min(H0, W0) >= 240:
                bg_k = int(max(65, min(181, (min(H0, W0) // 2) | 1)))
                mu_bg = cv2.boxFilter(g_f, -1, (bg_k, bg_k), borderType=cv2.BORDER_REPLICATE)
                if float(mu_bg.max() - mu_bg.min()) > 48.0:
                    sq_bg = cv2.boxFilter(g_f * g_f, -1, (bg_k, bg_k), borderType=cv2.BORDER_REPLICATE)
                    sd_bg = np.sqrt(np.maximum(sq_bg - mu_bg * mu_bg, 0.0))
                    est_bg = np.clip(mu_bg + 0.8 * sd_bg, 60.0, 255.0)
                    g_f = np.clip(g_f * (240.0 / est_bg), 0.0, 255.0)
            blk = int(max(17, min(161, (min(H0, W0) // 3) | 1)))
            mu = cv2.boxFilter(g_f, -1, (blk, blk), borderType=cv2.BORDER_REPLICATE)
            sq = cv2.boxFilter(g_f * g_f, -1, (blk, blk), borderType=cv2.BORDER_REPLICATE)
            sd = np.sqrt(np.maximum(sq - mu * mu, 0.0))
            c_base = 18.0 if is_wob else 12.0
            c_adapt = c_base + 0.25 * np.maximum(0.0, 48.0 - sd)
            bw = np.where(g_f < mu - c_adapt, 0, 255).astype(np.uint8)
    H, W = bw.shape
    for _ in range(4):
        dark = bw == 0
        rows_hit = np.where(dark.mean(1) > 0.88)[0]
        cols_hit = np.where(dark.mean(0) > 0.88)[0]
        if len(rows_hit) == 0 and len(cols_hit) == 0:
            break
        dirs = (
            int((rows_hit < H * 0.35).any()) + int((rows_hit > H * 0.65).any())
            + int((cols_hit < W * 0.35).any()) + int((cols_hit > W * 0.65).any())
        )
        if dirs < 3:
            break
        bw[rows_hit, :] = 255
        bw[:, cols_hit] = 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(255 - bw, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.where(areas >= max(4, areas.max() * 0.03))[0] + 1
        x = int(stats[keep, cv2.CC_STAT_LEFT].min())
        y = int(stats[keep, cv2.CC_STAT_TOP].min())
        x1 = int((stats[keep, cv2.CC_STAT_LEFT] + stats[keep, cv2.CC_STAT_WIDTH]).max())
        y1 = int((stats[keep, cv2.CC_STAT_TOP] + stats[keep, cv2.CC_STAT_HEIGHT]).max())
        pad = max(4, min(14, int(max(x1 - x, y1 - y) * 0.05)))
        bw = bw[max(0, y - pad) : min(H, y1 + pad), max(0, x - pad) : min(W, x1 + pad)]
    im = Image.fromarray(bw)
    im.thumbnail((208, 208), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


# ---------- Domain Augmentations ----------

class RandomNoise:
    def __init__(self, p: float = 0.25):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        a = np.asarray(img).astype(np.int16)
        if random.random() < 0.5:
            sigma = random.uniform(5, 15)
            a = a + np.random.normal(0, sigma, a.shape).astype(np.int16)
        else:
            n = int(a.size * random.uniform(0.002, 0.01))
            ys = np.random.randint(0, a.shape[0], n)
            xs = np.random.randint(0, a.shape[1], n)
            vals = np.random.choice(np.array([0, 255], dtype=np.int16), n)
            if a.ndim == 2:
                a[ys, xs] = vals
            else:
                a[ys, xs, :] = vals[:, None]
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


class RandomMorphology:
    """Random stroke dilation/erosion simulating heavy rubbing ink vs weathered thin strokes."""
    def __init__(self, p: float = 0.35):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.asarray(img)
        k = random.choice([2, 3])
        kernel = np.ones((k, k), np.uint8)
        if random.random() < 0.5:
            arr = cv2.erode(arr, kernel, iterations=1)  # thicken black strokes
        else:
            arr = cv2.dilate(arr, kernel, iterations=1)  # thin black strokes
        return Image.fromarray(arr)


class RandomStoneSpeckles:
    """Simulates stone rubbing weathering pits (石花) and bamboo/silk grain noise."""
    def __init__(self, p: float = 0.28):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.asarray(img).copy()
        h, w = arr.shape[:2]
        n = random.randint(6, 28)
        for _ in range(n):
            cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
            r = random.randint(1, max(2, min(h, w) // 45))
            val = 0 if random.random() < 0.55 else 255
            cv2.circle(arr, (cx, cy), r, int(val), -1)
        return Image.fromarray(arr)


class RandomBoneCracksAndBorders:
    """Simulates Yin Ruins turtle-shell burn cracks, chipped borders, and bronze seam lines."""
    def __init__(self, p: float = 0.24):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.asarray(img).copy()
        h, w = arr.shape[:2]
        if random.random() < 0.55:
            x0, y0 = random.randint(0, w - 1), random.randint(0, h // 3)
            x1, y1 = random.randint(0, w - 1), random.randint(2 * h // 3, h - 1)
            cv2.line(arr, (x0, y0), (x1, y1), 0 if random.random() < 0.70 else 255, thickness=random.randint(1, 3))
        else:
            edge = random.choice(["top", "bottom", "left", "right"])
            d = random.randint(max(6, h // 14), max(12, h // 6))
            v = random.randint(0, 40)
            if edge == "top":
                arr[:d, :] = np.minimum(arr[:d, :], v)
            elif edge == "bottom":
                arr[h - d :, :] = np.minimum(arr[h - d :, :], v)
            elif edge == "left":
                arr[:, :d] = np.minimum(arr[:, :d], v)
            else:
                arr[:, w - d :] = np.minimum(arr[:, w - d :], v)
        return Image.fromarray(arr)


class RandomPadToSquare:
    """Matches browser WASM 256x256 square canvas centering with slight margin jitter."""
    def __init__(self, p: float = 0.40, canvas_size: int = 256, inner_range: tuple[int, int] = (188, 214)):
        self.p = p
        self.canvas_size = canvas_size
        self.inner_range = inner_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        im = img.convert("L")
        inner = random.randint(*self.inner_range)
        im.thumbnail((inner, inner), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        ox = (self.canvas_size - im.width) // 2 + random.randint(-6, 6)
        oy = (self.canvas_size - im.height) // 2 + random.randint(-6, 6)
        canvas.paste(im, (max(0, min(self.canvas_size - im.width, ox)), max(0, min(self.canvas_size - im.height, oy))))
        return canvas


def make_transforms():
    """Base backbone training & standard evaluation transforms (matches browser WASM 256->224 pipeline)."""
    train_tf = T.Compose([
        RandomMorphology(p=0.25),
        RandomNoise(p=0.30),
        T.Grayscale(num_output_channels=3),
        T.RandomPerspective(distortion_scale=0.25, p=0.5),
        T.RandomRotation(12, fill=255),
        T.RandomAffine(0, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=255),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.ColorJitter(brightness=0.35, contrast=0.35),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12), value=1.0),
    ])
    val_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return train_tf, val_tf


def make_finetune_transforms():
    """Clean V10 domain-adaptive augmentation pipeline for real rubbings, bamboo slips, seals & steles."""
    return T.Compose([
        RandomMorphology(p=0.35),
        RandomStoneSpeckles(p=0.28),
        RandomBoneCracksAndBorders(p=0.24),
        RandomNoise(p=0.22),
        RandomPadToSquare(p=0.40, canvas_size=256, inner_range=(188, 214)),
        T.Grayscale(num_output_channels=3),
        T.RandomPerspective(distortion_scale=0.18, p=0.35, fill=255),
        T.RandomRotation(9, fill=255),
        T.RandomResizedCrop(224, scale=(0.76, 1.0), ratio=(0.60, 1.60)),
        T.ColorJitter(brightness=0.22, contrast=0.22),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
        T.RandomErasing(p=0.20, scale=(0.015, 0.085), value=1.0),
    ])


# ---------- Model Architecture ----------

class ArcFaceHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = nn.functional.normalize(emb.float())
        w = nn.functional.normalize(self.weight.float())
        cos = torch.matmul(emb, w.t()).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target_cos = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * target_cos + (1.0 - one_hot) * cos
        return logits * self.s

    def classify(self, emb: torch.Tensor) -> torch.Tensor:
        emb = nn.functional.normalize(emb)
        w = nn.functional.normalize(self.weight)
        return torch.matmul(emb, w.t()) * self.s


class GlyphModel(nn.Module):
    def __init__(self, num_classes: int, num_styles: int, num_dynasties: int, emb_dim: int = 512, pretrained: bool = False):
        super().__init__()
        weights = "IMAGENET1K_V2" if pretrained else None
        base = models.resnet50(weights=weights)
        base.fc = nn.Linear(base.fc.in_features, emb_dim)
        self.backbone = base
        self.head = ArcFaceHead(emb_dim, num_classes)
        self.style_head = nn.Linear(emb_dim, num_styles)
        self.dyn_head = nn.Linear(emb_dim, num_dynasties)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.backbone(x))

    def predict_style_dyn(self, x: torch.Tensor):
        emb = self.embed(x)
        return self.style_head(emb).argmax(1), self.dyn_head(emb).argmax(1)


def load_glyph_model(ckpt_path: str | Path, device: str = "cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    if isinstance(ck, dict) and "meta_cfg" in ck:
        from training.model_v12 import GlyphModelV12

        mc = ck["meta_cfg"]
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model = GlyphModelV12(
            arch=mc["arch"],
            num_classes=mc["num_classes"],
            num_styles=mc["num_styles"],
            num_dynasties=mc["num_dynasties"],
            num_ids=mc.get("num_ids", 0),
            subcenters=mc.get("subcenters", 3),
            pretrained=False,
        ).to(device)
        model.load_state_dict(cleaned, strict=True)
        return model, cleaned
    nc = sd["head.weight"].shape[0]
    ns = sd["style_head.weight"].shape[0]
    nd = sd["dyn_head.weight"].shape[0]
    model = GlyphModel(nc, ns, nd, pretrained=False).to(device)
    model.load_state_dict(sd, strict=True)
    return model, sd


def blend_state_dicts(sd_a: dict, sd_b: dict, alpha: float) -> dict:
    """WiSE-FT linear weight interpolation: (1 - alpha) * sd_a + alpha * sd_b."""
    out = {}
    for k, va in sd_a.items():
        vb = sd_b[k]
        if torch.is_floating_point(va):
            out[k] = ((1.0 - alpha) * va.float() + alpha * vb.float()).to(va.dtype)
        else:
            out[k] = vb.clone()
    return out
