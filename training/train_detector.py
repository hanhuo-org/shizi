#!/usr/bin/env python3
"""
汉火字形 (xing.hanhuo.org) 轻量神经单字检测器训练与导出 (Scheme B)
- 架构: MobileNetV3-Large (torchvision MIT/BSD) + Lightweight DW-Sep FPN (96ch) + Stride-4 CenterNet Head
- 输入: [B, 3, 512, 512] float32 (letterbox RGB, (x/255 - 0.5) / 0.5)
- 输出:
    hm     : [B, 1, 128, 128] sigmoid 字符中心概率图 (stride=4)
    wh_reg : [B, 4, 128, 128] [w_cells, h_cells, dx, dy] (单位: 特征图格子, 乘 4 即 512 尺度像素)
- 体积: Weight-Only Per-Channel U8 ONNX -> gzip 后仅 ~2.6~3.2 MB (按需懒加载)
- 数据: 从 126.8 万真实古文字/碑帖字形池按 18 书体分层采样, 合成 6 大真实域多字页面 (含拓片/描漆/狂草牵丝/界格/透视)
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models

ROOT = Path(__file__).resolve().parent.parent
WORK_DET = ROOT / "work/detector"
WORK_DET.mkdir(parents=True, exist_ok=True)
METAS_PATH = ROOT / "data/clean_base/clean_base_meta.jsonl.gz"
POOL_CACHE = ROOT / "data/detector_pool/glyph_pool_24k_v2_noseam.pt"

IMG_SIZE = 512
STRIDE = 4
FM_SIZE = IMG_SIZE // STRIDE  # 128

STYLE_GROUPS = {
    "cursive": ["caoshu", "zhangcao", "xingshu"],
    "regular": ["kaishu", "weibei", "lishu", "unknown"],
    "seal": ["zhuanshu", "xiaozhuan", "zhuanke", "jinshi"],
    "ancient": ["jiaguwen", "jinwen", "zhanguo", "chunqiu", "chuwenzi", "qinwenzi", "jiandu"],
}


# =====================================================================
# 1. 真字形掩码池构建（一次缓存，后续 0.5 秒加载）
# =====================================================================
def _load_one_mask(args):
    path, style, ch = args
    try:
        im = Image.open(path).convert("L")
        arr = np.asarray(im, dtype=np.uint8)
        h0, w0 = arr.shape[:2]
        if arr.ndim != 2 or h0 < 24 or w0 < 24:
            return None
        # 先裁掉最外圈 6% 防字典白边+黑底嵌套拓片误判极性
        my, mx = max(1, h0 // 16), max(1, w0 // 16)
        inner = arr[my : h0 - my, mx : w0 - mx]
        if inner.shape[0] < 16 or inner.shape[1] < 16:
            return None
        otsu_t, _ = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border = np.concatenate([inner[0, :], inner[-1, :], inner[:, 0], inner[:, -1]])
        border_med = float(np.median(border))
        if border_med < float(otsu_t):
            # 黑底白字：背景低于 otsu_t，笔画高于 otsu_t
            ink = inner.astype(np.float32)
            bg_floor = max(float(otsu_t) * 0.92, float(np.percentile(border, 85)))
            fg_ceil = float(np.percentile(ink, 99.5))
        else:
            # 白底黑字：翻转后背景低于 (255 - otsu_t)
            inv = 255.0 - inner.astype(np.float32)
            ink = inv
            inv_border = 255.0 - border.astype(np.float32)
            bg_floor = max((255.0 - float(otsu_t)) * 0.92, float(np.percentile(inv_border, 85)))
            fg_ceil = float(np.percentile(ink, 99.5))

        if fg_ceil - bg_floor < 30.0:
            return None
        # 严格零底噪：低于 bg_floor 的所有背景像素强制为 0，彻底消除矩形贴图接缝！
        ink_f = np.clip((ink - bg_floor) / max(1.0, fg_ceil - bg_floor), 0.0, 1.0)
        ink_u8 = (ink_f * 255.0).astype(np.uint8)
        ink_u8[ink_u8 < 24] = 0

        ys, xs = np.where(ink_u8 > 32)
        if len(xs) < 25:
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bw, bh = x1 - x0, y1 - y0
        if bw < 10 or bh < 10 or max(bw, bh) / max(1, min(bw, bh)) > 6.0:
            return None
        crop = ink_u8[y0:y1, x0:x1]
        # 拒绝满块反色残图或四边残留黑框
        fill = float((crop > 32).mean())
        if fill < 0.035 or fill > 0.68:
            return None
        cb = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]])
        if float((cb > 40).mean()) > 0.45:
            return None
        max_side = 112
        if max(bw, bh) > max_side:
            sc = max_side / max(bw, bh)
            crop = cv2.resize(crop, (max(4, int(round(bw * sc))), max(4, int(round(bh * sc)))), interpolation=cv2.INTER_AREA)
        return (crop, style, ch)
    except Exception:
        return None


def build_or_load_glyph_pool(total_target=24000, seed=42):
    if POOL_CACHE.exists():
        t0 = time.time()
        data = torch.load(POOL_CACHE, weights_only=False)
        print(f"[Pool] Loaded cached glyph pool: train={len(data['train'])}, val={len(data['val'])} ({time.time()-t0:.2f}s)", flush=True)
        return data

    print(f"[Pool] Building stratified real glyph mask pool (~{total_target} glyphs)...", flush=True)
    t0 = time.time()
    rng = random.Random(seed)
    metas = json.load(open(METAS_PATH))
    by_style = defaultdict(list)
    for path, cls_name, style, _dyn in metas:
        ch = cls_name.split("__")[0] if "__" in cls_name else cls_name
        by_style[style].append((path, style, ch))

    # 按组均衡采样，保证草书/行书/楷隶魏碑/篆刻/甲骨金文全面覆盖
    style_quotas = {
        "caoshu": 3600, "xingshu": 3600, "zhangcao": 1200,
        "kaishu": 2600, "lishu": 2400, "weibei": 1400, "unknown": 1200,
        "zhuanshu": 2000, "xiaozhuan": 1000, "zhuanke": 1200, "jinshi": 1000,
        "jiaguwen": 1400, "jinwen": 1200, "zhanguo": 800,
        "jiandu": 800, "chunqiu": 500, "chuwenzi": 400, "qinwenzi": 400,
    }
    candidates = []
    for st, q in style_quotas.items():
        pool = by_style.get(st, [])
        if not pool:
            continue
        k = min(len(pool), int(q * 1.15))
        candidates.extend(rng.sample(pool, k))
    rng.shuffle(candidates)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for r in ex.map(_load_one_mask, candidates, chunksize=64):
            if r is not None:
                results.append(r)

    # 按字切分 train / val（完全不重叠的字，严格检验跨字泛化）
    chars = sorted({r[2] for r in results})
    rng.shuffle(chars)
    val_chars = set(chars[: max(200, len(chars) // 12)])
    train_items, val_items = [], []
    for r in results:
        if r[2] in val_chars:
            val_items.append(r)
        else:
            train_items.append(r)

    data = {"train": train_items, "val": val_items}
    torch.save(data, POOL_CACHE)
    print(f"[Pool] Saved {POOL_CACHE}: train={len(train_items)}, val={len(val_items)} ({time.time()-t0:.1f}s)", flush=True)
    return data


# =====================================================================
# 2. 六大真实域多字碑帖/书法/拓片/描漆/狂草合成器
# =====================================================================
def make_textured_bg(h, w, base_rgb, rng, noise_amp=14.0, blotch_amp=22.0):
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:, :, 0] = base_rgb[0]
    img[:, :, 1] = base_rgb[1]
    img[:, :, 2] = base_rgb[2]

    # 低频斑驳纸纹/石纹
    s1 = int(rng.integers(6, 18))
    low = rng.normal(0.0, blotch_amp, size=(s1, s1, 1)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)[:, :, None]
    # 中频颗粒
    s2 = int(rng.integers(40, 96))
    mid = rng.normal(0.0, noise_amp, size=(s2, s2, 1)).astype(np.float32)
    mid = cv2.resize(mid, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
    # 方向光影渐变（手机斜拍阴影/光照不均）
    gx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    gy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    ang = rng.uniform(0, 2 * math.pi)
    grad = (gx * math.cos(ang) + gy * math.sin(ang)) * rng.uniform(-24.0, 24.0)

    img += low + mid + grad[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def add_stone_defects(img, rng, white_specks=True):
    h, w = img.shape[:2]
    # 石花/雪花斑点（碑刻拓片常见）
    if white_specks:
        n_specks = rng.randint(40, 260)
        for _ in range(n_specks):
            x = rng.randint(0, w - 1)
            y = rng.randint(0, h - 1)
            r = rng.choice([1, 1, 1, 2])
            v = rng.randint(155, 245)
            cv2.circle(img, (x, y), r, (v, v, v), -1)
    # 细裂纹/折痕
    n_cracks = rng.randint(0, 4)
    for _ in range(n_cracks):
        x0, y0 = rng.randint(0, w - 1), rng.randint(0, h - 1)
        pts = [(x0, y0)]
        ang = rng.uniform(0, 2 * math.pi)
        for _ in range(rng.randint(4, 12)):
            ang += rng.uniform(-0.45, 0.45)
            step = rng.uniform(12, 36)
            x0 = int(np.clip(x0 + math.cos(ang) * step, 0, w - 1))
            y0 = int(np.clip(y0 + math.sin(ang) * step, 0, h - 1))
            pts.append((x0, y0))
        col = rng.randint(140, 220) if white_specks else rng.randint(40, 110)
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (col, col, col), 1, cv2.LINE_AA)
    return img


def blend_glyph(canvas, mask_u8, x0, y0, ink_rgb, rng, alpha_scale=1.0):
    h_c, w_c = canvas.shape[:2]
    mh, mw = mask_u8.shape[:2]
    # 轻微笔画粗细扰动（腐蚀/膨胀模拟石刻风化或洇墨）
    p_morph = rng.random()
    if p_morph < 0.16 and min(mh, mw) >= 16:
        k = np.ones((2, 2), np.uint8)
        mask_u8 = cv2.erode(mask_u8, k, iterations=1)
    elif p_morph < 0.30 and min(mh, mw) >= 16:
        k = np.ones((2, 2), np.uint8)
        mask_u8 = cv2.dilate(mask_u8, k, iterations=1)

    # 裁剪边界
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w_c, x0 + mw), min(h_c, y0 + mh)
    if cx1 <= cx0 + 4 or cy1 <= cy0 + 4:
        return None
    mx0, my0 = cx0 - x0, cy0 - y0
    sub_m = mask_u8[my0 : my0 + (cy1 - cy0), mx0 : mx0 + (cx1 - cx0)]
    alpha = (sub_m.astype(np.float32) / 255.0) * alpha_scale
    # 轻微墨色不均（飞白/风化斑驳）
    if rng.random() < 0.35:
        fade_raw = np.array([[rng.uniform(0.65, 1.0) for _ in range(4)] for _ in range(4)], dtype=np.float32)
        fade = cv2.resize(fade_raw, (cx1 - cx0, cy1 - cy0))
        alpha *= fade
    a3 = alpha[:, :, None]
    roi = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    ink = np.array(ink_rgb, dtype=np.float32)[None, None, :]
    out = roi * (1.0 - a3) + ink * a3
    canvas[cy0:cy1, cx0:cx1] = np.clip(out, 0, 255).astype(np.uint8)

    # 返回实际可见有效墨迹 bbox
    ys, xs = np.where(sub_m > 42)
    if len(xs) < 12:
        return None
    return [float(cx0 + xs.min()), float(cy0 + ys.min()), float(cx0 + xs.max() + 1), float(cy0 + ys.max() + 1)]


def synthesize_page(pool_by_group, all_pool, rng_np):
    """合成一张 512x512 真实感古文字/碑帖多字图，返回 (rgb_u8, boxes_xyxy_float32)"""
    py_rng = random.Random(int(rng_np.integers(0, 2**31 - 1)))

    # 1) 抽选视觉域
    domain = py_rng.choices(
        ["paper_ink", "rubbing", "painted_stele", "cursive_scroll", "ancient_surface", "clean_dict"],
        weights=[0.26, 0.24, 0.16, 0.18, 0.10, 0.06],
        k=1,
    )[0]

    # 根据视觉域选表面背景色、墨色、字形风格组
    if domain == "paper_ink":
        bg_base = (py_rng.randint(212, 245), py_rng.randint(200, 238), py_rng.randint(172, 224))
        ink_rgb = (py_rng.randint(15, 58), py_rng.randint(14, 54), py_rng.randint(12, 50))
        canvas = make_textured_bg(IMG_SIZE, IMG_SIZE, bg_base, rng_np, noise_amp=9, blotch_amp=16)
        style_grp = py_rng.choice(["regular", "cursive", "seal"])
    elif domain == "rubbing":
        v = py_rng.randint(20, 58)
        bg_base = (v, v + py_rng.randint(-4, 4), v + py_rng.randint(-4, 6))
        iv = py_rng.randint(205, 250)
        ink_rgb = (iv, iv - py_rng.randint(0, 8), iv - py_rng.randint(0, 12))
        # 一半为干净哑光名碑拓片（如兰亭序/多宝塔），一半为斑驳石花风化拓片
        if py_rng.random() < 0.55:
            canvas = make_textured_bg(IMG_SIZE, IMG_SIZE, bg_base, rng_np, noise_amp=5, blotch_amp=10)
            if py_rng.random() < 0.35:
                canvas = add_stone_defects(canvas, py_rng, white_specks=True)
        else:
            canvas = make_textured_bg(IMG_SIZE, IMG_SIZE, bg_base, rng_np, noise_amp=15, blotch_amp=22)
            canvas = add_stone_defects(canvas, py_rng, white_specks=True)
        style_grp = py_rng.choice(["regular", "seal", "cursive", "seal", "regular", "ancient"])
    elif domain == "painted_stele":
        # 摩崖石壁 + 朱红/描金/明黄描漆（对齐 01_tiger_cursive.png 真照片域！）
        sv = py_rng.randint(72, 138)
        bg_base = (sv + py_rng.randint(-12, 10), sv + py_rng.randint(-8, 12), sv + py_rng.randint(-18, 8))
        paint_mode = py_rng.choices(["red", "gold", "white_rock", "dark_carve"], weights=[0.50, 0.22, 0.16, 0.12])[0]
        if paint_mode == "red":
            ink_rgb = (py_rng.randint(158, 225), py_rng.randint(36, 95), py_rng.randint(32, 92))
        elif paint_mode == "gold":
            ink_rgb = (py_rng.randint(188, 240), py_rng.randint(145, 205), py_rng.randint(35, 95))
        elif paint_mode == "white_rock":
            ink_rgb = (py_rng.randint(200, 245), py_rng.randint(195, 240), py_rng.randint(185, 232))
        else:
            ink_rgb = (py_rng.randint(18, 52), py_rng.randint(18, 50), py_rng.randint(16, 48))
        canvas = make_textured_bg(IMG_SIZE, IMG_SIZE, bg_base, rng_np, noise_amp=18, blotch_amp=30)
        canvas = add_stone_defects(canvas, py_rng, white_specks=False)
        style_grp = py_rng.choice(["cursive", "regular", "seal"])
    elif domain == "cursive_scroll":
        bg_base = (py_rng.randint(205, 242), py_rng.randint(192, 232), py_rng.randint(165, 212))
        ink_rgb = (py_rng.randint(14, 52), py_rng.randint(14, 50), py_rng.randint(14, 48))
        canvas = make_textured_bg(IMG_SIZE, IMG_SIZE, bg_base, rng_np, noise_amp=8, blotch_amp=14)
        style_grp = "cursive"
    elif domain == "ancient_surface":
        sub = py_rng.choice(["bone", "bronze", "seal_red", "seal_white"])
        if sub == "bone":
            bg_base = (py_rng.randint(185, 225), py_rng.randint(165, 205), py_rng.randint(125, 165))
            ink_rgb = (py_rng.randint(35, 85), py_rng.randint(28, 70), py_rng.randint(20, 58))
            style_grp = "ancient"
        elif sub == "bronze":
            bg_base = (py_rng.randint(52, 92), py_rng.randint(68, 108), py_rng.randint(58, 92))
            ink_rgb = (py_rng.randint(18, 45), py_rng.randint(24, 50), py_rng.randint(20, 44))
            style_grp = "ancient"
        elif sub == "seal_red":
            bg_base = (py_rng.randint(228, 250), py_rng.randint(220, 244), py_rng.randint(206, 234))
            ink_rgb = (py_rng.randint(165, 218), py_rng.randint(28, 65), py_rng.randint(25, 58))
            style_grp = "seal"
        else:
            bg_base = (py_rng.randint(155, 205), py_rng.randint(28, 60), py_rng.randint(24, 54))
            ink_rgb = (py_rng.randint(225, 250), py_rng.randint(215, 242), py_rng.randint(200, 232))
            style_grp = "seal"
        canvas = make_textured_bg(IMG_SIZE, IMG_SIZE, bg_base, rng_np, noise_amp=12, blotch_amp=20)
    else:  # clean_dict
        v = py_rng.randint(232, 255)
        bg_base = (v, v, v - py_rng.randint(0, 8))
        ink_rgb = (py_rng.randint(0, 35), py_rng.randint(0, 35), py_rng.randint(0, 35))
        canvas = make_textured_bg(IMG_SIZE, IMG_SIZE, bg_base, rng_np, noise_amp=3, blotch_amp=5)
        style_grp = py_rng.choice(["regular", "cursive", "seal", "ancient"])

    glyph_source = pool_by_group.get(style_grp) or all_pool
    hard_source = pool_by_group.get("hard_sparse") or glyph_source

    def pick_glyph():
        # 30% 概率强制抽选稀疏/扁平/左右分体难字（如 之/山/和/九/丑/一/二/三/川/林），根除稀疏字与分体字漏检！
        if py_rng.random() < 0.30:
            return py_rng.choice(hard_source)[0]
        return py_rng.choice(glyph_source)[0]

    # 2) 外围拍照留边（含紧密贴边裁图 + 残损锯齿拓片白边/双层截图黑边负样本，根治顶角字漏检与破损白边误检！）
    tight_crop = py_rng.random() < 0.28
    if tight_crop:
        pad_l = py_rng.randint(2, 12)
        pad_r = py_rng.randint(2, 12)
        pad_t = py_rng.randint(2, 12)
        pad_b = py_rng.randint(2, 12)
    else:
        pad_l = py_rng.randint(10, 62) if py_rng.random() < 0.7 else py_rng.randint(4, 22)
        pad_r = py_rng.randint(10, 62) if py_rng.random() < 0.7 else py_rng.randint(4, 22)
        pad_t = py_rng.randint(10, 62) if py_rng.random() < 0.7 else py_rng.randint(4, 22)
        pad_b = py_rng.randint(10, 62) if py_rng.random() < 0.7 else py_rng.randint(4, 22)

    if py_rng.random() < 0.44:
        # 模拟外围拍摄背景：含深色桌面、碑帖白边、以及「残破锯齿宣纸白边 + 外侧深色浏览器截屏条」(对齐真实拓片截图！)
        if domain == "rubbing" and py_rng.random() < 0.72:
            edge_v = py_rng.randint(218, 250)
            outer = np.full_like(canvas, (edge_v, edge_v, edge_v))
            # 50% 概率在拓片四边生成不规则残破锯齿白纸/泛黄托纸毛边，教会网络绝不把锯齿白边当成字！
            if py_rng.random() < 0.58:
                mask_rub = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                # 构造不规则多边形边界（含浅米黄破纸过渡边）
                pts_poly = []
                n_seg = 16
                # top edge
                for k in range(n_seg):
                    px = int(pad_l + (IMG_SIZE - pad_r - pad_l) * k / n_seg)
                    py = int(np.clip(pad_t + py_rng.randint(-5, 7), 0, IMG_SIZE - 1))
                    pts_poly.append([px, py])
                # right edge (with jagged torn notches)
                for k in range(n_seg):
                    px = int(np.clip(IMG_SIZE - pad_r + py_rng.randint(-9, 6), 0, IMG_SIZE - 1))
                    py = int(pad_t + (IMG_SIZE - pad_b - pad_t) * k / n_seg)
                    pts_poly.append([px, py])
                # bottom edge
                for k in range(n_seg):
                    px = int(IMG_SIZE - pad_r - (IMG_SIZE - pad_r - pad_l) * k / n_seg)
                    py = int(np.clip(IMG_SIZE - pad_b + py_rng.randint(-6, 6), 0, IMG_SIZE - 1))
                    pts_poly.append([px, py])
                # left edge
                for k in range(n_seg):
                    px = int(np.clip(pad_l + py_rng.randint(-6, 8), 0, IMG_SIZE - 1))
                    py = int(IMG_SIZE - pad_b - (IMG_SIZE - pad_b - pad_t) * k / n_seg)
                    pts_poly.append([px, py])
                poly = np.array([pts_poly], dtype=np.int32)
                cv2.fillPoly(mask_rub, poly, 255)
                # 外圈 2~5px 泛黄旧托纸毛边
                dil = cv2.dilate(mask_rub, np.ones((py_rng.choice([3, 5, 7]), py_rng.choice([3, 5, 7])), np.uint8))
                fringe = (dil > 0) & (mask_rub == 0)
                outer[fringe] = (py_rng.randint(195, 226), py_rng.randint(182, 212), py_rng.randint(150, 184))
                outer[mask_rub > 0] = canvas[mask_rub > 0]
                # 45% 概率在最外侧再叠加暗黑截屏留边（如手机/暗色阅读器截图黑条）
                if py_rng.random() < 0.45:
                    bar_w = py_rng.randint(14, max(16, min(72, max(pad_r, pad_l))))
                    dark_v = (py_rng.randint(14, 34), py_rng.randint(18, 38), py_rng.randint(22, 44))
                    if py_rng.random() < 0.65:
                        outer[:, IMG_SIZE - bar_w :] = dark_v
                    else:
                        outer[:, :bar_w] = dark_v
            else:
                outer[pad_t : IMG_SIZE - pad_b, pad_l : IMG_SIZE - pad_r] = canvas[pad_t : IMG_SIZE - pad_b, pad_l : IMG_SIZE - pad_r]
            canvas = outer
        else:
            edge_v = py_rng.randint(18, 70)
            outer = np.full_like(canvas, (edge_v, edge_v, edge_v))
            outer[pad_t : IMG_SIZE - pad_b, pad_l : IMG_SIZE - pad_r] = canvas[pad_t : IMG_SIZE - pad_b, pad_l : IMG_SIZE - pad_r]
            canvas = outer

    area_w = max(120, IMG_SIZE - pad_l - pad_r)
    area_h = max(120, IMG_SIZE - pad_t - pad_b)

    # 3) 排版布局选择（单字/单行/单列/多列整碑/散排）
    if domain == "cursive_scroll":
        layout = py_rng.choices(["multi_col", "single_col", "single_row"], weights=[0.58, 0.30, 0.12])[0]
    else:
        layout = py_rng.choices(
            ["multi_col", "single_col", "single_row", "single_char", "scattered"],
            weights=[0.44, 0.20, 0.18, 0.08, 0.10],
            k=1,
        )[0]

    boxes = []
    if layout == "single_char":
        mask = pick_glyph()
        target = py_rng.randint(130, min(area_w, area_h) - 16)
        sc = target / max(mask.shape)
        nw, nh = max(24, int(mask.shape[1] * sc)), max(24, int(mask.shape[0] * sc))
        rm = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
        x0 = pad_l + py_rng.randint(0, max(0, area_w - nw))
        y0 = pad_t + py_rng.randint(0, max(0, area_h - nh))
        b = blend_glyph(canvas, rm, x0, y0, ink_rgb, py_rng, alpha_scale=py_rng.uniform(0.82, 1.0))
        if b:
            boxes.append(b)
    elif layout == "scattered":
        n_chars = py_rng.randint(3, 14)
        base_sz = py_rng.randint(44, 110)
        for _ in range(n_chars * 3):
            if len(boxes) >= n_chars:
                break
            mask = pick_glyph()
            sz = int(base_sz * py_rng.uniform(0.7, 1.3))
            sc = sz / max(mask.shape)
            nw, nh = max(18, int(mask.shape[1] * sc)), max(18, int(mask.shape[0] * sc))
            if nw >= area_w - 8 or nh >= area_h - 8:
                continue
            x0 = pad_l + py_rng.randint(0, area_w - nw)
            y0 = pad_t + py_rng.randint(0, area_h - nh)
            # 防止严重重叠
            cand = [x0, y0, x0 + nw, y0 + nh]
            if any(box_iou_xyxy(cand, ob) > 0.12 for ob in boxes):
                continue
            rm = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
            b = blend_glyph(canvas, rm, x0, y0, ink_rgb, py_rng, alpha_scale=py_rng.uniform(0.78, 1.0))
            if b:
                boxes.append(b)
    else:
        if layout == "single_row":
            ncols = py_rng.randint(2, 8)
            nrows = 1
        elif layout == "single_col":
            ncols = 1
            nrows = py_rng.randint(2, 8)
        else:
            ncols = py_rng.randint(2, 7)
            nrows = py_rng.randint(2, 7)
            while ncols * nrows > 36:
                if ncols >= nrows:
                    ncols -= 1
                else:
                    nrows -= 1

        cell_w = area_w / ncols
        cell_h = area_h / nrows
        base_cell = min(cell_w, cell_h)
        # 防止单行/单列时格子过扁，把实际区域居中收拢成真实牌匾/对联比例
        if layout == "single_row":
            actual_h = min(area_h, int(cell_w * py_rng.uniform(0.95, 1.45)))
            pad_t = pad_t + py_rng.randint(0, max(0, area_h - actual_h))
            cell_h = float(actual_h)
            base_cell = min(cell_w, cell_h)
        elif layout == "single_col":
            actual_w = min(area_w, int(cell_h * py_rng.uniform(0.95, 1.45)))
            pad_l = pad_l + py_rng.randint(0, max(0, area_w - actual_w))
            cell_w = float(actual_w)
            base_cell = min(cell_w, cell_h)

        # 可选界格线（乌丝栏 / 朱丝栏 / 碑刻方格）
        draw_grid = domain in ("paper_ink", "rubbing", "clean_dict") and py_rng.random() < 0.34
        if draw_grid:
            if domain == "paper_ink" and py_rng.random() < 0.5:
                grid_col = (py_rng.randint(150, 200), py_rng.randint(60, 95), py_rng.randint(55, 90))  # 朱丝栏
            elif domain == "rubbing":
                grid_col = (py_rng.randint(135, 195), py_rng.randint(135, 195), py_rng.randint(135, 195))
            else:
                grid_col = (py_rng.randint(75, 145), py_rng.randint(75, 145), py_rng.randint(75, 145))
            gx0, gy0 = int(pad_l), int(pad_t)
            gx1, gy1 = int(pad_l + ncols * cell_w), int(pad_t + nrows * cell_h)
            cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), grid_col, 1)
            for c in range(1, ncols):
                x = int(pad_l + c * cell_w)
                cv2.line(canvas, (x, gy0), (x, gy1), grid_col, 1)
            if py_rng.random() < 0.55:
                for r in range(1, nrows):
                    y = int(pad_t + r * cell_h)
                    cv2.line(canvas, (gx0, y), (gx1, y), grid_col, 1)

        is_cursive = (domain == "cursive_scroll") or (style_grp == "cursive" and py_rng.random() < 0.5)
        for c in range(ncols):
            prev_bottom = None
            col_sway = py_rng.uniform(-0.06, 0.06) * cell_w
            for r in range(nrows):
                # 偶发缺字/落款空位
                if ncols * nrows >= 6 and py_rng.random() < 0.08:
                    prev_bottom = None
                    continue
                mask = pick_glyph()
                if is_cursive:
                    fill_ratio = py_rng.uniform(0.68, 1.06)
                else:
                    fill_ratio = py_rng.uniform(0.62, 0.90)
                target_sz = max(18.0, base_cell * fill_ratio)
                sc = target_sz / max(mask.shape)
                nw = max(14, min(int(cell_w * 1.15), int(round(mask.shape[1] * sc))))
                nh = max(14, min(int(cell_h * 1.18), int(round(mask.shape[0] * sc))))
                rm = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)

                cx = pad_l + (c + 0.5) * cell_w + col_sway + py_rng.uniform(-0.10, 0.10) * cell_w
                cy = pad_t + (r + 0.5) * cell_h + py_rng.uniform(-0.10, 0.10) * cell_h
                x0 = int(round(cx - nw * 0.5))
                y0 = int(round(cy - nh * 0.5))

                # 狂草牵丝连笔：在上一字底部到本字顶部之间画 1~2px 细墨线，训练网络学会切开连笔！
                if is_cursive and prev_bottom is not None and py_rng.random() < 0.42:
                    px, py_ = prev_bottom
                    tx = int(np.clip(cx + py_rng.uniform(-nw * 0.25, nw * 0.25), 0, IMG_SIZE - 1))
                    ty = int(np.clip(y0 + py_rng.uniform(0, nh * 0.2), 0, IMG_SIZE - 1))
                    cv2.line(canvas, (int(px), int(py_)), (tx, ty), ink_rgb, py_rng.choice([1, 2]), cv2.LINE_AA)

                b = blend_glyph(canvas, rm, x0, y0, ink_rgb, py_rng, alpha_scale=py_rng.uniform(0.80, 1.0))
                if b is not None:
                    boxes.append(b)
                    prev_bottom = ((b[0] + b[2]) * 0.5, b[3])
                else:
                    prev_bottom = None

    # 4) 边缘红色闲章干扰（仅在黑墨纸本上加小闲章作为负样本，不计入 boxes）
    if domain in ("paper_ink", "cursive_scroll") and len(boxes) >= 3 and py_rng.random() < 0.20:
        sx = py_rng.choice([py_rng.randint(6, 26), py_rng.randint(IMG_SIZE - 38, IMG_SIZE - 18)])
        sy = py_rng.randint(12, IMG_SIZE - 36)
        sw = py_rng.randint(14, 24)
        cand_seal = [sx, sy, sx + sw, sy + sw]
        if all(box_iou_xyxy(cand_seal, b) == 0.0 for b in boxes):
            scol = (py_rng.randint(170, 215), py_rng.randint(35, 65), py_rng.randint(32, 60))
            cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sw), scol, -1)
            cv2.rectangle(canvas, (sx + 2, sy + 2), (sx + sw - 2, sy + sw - 2), (240, 232, 220), 1)

    # 5) 手机斜拍轻度透视变换 (Homography) — 同步精确变换每个字的四顶点！
    if len(boxes) > 0 and py_rng.random() < 0.45:
        d = IMG_SIZE * 0.055
        src_pts = np.float32([[0, 0], [IMG_SIZE - 1, 0], [IMG_SIZE - 1, IMG_SIZE - 1], [0, IMG_SIZE - 1]])
        dst_pts = np.float32([
            [py_rng.uniform(0, d), py_rng.uniform(0, d)],
            [IMG_SIZE - 1 - py_rng.uniform(0, d), py_rng.uniform(0, d)],
            [IMG_SIZE - 1 - py_rng.uniform(0, d), IMG_SIZE - 1 - py_rng.uniform(0, d)],
            [py_rng.uniform(0, d), IMG_SIZE - 1 - py_rng.uniform(0, d)],
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        bval = tuple(int(c) for c in canvas[0, 0])
        canvas = cv2.warpPerspective(canvas, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_CONSTANT, borderValue=bval)
        new_boxes = []
        for x0, y0, x1, y1 in boxes:
            corners = np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]])
            warped = cv2.perspectiveTransform(corners, M)[0]
            nx0 = float(np.clip(warped[:, 0].min(), 0, IMG_SIZE - 1))
            ny0 = float(np.clip(warped[:, 1].min(), 0, IMG_SIZE - 1))
            nx1 = float(np.clip(warped[:, 0].max(), 0, IMG_SIZE - 1))
            ny1 = float(np.clip(warped[:, 1].max(), 0, IMG_SIZE - 1))
            if nx1 - nx0 >= 10 and ny1 - ny0 >= 10:
                new_boxes.append([nx0, ny0, nx1, ny1])
        boxes = new_boxes

    # 6) 相机光学退化（模糊、色偏、JPEG 压缩）
    if py_rng.random() < 0.30:
        k = py_rng.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (k, k), py_rng.uniform(0.4, 1.1))
    if py_rng.random() < 0.45:
        q = py_rng.randint(48, 92)
        ok, enc = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            canvas = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 7) 65% 概率模拟真实非正方形竖幅/横幅碑帖经 512x512 灰边 Letterbox（含 0..32px 四周留边，严格对齐线上推理！）
    if len(boxes) > 0 and py_rng.random() < 0.65:
        margin = py_rng.choice([0, 0, 16, 24, 32])
        inner_sz = IMG_SIZE - 2 * margin
        if py_rng.random() < 0.65:
            # 竖幅碑帖（如兰亭序/峄山碑局部 0.50..0.92）
            rw = int(round(inner_sz * py_rng.uniform(0.50, 0.92)))
            rh = inner_sz
        else:
            # 横幅牌匾（0.45..0.92）
            rw = inner_sz
            rh = int(round(inner_sz * py_rng.uniform(0.45, 0.92)))
        resized = cv2.resize(canvas, (rw, rh), interpolation=cv2.INTER_AREA)
        lb = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        ox = (IMG_SIZE - rw) // 2
        oy = (IMG_SIZE - rh) // 2
        lb[oy : oy + rh, ox : ox + rw] = resized
        sx, sy = rw / float(IMG_SIZE), rh / float(IMG_SIZE)
        canvas = lb
        boxes = [
            [x0 * sx + ox, y0 * sy + oy, x1 * sx + ox, y1 * sy + oy]
            for (x0, y0, x1, y1) in boxes
            if (x1 - x0) * sx >= 8 and (y1 - y0) * sy >= 8
        ]

    boxes_arr = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    return canvas, boxes_arr


def box_iou_xyxy(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-6, ua)


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2
    return max(2, int(min(r1, r2, r3)))


def draw_gaussian(heatmap, cx, cy, radius):
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    ys = xs[:, None]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if right <= 0 or bottom <= 0 or left < 0 or top < 0:
        return
    masked_hm = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    masked_g = g[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(masked_hm, masked_g, out=masked_hm)


def encode_targets(boxes_xyxy):
    hm = np.zeros((1, FM_SIZE, FM_SIZE), dtype=np.float32)
    wh_reg = np.zeros((4, FM_SIZE, FM_SIZE), dtype=np.float32)
    pos_mask = np.zeros((1, FM_SIZE, FM_SIZE), dtype=np.float32)
    for x0, y0, x1, y1 in boxes_xyxy:
        bw = (x1 - x0) / STRIDE
        bh = (y1 - y0) / STRIDE
        if bw < 1.5 or bh < 1.5:
            continue
        cx_f = ((x0 + x1) * 0.5) / STRIDE
        cy_f = ((y0 + y1) * 0.5) / STRIDE
        cx = int(np.clip(math.floor(cx_f), 0, FM_SIZE - 1))
        cy = int(np.clip(math.floor(cy_f), 0, FM_SIZE - 1))
        rad = gaussian_radius((bh, bw), min_overlap=0.7)
        draw_gaussian(hm[0], cx, cy, rad)
        wh_reg[0, cy, cx] = bw
        wh_reg[1, cy, cx] = bh
        wh_reg[2, cy, cx] = cx_f - cx
        wh_reg[3, cy, cx] = cy_f - cy
        pos_mask[0, cy, cx] = 1.0
    return hm, wh_reg, pos_mask


class SynthCharDetDataset(Dataset):
    def __init__(self, glyph_list, length, fixed_seed=None):
        self.all_pool = glyph_list
        self.pool_by_group = defaultdict(list)
        for item in glyph_list:
            m, st, _ch = item
            for gname, styles in STYLE_GROUPS.items():
                if st in styles:
                    self.pool_by_group[gname].append(item)
            # 筛选稀疏/扁平/中心空隙大的难字（如 之/山/和/九/丑/川/林/一/二/三）用于 3x 过采样
            mh, mw = m.shape[:2]
            fill = float((m > 40).mean())
            ar = mw / max(1.0, float(mh))
            # 检查中轴垂直空隙（左右分体字如 和/明/川/林）
            mid_x0, mid_x1 = max(0, int(mw * 0.42)), min(mw, int(mw * 0.58) + 1)
            mid_fill = float((m[:, mid_x0:mid_x1] > 40).mean()) if mid_x1 > mid_x0 else fill
            if fill < 0.20 or ar > 1.28 or mid_fill < 0.08:
                self.pool_by_group["hard_sparse"].append(item)
        self.length = length
        self.fixed_seed = fixed_seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.fixed_seed is not None:
            rng = np.random.default_rng(self.fixed_seed + idx * 1009)
        else:
            rng = np.random.default_rng()
        rgb, boxes = synthesize_page(self.pool_by_group, self.all_pool, rng)
        hm, wh_reg, pos_mask = encode_targets(boxes)
        x = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))
        if self.fixed_seed is not None:
            return torch.from_numpy(x), torch.from_numpy(hm), torch.from_numpy(wh_reg), torch.from_numpy(pos_mask), boxes
        return torch.from_numpy(x), torch.from_numpy(hm), torch.from_numpy(wh_reg), torch.from_numpy(pos_mask)


# =====================================================================
# 3. 轻量神经单字检测模型: MobileNetV3-Large + DW-Sep FPN + Stride-4 Head
# =====================================================================
class DWSepConv(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, k, padding=k // 2, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class GlyphCenterNet(nn.Module):
    def __init__(self, fpn_dim=96, pretrained=True):
        super().__init__()
        weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        feats = tv_models.mobilenet_v3_large(weights=weights).features
        # Stage splits for 512x512 input:
        # s2: [:4]   -> stride 4,  24ch  (128x128)
        # s3: [4:7]  -> stride 8,  40ch  (64x64)
        # s4: [7:13] -> stride 16, 112ch (32x32)
        # s5: [13:]  -> stride 32, 960ch (16x16)
        self.s2 = feats[:4]
        self.s3 = feats[4:7]
        self.s4 = feats[7:13]
        self.s5 = feats[13:]

        self.lat5 = nn.Sequential(nn.Conv2d(960, fpn_dim, 1, bias=False), nn.BatchNorm2d(fpn_dim), nn.Hardswish(inplace=True))
        self.lat4 = nn.Sequential(nn.Conv2d(112, fpn_dim, 1, bias=False), nn.BatchNorm2d(fpn_dim), nn.Hardswish(inplace=True))
        self.lat3 = nn.Sequential(nn.Conv2d(40, fpn_dim, 1, bias=False), nn.BatchNorm2d(fpn_dim), nn.Hardswish(inplace=True))
        self.lat2 = nn.Sequential(nn.Conv2d(24, fpn_dim, 1, bias=False), nn.BatchNorm2d(fpn_dim), nn.Hardswish(inplace=True))

        self.smooth4 = DWSepConv(fpn_dim, fpn_dim)
        self.smooth3 = DWSepConv(fpn_dim, fpn_dim)
        self.smooth2 = DWSepConv(fpn_dim, fpn_dim)

        self.hm_head = nn.Sequential(DWSepConv(fpn_dim, fpn_dim), nn.Conv2d(fpn_dim, 1, 1))
        self.wh_head = nn.Sequential(DWSepConv(fpn_dim, fpn_dim), nn.Conv2d(fpn_dim, 2, 1))
        self.reg_head = nn.Sequential(DWSepConv(fpn_dim, 64), nn.Conv2d(64, 2, 1))

        # CenterNet 经典先验偏置初始化，防止初始热力图全正导致梯度爆炸
        nn.init.constant_(self.hm_head[-1].bias, -2.19)

    def forward(self, x):
        c2 = self.s2(x)
        c3 = self.s3(c2)
        c4 = self.s4(c3)
        c5 = self.s5(c4)

        p5 = self.lat5(c5)
        p4 = self.smooth4(self.lat4(c4) + F.interpolate(p5, scale_factor=2.0, mode="bilinear", align_corners=False))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, scale_factor=2.0, mode="bilinear", align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, scale_factor=2.0, mode="bilinear", align_corners=False))

        hm = torch.sigmoid(self.hm_head(p2))
        wh = F.softplus(self.wh_head(p2))
        reg = torch.sigmoid(self.reg_head(p2))
        wh_reg = torch.cat([wh, reg], dim=1)
        return hm, wh_reg


def centernet_loss(hm_pred, wh_reg_pred, hm_gt, wh_reg_gt, pos_mask):
    # 1) Penalty-reduced Focal Loss
    pred = hm_pred.clamp(1e-4, 1.0 - 1e-4)
    pos_inds = hm_gt.eq(1.0).float()
    neg_weights = torch.pow(1.0 - hm_gt, 4.0)
    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, 2.0) * pos_inds
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, 2.0) * neg_weights * (1.0 - pos_inds)
    num_pos = pos_inds.sum().clamp(min=1.0)
    hm_loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos

    # 2) Masked L1 on WH + Reg + GIoU on positive centers
    m = pos_mask.expand_as(wh_reg_pred)
    num_m = pos_mask.sum().clamp(min=1.0)
    wh_l1 = (F.smooth_l1_loss(wh_reg_pred[:, :2], wh_reg_gt[:, :2], reduction="none", beta=1.0) * pos_mask).sum() / num_m
    reg_l1 = (F.l1_loss(wh_reg_pred[:, 2:], wh_reg_gt[:, 2:], reduction="none") * pos_mask).sum() / num_m

    # GIoU on positive positions
    b_idx, _, ys, xs = torch.where(pos_mask > 0.5)
    if len(b_idx) > 0:
        px = xs.float() + wh_reg_pred[b_idx, 2, ys, xs]
        py = ys.float() + wh_reg_pred[b_idx, 3, ys, xs]
        pw = wh_reg_pred[b_idx, 0, ys, xs].clamp(min=0.5)
        ph = wh_reg_pred[b_idx, 1, ys, xs].clamp(min=0.5)

        gx = xs.float() + wh_reg_gt[b_idx, 2, ys, xs]
        gy = ys.float() + wh_reg_gt[b_idx, 3, ys, xs]
        gw = wh_reg_gt[b_idx, 0, ys, xs].clamp(min=0.5)
        gh = wh_reg_gt[b_idx, 1, ys, xs].clamp(min=0.5)

        px0, py0, px1, py1 = px - pw * 0.5, py - ph * 0.5, px + pw * 0.5, py + ph * 0.5
        gx0, gy0, gx1, gy1 = gx - gw * 0.5, gy - gh * 0.5, gx + gw * 0.5, gy + gh * 0.5

        ix0, iy0 = torch.maximum(px0, gx0), torch.maximum(py0, gy0)
        ix1, iy1 = torch.minimum(px1, gx1), torch.minimum(py1, gy1)
        inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
        union = pw * ph + gw * gh - inter + 1e-6
        iou = inter / union
        cx0, cy0 = torch.minimum(px0, gx0), torch.minimum(py0, gy0)
        cx1, cy1 = torch.maximum(px1, gx1), torch.maximum(py1, gy1)
        c_area = ((cx1 - cx0) * (cy1 - cy0)).clamp(min=1e-6)
        giou = iou - (c_area - union) / c_area
        giou_loss = (1.0 - giou).mean()
    else:
        giou_loss = hm_loss * 0.0

    total = hm_loss + 0.12 * wh_l1 + 1.0 * reg_l1 + 1.5 * giou_loss
    return total, hm_loss.item(), wh_l1.item(), giou_loss.item()


def decode_predictions(hm_np, wh_reg_np, score_thresh=0.35, nms_iou=0.45, max_dets=48):
    """hm_np: [128, 128], wh_reg_np: [4, 128, 128] -> list of (x0, y0, x1, y1, score)"""
    # 3x3 local max peak filter
    h_t = torch.from_numpy(hm_np)[None, None]
    hmax = F.max_pool2d(h_t, kernel_size=3, stride=1, padding=1)[0, 0].numpy()
    keep = (hm_np == hmax) & (hm_np >= score_thresh)
    ys, xs = np.where(keep)
    if len(xs) == 0:
        return []
    scores = hm_np[ys, xs]
    order = np.argsort(-scores)[: max_dets * 2]
    cands = []
    for i in order:
        x, y, sc = int(xs[i]), int(ys[i]), float(scores[i])
        bw = float(wh_reg_np[0, y, x]) * STRIDE
        bh = float(wh_reg_np[1, y, x]) * STRIDE
        dx = float(wh_reg_np[2, y, x])
        dy = float(wh_reg_np[3, y, x])
        cx = (x + dx) * STRIDE
        cy = (y + dy) * STRIDE
        x0 = max(0.0, cx - bw * 0.5)
        y0 = max(0.0, cy - bh * 0.5)
        x1 = min(float(IMG_SIZE), cx + bw * 0.5)
        y1 = min(float(IMG_SIZE), cy + bh * 0.5)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        cands.append((x0, y0, x1, y1, sc))

    # Greedy IoU NMS
    kept = []
    for c in cands:
        if all(box_iou_xyxy(c, k) < nms_iou for k in kept):
            kept.append(c)
            if len(kept) >= max_dets:
                break
    return kept


@torch.no_grad()
def evaluate_detector(model, val_ds, device, score_thresh=0.35):
    model.eval()
    tp, fp, fn = 0, 0, 0
    ious = []
    for idx in range(len(val_ds)):
        x, _hm, _wh, _m, gt_boxes = val_ds[idx]
        hm_p, wh_p = model(x[None].to(device))
        preds = decode_predictions(hm_p[0, 0].cpu().numpy(), wh_p[0].cpu().numpy(), score_thresh=score_thresh)
        used = set()
        for p in preds:
            best_i, best_iou = -1, 0.0
            for gi, g in enumerate(gt_boxes):
                if gi in used:
                    continue
                v = box_iou_xyxy(p, g)
                if v > best_iou:
                    best_iou, best_i = v, gi
            if best_iou >= 0.5 and best_i >= 0:
                tp += 1
                used.add(best_i)
                ious.append(best_iou)
            else:
                fp += 1
        fn += len(gt_boxes) - len(used)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    miou = float(np.mean(ious)) if ious else 0.0
    return {"prec": prec, "rec": rec, "f1": f1, "miou": miou, "tp": tp, "fp": fp, "fn": fn}


def export_onnx_u8(model, out_fp32_path, out_u8_path):
    model.eval().cpu()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(out_fp32_path),
        input_names=["image"],
        output_names=["hm", "wh_reg"],
        opset_version=17,
        do_constant_folding=True,
    )
    # Weight-Only Per-Channel U8 quantization (proven zero-drift recipe from ResNet50)
    m = onnx.load(str(out_fp32_path))
    del m.graph.value_info[:]
    init_map = {t.name: t for t in m.graph.initializer}
    new_inits, deq_nodes, replaced = [], [], set()
    for node in m.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 2 and node.input[1] in init_map:
            wname = node.input[1]
            if wname in replaced:
                node.input[1] = wname + "_deq"
                continue
            arr = numpy_helper.to_array(init_map[wname])
            if arr.size < 4096 or arr.ndim != 4:
                continue
            reduce_axes = (1, 2, 3)
            wmin = arr.min(axis=reduce_axes)
            wmax = arr.max(axis=reduce_axes)
            scales = np.maximum((wmax - wmin) / 255.0, 1e-8).astype(np.float32)
            zps = np.clip(np.round(-wmin / scales), 0, 255).astype(np.uint8)
            q = np.clip(np.round(arr / scales.reshape(-1, 1, 1, 1)) + zps.reshape(-1, 1, 1, 1), 0, 255).astype(np.uint8)
            qn, sn, zn, dn = wname + "_q8", wname + "_s8", wname + "_z8", wname + "_deq"
            new_inits.extend([
                numpy_helper.from_array(q, name=qn),
                numpy_helper.from_array(scales, name=sn),
                numpy_helper.from_array(zps, name=zn),
            ])
            deq_nodes.append(helper.make_node("DequantizeLinear", inputs=[qn, sn, zn], outputs=[dn], axis=0))
            node.input[1] = dn
            replaced.add(wname)

    kept_inits = [t for t in m.graph.initializer if t.name not in replaced]
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept_inits)
    m.graph.initializer.extend(new_inits)
    old_nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(deq_nodes)
    m.graph.node.extend(old_nodes)
    onnx.save(m, str(out_u8_path))

    raw_bytes = Path(out_u8_path).read_bytes()
    gz_bytes = gzip.compress(raw_bytes, compresslevel=6)
    print(f"[ONNX] FP32={Path(out_fp32_path).stat().st_size/1e6:.2f}MB -> WO-U8={len(raw_bytes)/1e6:.2f}MB, gz={len(gz_bytes)/1e6:.2f}MB (quantized {len(replaced)} convs)", flush=True)

    # Verify ONNX U8 vs PyTorch parity
    sess = ort.InferenceSession(str(out_u8_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_hm, pt_wh = model(dummy)
    ort_hm, ort_wh = sess.run(None, {"image": dummy.numpy()})
    hm_diff = float(np.abs(pt_hm.numpy() - ort_hm).max())
    wh_diff = float(np.abs(pt_wh.numpy() - ort_wh).max())
    print(f"[ONNX] Parity check: max|hm_diff|={hm_diff:.5f}, max|wh_diff|={wh_diff:.5f}", flush=True)
    return raw_bytes, gz_bytes


def render_visual_audit(onnx_path, val_ds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # 1) 6 diverse held-out synthetic pages
    for idx in range(6):
        x, _, _, _, gt = val_ds[idx]
        rgb = np.clip((np.transpose(x.numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8).copy()
        hm, wh = sess.run(None, {"image": x[None].numpy()})
        preds = decode_predictions(hm[0, 0], wh[0], score_thresh=0.35)
        for gx0, gy0, gx1, gy1 in gt:
            cv2.rectangle(rgb, (int(gx0), int(gy0)), (int(gx1), int(gy1)), (0, 200, 0), 1)
        for i, (x0, y0, x1, y1, sc) in enumerate(preds):
            cv2.rectangle(rgb, (int(x0), int(y0)), (int(x1), int(y1)), (220, 40, 30), 2)
            cv2.putText(rgb, f"{i+1}:{sc:.2f}", (int(x0), max(12, int(y0) - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 40, 30), 1)
        Image.fromarray(rgb).save(out_dir / f"val_{idx:02d}_gt{len(gt)}_pred{len(preds)}.png")

    # 2) Real photos in eval_real/
    for rp in sorted((ROOT / "eval_real").glob("*.png")):
        im = Image.open(rp).convert("RGB")
        w, h = im.size
        sc = IMG_SIZE / max(w, h)
        nw, nh = int(round(w * sc)), int(round(h * sc))
        res = im.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        ox, oy = (IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2
        canvas[oy : oy + nh, ox : ox + nw] = np.asarray(res)
        x = np.transpose((canvas.astype(np.float32) / 255.0 - 0.5) / 0.5, (2, 0, 1))[None]
        hm, wh = sess.run(None, {"image": x})
        preds = decode_predictions(hm[0, 0], wh[0], score_thresh=0.30)
        vis = canvas.copy()
        for i, (x0, y0, x1, y1, s) in enumerate(preds):
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (220, 40, 30), 2)
            cv2.putText(vis, f"{i+1}:{s:.2f}", (int(x0), max(14, int(y0) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 40, 30), 1)
        Image.fromarray(vis).save(out_dir / f"real_{rp.stem}_pred{len(preds)}.png")
    print(f"[Visual] Saved audit visualizations to {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--samples-per-epoch", type=int, default=2400)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = build_or_load_glyph_pool()
    train_ds = SynthCharDetDataset(pool["train"], length=args.samples_per_epoch, fixed_seed=None)
    val_ds = SynthCharDetDataset(pool["val"], length=160, fixed_seed=20261120)
    print(f"[Pool] hard_sparse={len(train_ds.pool_by_group['hard_sparse'])}/{len(train_ds.all_pool)}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = GlyphCenterNet(fpn_dim=96, pretrained=not bool(args.resume)).to(device)
    if args.resume and Path(args.resume).exists():
        ck_in = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ck_in["model"])
        print(f"[Resume] Loaded weights from {args.resume} (prev_best={ck_in.get('metrics', {}).get('f1', 0):.4f})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] GlyphCenterNet (MobileNetV3-Large + 96ch FPN) params={n_params/1e6:.2f}M on {device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=5e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_f1 = -1.0
    best_ckpt = WORK_DET / "best_det.pt"
    t_start = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_hm, ep_wh, ep_giou, n_steps = 0.0, 0.0, 0.0, 0.0, 0
        t0 = time.time()
        for x, hm_gt, wh_gt, pos_m in train_loader:
            x = x.to(device, non_blocking=True)
            hm_gt = hm_gt.to(device, non_blocking=True)
            wh_gt = wh_gt.to(device, non_blocking=True)
            pos_m = pos_m.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                hm_p, wh_p = model(x)
                loss, l_hm, l_wh, l_giou = centernet_loss(hm_p, wh_p, hm_gt, wh_gt, pos_m)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()
            ep_hm += l_hm
            ep_wh += l_wh
            ep_giou += l_giou
            n_steps += 1
        sched.step()
        dt = time.time() - t0
        log = f"[Ep {ep:02d}/{args.epochs}] loss={ep_loss/n_steps:.4f} (hm={ep_hm/n_steps:.3f}, wh={ep_wh/n_steps:.3f}, giou={ep_giou/n_steps:.3f}) lr={opt.param_groups[0]['lr']:.2e} {dt:.1f}s"

        if ep % 4 == 0 or ep == args.epochs:
            metrics = evaluate_detector(model, val_ds, device, score_thresh=0.35)
            log += f" | VAL P={metrics['prec']:.4f} R={metrics['rec']:.4f} F1={metrics['f1']:.4f} mIoU={metrics['miou']:.4f}"
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save({"model": model.state_dict(), "metrics": metrics, "epoch": ep}, best_ckpt)
                log += " *BEST*"
        print(log, flush=True)

    print(f"[Train] Done in {(time.time()-t_start)/60:.1f} min, best_F1={best_f1:.4f}", flush=True)
    ck = torch.load(best_ckpt, weights_only=False)
    model.load_state_dict(ck["model"])
    fp32_p = WORK_DET / "char_det_fp32.onnx"
    u8_p = WORK_DET / "char_det_u8.onnx"
    export_onnx_u8(model, fp32_p, u8_p)
    render_visual_audit(u8_p, val_ds, WORK_DET / "audit_vis")


if __name__ == "__main__":
    main()
