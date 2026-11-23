#!/usr/bin/env python3
"""
Portable, relative-path dataset loaders for Hanfire Shizi (Clean V10).

Reads modular datasets downloaded via `python scripts/download_dataset.py`:
  - `data/clean_base/`     : 1,235,277 deduplicated multi-style base glyphs (`clean_base_meta.jsonl.gz`)
  - `data/real_artifacts/` : museum calligraphy W1-W6, OBIMD oracle bones, CHUBS Chu bamboo slips,
                             Western Zhou bronzes + Song Stone Drums + Mawangdui, Han Dynasty seals!
"""
from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from training.model import is_valid_cjk, load_glyph, load_rubbing_autoinv, simp


def load_clean_base(clean_base_dir: Path, smoke: bool = False):
    """Load clean_base_meta.jsonl.gz into deterministic train/val splits (85/15 per class)."""
    meta_gz = clean_base_dir / "clean_base_meta.jsonl.gz"
    csd_p = clean_base_dir / "classes_styles_dynasties.json"
    if not meta_gz.exists():
        raise FileNotFoundError(f"未找到 {meta_gz}，请先运行: python scripts/download_dataset.py --base")

    csd = json.loads(csd_p.read_text(encoding="utf-8"))
    classes, styles, dynasties = csd["classes"], csd["styles"], csd["dynasties"]
    cls2id = {c: i for i, c in enumerate(classes)}
    sty2id = {s: i for i, s in enumerate(styles)}
    dyn2id = {d: i for i, d in enumerate(dynasties)}

    by_cls = defaultdict(list)
    with gzip.open(meta_gz, "rt", encoding="utf-8") as f:
        for line in f:
            rel, cls_name, st, dyn = json.loads(line)
            if cls_name not in cls2id:
                continue
            full_p = str(clean_base_dir / rel)
            by_cls[cls_name].append((full_p, cls2id[cls_name], sty2id.get(st, 0), dyn2id.get(dyn, 0)))

    rng = random.Random(42)
    keys = sorted(by_cls.keys())
    if smoke:
        keys = random.Random(0).sample(keys, min(400, len(keys)))

    train_items, val_items = [], []
    for k in keys:
        lst = list(by_cls[k])
        if len(lst) < 2:
            continue
        rng.shuffle(lst)
        n_val = max(1, int(len(lst) * 0.15))
        val_items.extend(lst[:n_val])
        train_items.extend(lst[n_val:])
    return classes, styles, dynasties, train_items, val_items


class GlyphClassifyDataset(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, s, d = self.items[i]
        return self.tf(load_glyph(p)), y, s, d


def load_real_artifact_pools(real_dir: Path, cls2id: dict[str, int]):
    """Load all 5 real-artifact training pools (strictly disjoint from all held-out test sets)."""
    pools = {
        "museum": defaultdict(list),
        "obimd": defaultdict(list),
        "bronze_stone": defaultdict(list),
        "chubs": defaultdict(list),
        "hanyin": defaultdict(list),
    }

    def add_json_pool(key: str, meta_path: Path, default_style: str):
        if not meta_path.exists():
            return 0
        recs = json.loads(meta_path.read_text(encoding="utf-8"))
        n = 0
        for r in recs:
            bw = real_dir / r["bw_path"]
            raw = real_dir / r.get("raw_path", r["bw_path"])
            if not bw.is_file():
                continue
            ch = simp(r["char"])
            if not is_valid_cjk(ch):
                continue
            st = r.get("style") or default_style
            cid = cls2id.get(f"{ch}__{st}", -1)
            grp = r.get("plate") or r.get("work") or ""
            pools[key][ch].append((str(bw), str(raw if raw.is_file() else bw), cid, grp, st))
            n += 1
        return n

    # 1. Museum calligraphy Wave 1-6 (auto_accept ONLY; goldmine_review is strictly held-out test!)
    mus_meta = real_dir / "museum/museum_meta.jsonl"
    n_mus = 0
    if mus_meta.exists():
        for line in mus_meta.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("bucket") != "auto_accept":
                continue
            bw = real_dir / r["bw_path"]
            raw = real_dir / r.get("raw_path", r["bw_path"])
            if not bw.is_file():
                continue
            ch = simp(r["char"])
            if not is_valid_cjk(ch):
                continue
            st = r.get("style", "unknown")
            cid = cls2id.get(f"{ch}__{st}", -1)
            pools["museum"][ch].append((str(bw), str(raw if raw.is_file() else bw), cid, r.get("work", ""), st))
            n_mus += 1

    n_ob = add_json_pool("obimd", real_dir / "obimd/obimd_train_meta.json", "jiaguwen")
    n_jw = add_json_pool("bronze_stone", real_dir / "bronze_stone/jinwen_train_meta.json", "jinwen")
    n_v10 = add_json_pool("bronze_stone", real_dir / "bronze_stone/v10_artifacts_meta.json", "jinwen")
    n_ch = add_json_pool("chubs", real_dir / "chubs/chubs_train_meta.json", "chuwenzi")
    n_hy = add_json_pool("hanyin", real_dir / "hanyin/hanyin_train_meta.json", "zhuanke")
    counts = {"museum": n_mus, "obimd": n_ob, "bronze_stone": n_jw + n_v10, "chubs": n_ch, "hanyin": n_hy}
    return pools, counts


class PairedMultiStreamDataset(Dataset):
    """
    Balanced 6-stream paired contrastive + ArcFace dataset (Clean V10).
    Gracefully supports both:
      - Full mode (clean_base + all 5 real-artifact pools), AND
      - Real-only mode (if user only downloaded the 2.3 GB `train_real_artifacts`!).
    """
    def __init__(self, clean_by_char, clean_by_char_style, pools: dict, char2int: dict[str, int], tf, n_pairs: int = 300000):
        self.clean_by_char = clean_by_char
        self.clean_by_char_style = clean_by_char_style
        self.pools = pools
        self.char2int = char2int
        self.tf = tf
        self.n_pairs = n_pairs
        self.has_clean = len(clean_by_char) > 0

        self.stream_meta = {}
        weights_cfg = [
            ("chubs", 0.26, 0.36, ("chuwenzi", "jiandu", "jinwen"), 0.48),
            ("hanyin", 0.14, 0.45, ("zhuanke", "zhuanshu", "xiaozhuan"), 0.55),
            ("bronze_stone", 0.12, 0.45, ("jinwen", "qinwenzi", "zhuanshu", "jiandu"), 0.55),
            ("obimd", 0.14, 0.38, ("jiaguwen",), 0.52),
            ("museum", 0.18, 0.35, (), 0.48),
        ]
        active = []
        for name, prob, pwr, pref_styles, cross_p in weights_cfg:
            pool = pools.get(name, {})
            chs = [c for c, lst in pool.items() if len(lst) >= 2 or len(clean_by_char.get(c, [])) >= 1]
            if not chs:
                continue
            w = np.array([len(pool[c]) ** pwr for c in chs], dtype=np.float64)
            self.stream_meta[name] = (pool, chs, np.cumsum(w) / w.sum(), pref_styles, cross_p)
            active.append((name, prob))

        self.clean_multi_chars = [c for c, lst in clean_by_char.items() if len(lst) >= 2]
        if self.clean_multi_chars:
            active.append(("clean", 0.16))
        if not active:
            raise RuntimeError("没有任何可用的训练样本流！请检查 --real-dir 或 --base-dir。")

        tot_p = sum(p for _, p in active)
        self.stream_names = [n for n, _ in active]
        self.stream_cdf = np.cumsum([p / tot_p for _, p in active])

        target_styles = ("qinwenzi", "jiandu", "chuwenzi", "zhuanke", "jinwen", "jiaguwen")
        self.clean_preqin = [
            (c, st) for (c, st), lst in clean_by_char_style.items() if st in target_styles and len(lst) >= 2
        ]
        if self.clean_preqin:
            pw = np.array([2.2 if st in ("qinwenzi", "jiandu") else 1.2 for _, st in self.clean_preqin], dtype=np.float64)
            self.clean_preqin_cdf = np.cumsum(pw) / pw.sum()
        else:
            self.clean_preqin_cdf = None

    def __len__(self):
        return self.n_pairs

    def _load(self, path: str, auto_inv: bool) -> torch.Tensor:
        try:
            im = load_rubbing_autoinv(path) if auto_inv else load_glyph(path)
            return self.tf(im)
        except Exception:
            return torch.zeros(3, 224, 224)

    def _pick(self, rec):
        bw, raw, cid, grp = rec[0], rec[1], rec[2], rec[3]
        if raw != bw and random.random() < 0.25:
            return raw, True, cid, grp
        return bw, False, cid, grp

    def __getitem__(self, idx):
        si = int(np.searchsorted(self.stream_cdf, random.random()))
        sname = self.stream_names[min(si, len(self.stream_names) - 1)]

        if sname == "clean":
            if self.clean_preqin and random.random() < 0.50:
                pi = int(np.searchsorted(self.clean_preqin_cdf, random.random()))
                ch, st = self.clean_preqin[min(pi, len(self.clean_preqin) - 1)]
                (p1, cid1), (p2, cid2) = random.sample(self.clean_by_char_style[(ch, st)], 2)
            else:
                ch = random.choice(self.clean_multi_chars)
                (p1, cid1), (p2, cid2) = random.sample(self.clean_by_char[ch], 2)
            inv1 = inv2 = False
        else:
            pool, chs, cdf, pref_styles, cross_p = self.stream_meta[sname]
            ci = int(np.searchsorted(cdf, random.random()))
            ch = chs[min(ci, len(chs) - 1)]
            lst = pool[ch]
            r1 = random.choice(lst)
            p1, inv1, cid1, grp1 = self._pick(r1)

            clean_pref = []
            for st in pref_styles:
                clean_pref.extend(self.clean_by_char_style.get((ch, st), []))
            clean_any = self.clean_by_char.get(ch, [])

            if (clean_pref or clean_any) and (len(lst) == 1 or random.random() < cross_p):
                if clean_pref and random.random() < 0.85:
                    p2, cid2 = random.choice(clean_pref)
                else:
                    p2, cid2 = random.choice(clean_any)
                inv2 = False
            else:
                diff = [r for r in lst if r[3] != grp1]
                r2 = random.choice(diff) if diff else random.choice(lst)
                p2, inv2, cid2, _ = self._pick(r2)

        return self._load(p1, inv1), self._load(p2, inv2), cid1, cid2, self.char2int.get(ch, -1)
