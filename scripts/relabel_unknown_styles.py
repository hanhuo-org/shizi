#!/usr/bin/env python3
"""
Step 1: 高精度回标 `unknown` 书体原型 + 清洗异常朝代标签（如 "颂鼎" -> "周"）。

原理：
- 用现有 19.6 万条已知书体原型（512 维特征 + 朝代 one-hot）训练 17 类书体 MLP 分类器（15k 留出验证）；
- 仅对置信度 >= threshold（默认 0.65，验证集准确率 > 91%）的 `unknown` 原型回填真实书体；
- 合并异常朝代名 "颂鼎" -> "周"，同步更新 `style_counts` 与 `style_orig_counts`；
- 写出更新后的 `proto_meta.json.gz`（并保留备份与映射表，供 V11 复现复用）。
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SHIZI_ROOT = Path(__file__).resolve().parent.parent
if str(SHIZI_ROOT) not in sys.path:
    sys.path.insert(0, str(SHIZI_ROOT))
from training.evaluate import load_hxv1_from_dist


def main():
    ap = argparse.ArgumentParser(description="回标 unknown 书体原型并清洗异常朝代")
    ap.add_argument("--dist-dir", type=Path, default=SHIZI_ROOT / "web/dist")
    ap.add_argument("--conf", type=float, default=0.65, help="回标最低置信度阈值")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--save-map", type=Path, default=Path("/mnt/s790/project/hanzi_train/work_v11/unknown_relabel_map.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, meta = load_hxv1_from_dist(args.dist_dir)
    styles: list[str] = list(meta["styles"])
    dynasties: list[str] = list(meta["dynasties"])
    sids = np.array(meta["style_ids"], dtype=np.int64)
    dids = np.array(meta["dynasty_ids"], dtype=np.int64)
    unk_id = styles.index("unknown")

    # 1. 清洗异常朝代标签 ("颂鼎" -> "周")
    dyn_fixed = 0
    if "颂鼎" in dynasties and "周" in dynasties:
        sd_idx = dynasties.index("颂鼎")
        zhou_idx = dynasties.index("周")
        m_sd = dids == sd_idx
        dyn_fixed = int(m_sd.sum())
        dids[m_sd] = zhou_idx
        # 移除 "颂鼎" 并压缩索引
        dynasties.pop(sd_idx)
        dids = np.where(dids > sd_idx, dids - 1, dids)
        print(f"[Dynasty] Merged '颂鼎' ({dyn_fixed} protos) -> '周' | dynasties={len(dynasties)}", flush=True)

    # 2. 构造 [512-d embedding + dynasty one-hot] 特征
    n_dyn = len(dynasties)
    D_in = P.shape[1] + n_dyn
    dyn_oh = np.zeros((len(dids), n_dyn), dtype=np.float32)
    dyn_oh[np.arange(len(dids)), np.clip(dids, 0, n_dyn - 1)] = 0.25  # 温和权重避免朝代盖过字形视觉特征
    X_all = np.concatenate([P, dyn_oh], axis=1)

    known_idx = np.where(sids != unk_id)[0]
    unk_idx = np.where(sids == unk_id)[0]
    rng = np.random.default_rng(42)
    perm = rng.permutation(known_idx)
    n_val = min(15000, len(perm) // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr = torch.from_numpy(X_all[tr_idx]).to(device)
    y_tr = torch.from_numpy(sids[tr_idx]).to(device)
    X_va = torch.from_numpy(X_all[val_idx]).to(device)
    y_va = torch.from_numpy(sids[val_idx]).to(device)

    torch.manual_seed(42)
    clf = nn.Sequential(
        nn.Linear(D_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.15),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.10),
        nn.Linear(256, len(styles)),
    ).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bs = 2048
    for ep in range(args.epochs):
        clf.train()
        order = torch.randperm(len(X_tr), device=device)
        for i in range(0, len(X_tr), bs):
            b = order[i : i + bs]
            logits = clf(X_tr[b])
            logits[:, unk_id] = -1e4
            loss = F.cross_entropy(logits, y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    clf.eval()
    with torch.no_grad():
        va_logits = clf(X_va)
        va_logits[:, unk_id] = -1e9
        va_probs = torch.softmax(va_logits, dim=1)
        va_conf, va_pred = va_probs.max(1)
        full_acc = (va_pred == y_va).float().mean().item() * 100
        m_thr = va_conf >= args.conf
        thr_cov = m_thr.float().mean().item() * 100
        thr_acc = (va_pred[m_thr] == y_va[m_thr]).float().mean().item() * 100
        print(
            f"[Val] 17-class acc={full_acc:.2f}% | @conf>={args.conf:.2f}: cov={thr_cov:.1f}%, acc={thr_acc:.2f}% ({time.time()-t0:.1f}s)",
            flush=True,
        )

        # 推理全部 unknown 原型
        X_unk = torch.from_numpy(X_all[unk_idx]).to(device)
        u_logits = clf(X_unk)
        u_logits[:, unk_id] = -1e9
        u_probs = torch.softmax(u_logits, dim=1)
        u_conf, u_pred = u_probs.max(1)
        u_conf = u_conf.cpu().numpy()
        u_pred = u_pred.cpu().numpy()

    relabel_mask = u_conf >= args.conf
    n_relabel = int(relabel_mask.sum())
    relabel_dist = collections.Counter(styles[p] for p in u_pred[relabel_mask])
    print(f"[Relabel] {n_relabel:,}/{len(unk_idx):,} ({100*n_relabel/max(1,len(unk_idx)):.1f}%) unknown prototypes relabeled:")
    for st, cnt in relabel_dist.most_common():
        print(f"  + {st:12s}: {cnt:5,d}")

    if args.dry_run:
        return

    # 3. 更新 style_ids / style_counts / style_orig_counts
    old_unk_protos = int((sids == unk_id).sum())
    old_unk_orig = int(meta.get("style_orig_counts", {}).get("unknown", old_unk_protos))
    avg_orig_per_unk = old_unk_orig / max(1, old_unk_protos)

    new_sids = sids.copy()
    relabel_map = {}
    for ui, pred_id, conf_v in zip(unk_idx[relabel_mask], u_pred[relabel_mask], u_conf[relabel_mask]):
        new_sids[ui] = int(pred_id)
        relabel_map[int(ui)] = [styles[int(pred_id)], round(float(conf_v), 4)]

    new_style_counts = {s: int((new_sids == i).sum()) for i, s in enumerate(styles)}
    new_orig_counts = dict(meta.get("style_orig_counts", {}))
    moved_orig_total = 0
    for st, cnt in relabel_dist.items():
        add_orig = int(round(cnt * avg_orig_per_unk))
        new_orig_counts[st] = int(new_orig_counts.get(st, 0)) + add_orig
        moved_orig_total += add_orig
    new_orig_counts["unknown"] = max(new_style_counts["unknown"], old_unk_orig - moved_orig_total)

    meta["dynasties"] = dynasties
    meta["dynasty_ids"] = dids.tolist()
    meta["style_ids"] = new_sids.tolist()
    meta["style_counts"] = new_style_counts
    meta["style_orig_counts"] = new_orig_counts

    meta_p = args.dist_dir / "assets/proto_meta.json.gz"
    bak_p = args.dist_dir / "assets/proto_meta.v10_pre_relabel.json.gz"
    if not bak_p.exists():
        shutil.copy2(meta_p, bak_p)
    meta_bytes = gzip.compress(json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), compresslevel=9)
    meta_p.write_bytes(meta_bytes)
    print(f"[Saved] Updated {meta_p} ({len(meta_bytes)/1024:.1f} KB) | remaining unknown={new_style_counts['unknown']:,}", flush=True)

    args.save_map.parent.mkdir(parents=True, exist_ok=True)
    args.save_map.write_text(json.dumps({
        "conf_threshold": args.conf,
        "val_full_acc": round(full_acc, 2),
        "val_thr_acc": round(thr_acc, 2),
        "val_thr_cov": round(thr_cov, 2),
        "relabeled_count": n_relabel,
        "remaining_unknown": new_style_counts["unknown"],
        "dynasty_fixed": dyn_fixed,
        "dist": dict(relabel_dist),
        "map": relabel_map,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved] Relabel map -> {args.save_map}", flush=True)


if __name__ == "__main__":
    main()
