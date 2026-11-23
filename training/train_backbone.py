#!/usr/bin/env python3
"""
从零训练 ResNet50 + ArcFace + 书体/朝代多任务基础字形骨干 (Step 1)。

前置数据：
  python scripts/download_dataset.py --base

用法：
  python -m training.train_backbone --epochs 15 --batch-size 256
  python -m training.train_backbone --smoke --epochs 1   # 400 类快速冒烟验证
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.dataset import GlyphClassifyDataset, load_clean_base
from training.model import GlyphModel, make_transforms

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="从零训练 ResNet50+ArcFace 古文字主干")
    ap.add_argument("--base-dir", type=Path, default=ROOT / "data/clean_base")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "work/backbone")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="仅抽 400 类快速验证全链路")
    ap.add_argument("--resume", type=Path, default=None, help="断点续训权重路径")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = device == "cuda"

    classes, styles, dynasties, train_items, val_items = load_clean_base(args.base_dir, smoke=args.smoke)
    print(f"[backbone] classes={len(classes):,} train={len(train_items):,} val={len(val_items):,} device={device}", flush=True)

    train_tf, val_tf = make_transforms()
    cnt = Counter(y for _, y, _, _ in train_items)
    sampler = WeightedRandomSampler([1.0 / cnt[y] for _, y, _, _ in train_items], len(train_items))
    train_loader = DataLoader(GlyphClassifyDataset(train_items, train_tf), batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=has_cuda)
    val_loader = DataLoader(GlyphClassifyDataset(val_items, val_tf), batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=has_cuda)

    model = GlyphModel(len(classes), len(styles), len(dynasties), pretrained=True).to(device)
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("model", ck), strict=True)
        print(f"[backbone] Resumed from {args.resume}", flush=True)

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-5},
        {"params": model.head.parameters(), "lr": 3e-4},
        {"params": model.style_head.parameters(), "lr": 3e-4},
        {"params": model.dyn_head.parameters(), "lr": 3e-4},
    ], weight_decay=1e-4)
    base_lrs = [g["lr"] for g in opt.param_groups]
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device, enabled=has_cuda)

    steps_per_ep = max(1, len(train_loader))
    warm_steps = min(2 * steps_per_ep, max(1, args.epochs * steps_per_ep // 5))
    total_steps = args.epochs * steps_per_ep
    gstep = 0
    best_acc = 0.0

    for ep in range(args.epochs):
        model.head.m = min(0.5, 0.2 + 0.1 * (ep + 1))
        model.train()
        t0 = time.time()
        run_loss = 0.0
        run_n = 0
        for it, (x, y, s, d) in enumerate(train_loader):
            if gstep < warm_steps:
                scale = (gstep + 1) / warm_steps
            else:
                prog = (gstep - warm_steps) / max(1, total_steps - warm_steps)
                scale = max(0.01, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * scale
            gstep += 1

            x, y, s, d = x.to(device), y.to(device), s.to(device), d.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=has_cuda):
                emb = model.backbone(x)
                loss = ce(model.head(emb, y), y) + 0.3 * ce(model.style_head(emb), s) + 0.1 * ce(model.dyn_head(emb), d)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.item()) * len(x)
            run_n += len(x)
            if it % 100 == 0:
                print(f"  ep {ep+1}/{args.epochs} it {it}/{steps_per_ep} loss={run_loss/max(1,run_n):.4f} elapsed={(time.time()-t0)/60:.1f}m", flush=True)

        model.eval()
        correct = total = 0
        with torch.no_grad(), torch.amp.autocast(device, enabled=has_cuda):
            for x, y, _, _ in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model.head.classify(model.backbone(x)).argmax(1)
                correct += int((pred == y).sum().item())
                total += len(y)
        acc = correct / max(1, total)
        print(f"[ep {ep+1}/{args.epochs}] val_acc={acc*100:.2f}% (best={max(best_acc, acc)*100:.2f}%)", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep + 1, "val_acc": acc}, args.out_dir / "last.pt")
        if acc >= best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "epoch": ep + 1, "val_acc": acc}, args.out_dir / "best.pt")

    (args.out_dir / "meta.json").write_text(
        json.dumps({"classes": len(classes), "styles": len(styles), "dynasties": len(dynasties), "best_acc": best_acc}, indent=2),
        encoding="utf-8",
    )
    print(f"Done! Best checkpoint saved to {args.out_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
