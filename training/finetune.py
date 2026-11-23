#!/usr/bin/env python3
"""
汉火识字 Clean V10 多流域自适应微调 + WiSE-FT 权重融合 (Step 2)。

支持两种模式：
  1. 轻量微调（仅需 2.3 GB `data/real_artifacts/` + 214 MB `checkpoints/wiseft_v10_a070.pt`）：
     python -m training.finetune --base-ckpt checkpoints/wiseft_v10_a070.pt --steps 3000 --alpha 0.70
  2. 全量 6 流微调（同时存在 `data/clean_base/` 与 `data/real_artifacts/` 时自动启用完整 6 流配比）：
     python -m training.finetune --base-ckpt checkpoints/wiseft_v10_a070.pt --steps 4500 --alpha 0.70
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.dataset import PairedMultiStreamDataset, load_clean_base, load_real_artifact_pools
from training.model import blend_state_dicts, load_glyph_model, make_finetune_transforms, simp

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="汉火识字多流域自适应微调 + WiSE-FT 融合")
    ap.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/wiseft_v10_a070.pt")
    ap.add_argument("--classes-dir", type=Path, default=ROOT / "checkpoints/classes_meta")
    ap.add_argument("--real-dir", type=Path, default=ROOT / "data/real_artifacts")
    ap.add_argument("--base-dir", type=Path, default=ROOT / "data/clean_base")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "work/finetune")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bb-lr", type=float, default=1.8e-5)
    ap.add_argument("--head-lr", type=float, default=5.5e-5)
    ap.add_argument("--tau", type=float, default=0.065)
    ap.add_argument("--alpha", type=float, default=0.70, help="WiSE-FT 融合系数: (1-a)*base + a*ft (设为 1.0 则不融合)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args()

    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = device == "cuda"

    # Load class vocabulary (prefer classes_meta, fallback to clean_base)
    classes = []
    if (args.classes_dir / "classes.json").exists():
        classes = json.loads((args.classes_dir / "classes.json").read_text(encoding="utf-8"))
    elif (args.base_dir / "classes_styles_dynasties.json").exists():
        classes = json.loads((args.base_dir / "classes_styles_dynasties.json").read_text(encoding="utf-8"))["classes"]
    cls2id = {c: i for i, c in enumerate(classes)}

    # Optional clean_base stream (if downloaded)
    tracked_styles = {"jiaguwen", "jinwen", "chuwenzi", "qinwenzi", "jiandu", "zhuanke", "zhuanshu", "xiaozhuan"}
    clean_by_char = defaultdict(list)
    clean_by_char_style = defaultdict(list)
    if (args.base_dir / "clean_base_meta.jsonl.gz").exists() and (args.base_dir / "all_data").exists():
        base_classes, _, _, train_items, _ = load_clean_base(args.base_dir)
        if not cls2id:
            cls2id = {c: i for i, c in enumerate(base_classes)}
        for p, cid, _, _ in train_items:
            ch_raw, st = base_classes[cid].split("__", 1)
            ch = simp(ch_raw)
            clean_by_char[ch].append((p, cid))
            if st in tracked_styles:
                clean_by_char_style[(ch, st)].append((p, cid))
        print(f"[data] Loaded clean_base train={len(train_items):,} ({len(clean_by_char):,} chars)", flush=True)
    else:
        print("[data] clean_base 未下载，自动以真迹多流模式微调 (如需 6 流全量可运行 download_dataset.py --base)", flush=True)

    pools, counts = load_real_artifact_pools(args.real_dir, cls2id)
    print(f"[data] Loaded real_artifacts: {counts}", flush=True)

    all_chars = set(clean_by_char.keys())
    for pool in pools.values():
        all_chars.update(pool.keys())
    char2int = {c: i for i, c in enumerate(sorted(all_chars))}

    model, sd_base = load_glyph_model(args.base_ckpt, device=device)
    sd_base_cpu = {k: v.detach().cpu().clone() for k, v in sd_base.items()}
    print(f"[model] Loaded base checkpoint {args.base_ckpt} on {device}", flush=True)

    bb_params, head_params = [], []
    for name, p in model.named_parameters():
        (bb_params if name.startswith("backbone.") else head_params).append(p)

    opt = torch.optim.AdamW([
        {"params": bb_params, "lr": args.bb_lr, "weight_decay": 1e-4},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 1e-4},
    ])

    def lr_lambda(step: int):
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(device, enabled=has_cuda)

    ds = PairedMultiStreamDataset(
        clean_by_char, clean_by_char_style, pools, char2int,
        make_finetune_transforms(), n_pairs=args.steps * args.batch_size + 512,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=has_cuda, drop_last=True,
    )

    model.train()
    t_train = time.time()
    acc_con = acc_arc = 0.0
    n_log = 0

    for step, (x1, x2, cid1, cid2, ch_id) in enumerate(loader, start=1):
        if step > args.steps:
            break
        x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
        cid1, cid2 = cid1.to(device, non_blocking=True), cid2.to(device, non_blocking=True)
        ch_id = ch_id.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device, enabled=has_cuda):
            emb = model.embed(torch.cat([x1, x2], dim=0))
            e1, e2 = emb[: args.batch_size], emb[args.batch_size :]

            sim = (e1 @ e2.T) / args.tau
            same_char = (ch_id[:, None] == ch_id[None, :]) & (ch_id[:, None] >= 0)
            eye = torch.eye(args.batch_size, dtype=torch.bool, device=sim.device)
            sim_masked = sim.masked_fill(same_char & (~eye), -1e4)
            labels = torch.arange(args.batch_size, device=sim.device)
            loss_con = 0.5 * (F.cross_entropy(sim_masked, labels) + F.cross_entropy(sim_masked.T, labels))

            cids = torch.cat([cid1, cid2], dim=0)
            valid = cids >= 0
            if valid.any():
                loss_arc = F.cross_entropy(model.head(emb[valid], cids[valid]), cids[valid])
            else:
                loss_arc = torch.tensor(0.0, device=sim.device)
            loss = loss_con + 0.35 * loss_arc

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        acc_con += float(loss_con.item())
        acc_arc += float(loss_arc.item())
        n_log += 1

        if step == 1 or step % 100 == 0 or step == args.steps:
            print(
                f"step {step:4d}/{args.steps} | con={acc_con/n_log:.4f} | arc={acc_arc/n_log:.4f} "
                f"| lr={opt.param_groups[0]['lr']:.2e} | elapsed={(time.time()-t_train)/60:.1f}m",
                flush=True,
            )
            acc_con = acc_arc = 0.0
            n_log = 0

        if step in (1500, 3000, 4500) and step != args.steps:
            torch.save({"model": model.state_dict(), "step": step}, args.out_dir / f"ft_s{step}.pt")

    ft_path = args.out_dir / f"ft_s{args.steps}.pt"
    sd_ft_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model": sd_ft_cpu, "step": args.steps}, ft_path)
    print(f"[save] Saved raw fine-tuned checkpoint -> {ft_path}", flush=True)

    if 0.0 < args.alpha < 1.0:
        sd_blend = blend_state_dicts(sd_base_cpu, sd_ft_cpu, args.alpha)
        blend_path = args.out_dir / f"wiseft_a{int(round(args.alpha*100)):03d}.pt"
        torch.save({"model": sd_blend, "alpha": args.alpha, "base_ckpt": str(args.base_ckpt)}, blend_path)
        print(f"[save] Saved WiSE-FT blended checkpoint (alpha={args.alpha:.2f}) -> {blend_path}", flush=True)

    print(f"Fine-tune DONE in {(time.time()-t0)/60:.1f}m!", flush=True)


if __name__ == "__main__":
    main()
