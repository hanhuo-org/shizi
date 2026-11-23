#!/usr/bin/env python3
"""
零泄漏真实文物评测脚本：直接加载 web/dist/ 中的 HXV1 INT8 向量库 + PyTorch 权重，
在全部零泄漏真实测试集上实测 Top-1 / Top-5 / Top-12（含线上混合 TTA 0.65*bw + 0.35*inv_gray）：
  - 346-Hard (名碑帖疑难字)
  - OBIMD_Real_Test (998 张未见殷墟甲骨片)
  - Jinwen_Real_Test (114 张未见西周青铜器：虢季子白盘 + 何尊)
  - CHUBS_Real_Test (1,050 张未见战国楚简)
  - HanYin_Real_Test (93 张未见汉代玺印，零重叠字)

用法：
  python -m training.evaluate
  python -m training.evaluate --ckpt checkpoints/wiseft_v10_a070.pt
"""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from training.model import load_glyph_model, load_rubbing_autoinv, make_transforms, simp

ROOT = Path(__file__).resolve().parent.parent


def load_hxv1_from_dist(dist_dir: Path) -> tuple[np.ndarray, dict]:
    man = json.loads((dist_dir / "models_chunks/manifest.json").read_text(encoding="utf-8"))
    chunks = man["models"]["proto_vec"]["chunks"]
    raw = bytearray()
    for ch in chunks:
        raw.extend(gzip.decompress((dist_dir / "models_chunks" / ch["file"]).read_bytes()))
    magic, N, D, _ = struct.unpack_from("<4sIII", raw, 0)
    assert magic == b"HXV1" and D == 512, f"Unexpected HXV1 header: {magic}, N={N}, D={D}"
    inv_norm = np.frombuffer(raw, dtype=np.float32, count=N, offset=16)
    q = np.frombuffer(raw, dtype=np.int8, count=N * D, offset=16 + 4 * N).reshape(N, D).astype(np.float32)
    P = q * inv_norm[:, None]
    meta = json.loads(gzip.decompress((dist_dir / "assets/proto_meta.json.gz").read_bytes()))
    return P, meta


def apply_abtt(X: np.ndarray, abtt: dict | None) -> np.ndarray:
    if not abtt or not abtt.get("mean") or not abtt.get("pcs"):
        return X
    mu = np.asarray(abtt["mean"], dtype=np.float32)
    Y = X - mu[None, :]
    for pc in abtt["pcs"]:
        v = np.asarray(pc, dtype=np.float32)
        Y = Y - (Y @ v)[:, None] * v[None, :]
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-8)


@torch.no_grad()
def embed_images(model, imgs: list[Image.Image], val_tf, device: str, bs: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(imgs), bs):
        xs = torch.stack([val_tf(im.convert("RGB")) for im in imgs[i : i + bs]]).to(device)
        with torch.amp.autocast(device, enabled=(device == "cuda")):
            e = model.embed(xs).float().cpu().numpy()
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
        out.append(e)
    return np.concatenate(out, axis=0)


def score_lse(Q: np.ndarray, P_t: torch.Tensor, chars: list[str], gts: list[str], tau: float = 0.042, device: str = "cuda"):
    Q_t = torch.from_numpy(Q).to(device)
    t1 = t5 = t12 = 0
    for i in range(0, len(Q), 256):
        S = Q_t[i : i + 256] @ P_t.T
        tv, ti = torch.topk(S, k=min(120, P_t.shape[0]), dim=1)
        tv, ti = tv.cpu().numpy(), ti.cpu().numpy()
        for row_idx, (vals, idxs) in enumerate(zip(tv, ti)):
            gt = gts[i + row_idx]
            by_c = {}
            for v, j in zip(vals, idxs):
                c = chars[j]
                if c not in by_c:
                    by_c[c] = float(v)
                else:
                    a, b = by_c[c], float(v)
                    m = max(a, b)
                    by_c[c] = m + tau * float(np.log(np.exp((a - m) / tau) + np.exp((b - m) / tau)))
            ranked = [c for c, _ in sorted(by_c.items(), key=lambda kv: -kv[1])]
            if gt in ranked[:1]:
                t1 += 1
            if gt in ranked[:5]:
                t5 += 1
            if gt in ranked[:12]:
                t12 += 1
    n = max(1, len(gts))
    return 100.0 * t1 / n, 100.0 * t5 / n, 100.0 * t12 / n


def main():
    ap = argparse.ArgumentParser(description="汉火识字零泄漏真实文物评测套件")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints/wiseft_v10_a070.pt")
    ap.add_argument("--dist-dir", type=Path, default=ROOT / "web/dist")
    ap.add_argument("--real-dir", type=Path, default=ROOT / "data/real_artifacts")
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, meta = load_hxv1_from_dist(args.dist_dir)
    chars = [simp(c) for c in meta["chars"]]
    styles = meta["styles"]
    style_ids = np.asarray(meta["style_ids"], dtype=np.int32)
    abtt = meta.get("abtt")
    tau = float(meta.get("lse_tau", 0.042))

    model, _ = load_glyph_model(args.ckpt, device=device)
    model.eval()
    _, val_tf = make_transforms()
    P_t = torch.from_numpy(P).to(device)
    print(f"Loaded HXV1 N={len(chars):,} (version={meta.get('version')}, tau={tau}) on {device} in {time.time()-t0:.1f}s\n", flush=True)

    suites = []
    # 346-Hard from museum_meta.jsonl (v3_w1 + v4_w2 goldmine_review)
    mus_p = args.real_dir / "museum/museum_meta.jsonl"
    if mus_p.exists():
        recs = [json.loads(l) for l in mus_p.open(encoding="utf-8")]
        g346 = [r for r in recs if r.get("bucket") == "goldmine_review" and r.get("wave") in ("v3_w1", "v4_w2")]
        if g346:
            suites.append(("346-Hard (Calligraphy)", g346, None))
    for name, rel_p, st_flt in [
        ("OBIMD_Real_Test (甲骨)", "obimd/obimd_test_meta.json", "jiaguwen"),
        ("Jinwen_Real_Test (金文)", "bronze_stone/jinwen_test_meta.json", "jinwen"),
        ("CHUBS_Real_Test (楚简)", "chubs/chubs_test_meta.json", "chuwenzi"),
        ("HanYin_Real_Test (汉印)", "hanyin/hanyin_test_meta.json", None),
    ]:
        p = args.real_dir / rel_p
        if p.exists():
            suites.append((name, json.loads(p.read_text(encoding="utf-8")), st_flt))

    if not suites:
        print(f"未在 {args.real_dir} 找到评测集，请先运行: python scripts/download_dataset.py --real")
        return

    print(f"{'Benchmark':26s} | {'N':4s} | {'BW ALL T1/T5/T12':20s} | {'Hybrid TTA T1/T5':18s} | {'Style T5':8s}")
    print("-" * 88)
    for name, recs, st_flt in suites:
        bws = [Image.open(args.real_dir / r["bw_path"]).convert("RGB") for r in recs]
        raws = [load_rubbing_autoinv(args.real_dir / r.get("raw_path", r["bw_path"])).convert("RGB") for r in recs]
        gts = [simp(r["char"]) for r in recs]

        q_bw = apply_abtt(embed_images(model, bws, val_tf, device), abtt)
        q_raw = apply_abtt(embed_images(model, raws, val_tf, device), abtt)
        q_hyb = 0.65 * q_bw + 0.35 * q_raw
        q_hyb /= np.maximum(np.linalg.norm(q_hyb, axis=1, keepdims=True), 1e-8)

        t1, t5, t12 = score_lse(q_bw, P_t, chars, gts, tau=tau, device=device)
        h1, h5, _ = score_lse(q_hyb, P_t, chars, gts, tau=tau, device=device)
        st_str = "—"
        if st_flt and st_flt in styles:
            mask = style_ids == styles.index(st_flt)
            idx = np.where(mask)[0]
            _, st5, _ = score_lse(q_bw, P_t[idx], [chars[k] for k in idx], gts, tau=tau, device=device)
            st_str = f"{st5:5.2f}%"
        print(f"{name:26s} | {len(recs):4d} | {t1:5.2f}/{t5:5.2f}/{t12:5.2f}%  | {h1:5.2f}% / {h5:5.2f}%   | {st_str:>8s}", flush=True)


if __name__ == "__main__":
    main()
