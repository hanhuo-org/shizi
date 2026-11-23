#!/usr/bin/env python3
"""
一键下载并安装「汉火识字 Clean V10」预训练权重与本地 Web 离线模型包。

默认从 GitHub Release 拉取（支持通过 --local-dir 直接从本地 release_artifacts/ 安装，方便离线或尚未发版时测试）：
  - shizi-v10-web-models.tar.gz   (~137 MB) -> 解压至 web/dist/（即可直接 python web/serve_local.py 启动本地 Web！）
  - wiseft_v10_a070.pt            (~214 MB) -> checkpoints/wiseft_v10_a070.pt（微调/重算原型用）
  - wiseft_v10_a070_op17_wo_u8.onnx (~27 MB)-> checkpoints/wiseft_v10_a070_op17_wo_u8.onnx
  - char_det_best.pt              (~13 MB)  -> checkpoints/char_det_best.pt
  - char_det_u8.onnx              (~3.7 MB) -> checkpoints/char_det_u8.onnx
  - classes_meta.tar.gz           (~188 KB) -> checkpoints/classes_meta/

用法：
  python scripts/download_weights.py                       # 仅拉取本地 Web 必需包 (~137 MB)
  python scripts/download_weights.py --all                 # 拉取 Web 包 + 全部训练/推理权重 (~393 MB)
  python scripts/download_weights.py --local-dir release_artifacts --all
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "hanhuo-org/shizi"
DEFAULT_TAG = "v10"

WEB_ASSETS = ["shizi-v10-web-models.tar.gz"]
TRAIN_ASSETS = [
    "wiseft_v10_a070.pt",
    "wiseft_v10_a070_op17_wo_u8.onnx",
    "char_det_best.pt",
    "char_det_u8.onnx",
    "classes_meta.tar.gz",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  [GET] {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "HanfireShiziInstaller/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total > 0:
                print(f"\r        {done/1048576:6.1f} / {total/1048576:6.1f} MB ({100*done/total:5.1f}%)", end="", flush=True)
        print("", flush=True)
    tmp.replace(dst)


def fetch_asset(name: str, repo: str, tag: str, cache_dir: Path, local_dir: Path | None) -> Path:
    if local_dir is not None:
        src = local_dir / name
        if not src.exists():
            raise FileNotFoundError(f"本地目录中未找到 {src}")
        return src
    dst = cache_dir / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  [HIT] {dst} ({dst.stat().st_size/1048576:.1f} MB)", flush=True)
        return dst
    url = f"https://github.com/{repo}/releases/download/{tag}/{name}"
    download_url(url, dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description="下载并安装汉火识字 Clean V10 权重与 Web 模型包")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub 仓库 (默认 {DEFAULT_REPO})")
    ap.add_argument("--tag", default=DEFAULT_TAG, help=f"Release 标签 (默认 {DEFAULT_TAG})")
    ap.add_argument("--all", action="store_true", help="同时下载 PyTorch/ONNX 训练与独立推理权重")
    ap.add_argument("--train-only", action="store_true", help="仅下载训练权重，不解压 Web 模型包")
    ap.add_argument("--local-dir", type=Path, default=None, help="从本地打包目录安装（例如 release_artifacts）")
    args = ap.parse_args()

    cache_dir = ROOT / "release_artifacts"
    ckpt_dir = ROOT / "checkpoints"
    dist_dir = ROOT / "web" / "dist"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    # Optional checksum verification
    sums = {}
    try:
        sums_p = fetch_asset("SHA256SUMS.txt", args.repo, args.tag, cache_dir, args.local_dir)
        for line in sums_p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) == 2:
                sums[parts[1]] = parts[0]
    except Exception:
        pass

    targets = []
    if not args.train_only:
        targets.extend(WEB_ASSETS)
    if args.all or args.train_only:
        targets.extend(TRAIN_ASSETS)

    for name in targets:
        p = fetch_asset(name, args.repo, args.tag, cache_dir, args.local_dir)
        if name in sums:
            got = sha256_file(p)
            if got != sums[name]:
                raise RuntimeError(f"SHA256 校验失败: {name} (got={got}, expected={sums[name]})")
            print(f"  [OK ] SHA256 verified: {name}", flush=True)

        if name == "shizi-v10-web-models.tar.gz":
            print(f"  [EXT] 解压 Web 运行时模型包 -> {dist_dir}", flush=True)
            with tarfile.open(p, "r:gz") as tf:
                tf.extractall(dist_dir, filter="data")
        elif name == "classes_meta.tar.gz":
            out_meta = ckpt_dir / "classes_meta"
            out_meta.mkdir(parents=True, exist_ok=True)
            with tarfile.open(p, "r:gz") as tf:
                tf.extractall(out_meta, filter="data")
            print(f"  [EXT] 解压词表元数据 -> {out_meta}", flush=True)
        else:
            dst = ckpt_dir / name
            if p.resolve() != dst.resolve():
                shutil.copy2(p, dst)
            print(f"  [CP ] 安装权重 -> {dst}", flush=True)

    print("\n安装完成！", flush=True)
    if not args.train_only:
        print("  启动本地 Web 识字服务: python web/serve_local.py", flush=True)
    if args.all or args.train_only:
        print(f"  训练/推理权重已就绪于: {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
