#!/usr/bin/env python3
"""
一键从 Hugging Face 数据集 (hanhuo-org/hanhuo-shizi-dataset) 按需下载并解压「汉火识字」图片数据。

支持模块化按需拉取（不浪费硬盘与带宽）：
  --web-packs  (~2.5 GB)  离线 Web 原图包 (packs_v2/) + 整页拓片包 (stele_pages/) -> web/packs/
  --real       (~2.3 GB)  真迹域自适应微调 + 12 项零泄漏评测集 (名碑帖/甲骨文/金文/战国楚简/汉印/石鼓文) -> data/real_artifacts/
  --base       (~15 GB)   123.5 万清洗后多书体基础字形分片 (16 tar) -> data/clean_base/
  --detector   (~237 MB)  CenterNet 多字检测器合成字池 -> data/detector_pool/
  --all                   拉取上述全部数据

用法：
  python scripts/download_dataset.py --web-packs
  python scripts/download_dataset.py --real
  python scripts/download_dataset.py --all
"""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "hanhuo-org/hanhuo-shizi-dataset"


def extract_tars(tar_dir: Path, dst_dir: Path, remove_tar: bool = False):
    dst_dir.mkdir(parents=True, exist_ok=True)
    tars = sorted(list(tar_dir.glob("*.tar")) + list(tar_dir.glob("*.tar.gz")))
    for tp in tars:
        marker = dst_dir / f".extracted_{tp.name}"
        if marker.exists():
            print(f"  [SKIP] 已解压: {tp.name}", flush=True)
            continue
        print(f"  [EXT ] 解压 {tp.name} -> {dst_dir} ...", flush=True)
        mode = "r:gz" if tp.name.endswith(".gz") else "r:"
        with tarfile.open(tp, mode) as tf:
            tf.extractall(dst_dir, filter="data")
        marker.write_text("ok\n", encoding="utf-8")
        if remove_tar:
            tp.unlink()


def main():
    ap = argparse.ArgumentParser(description="从 Hugging Face 下载汉火识字模块化数据集")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"HF Dataset 仓库 (默认 {DEFAULT_REPO})")
    ap.add_argument("--web-packs", action="store_true", help="下载离线 Web 原图包 + 整页拓片包 (~2.5 GB)")
    ap.add_argument("--real", action="store_true", help="下载真迹微调 + 12项零泄漏评测集 (~2.3 GB)")
    ap.add_argument("--base", action="store_true", help="下载 123.5 万基础字库 tar 分片 (~15 GB)")
    ap.add_argument("--detector", action="store_true", help="下载多字检测器字池 (~237 MB)")
    ap.add_argument("--all", action="store_true", help="下载全部模块")
    ap.add_argument("--no-extract", action="store_true", help="仅下载 tar 分片不自动解压")
    ap.add_argument("--remove-tar", action="store_true", help="解压完成后删除 tar 压缩包节省空间")
    args = ap.parse_args()

    if not any([args.web_packs, args.real, args.base, args.detector, args.all]):
        ap.error("请至少指定一个模块: --web-packs / --real / --base / --detector / --all")

    want_web = args.all or args.web_packs
    want_real = args.all or args.real
    want_base = args.all or args.base
    want_det = args.all or args.detector

    if want_web:
        packs_dir = ROOT / "web" / "packs"
        print(f"[1] 下载 Web 离线原图包与整页拓片 -> {packs_dir}", flush=True)
        dl = Path(
            snapshot_download(
                repo_id=args.repo,
                repo_type="dataset",
                allow_patterns=["web_packs/**"],
                local_dir=str(ROOT / "data" / "_hf_cache"),
            )
        )
        src_web = dl / "web_packs"
        packs_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("packs_v2", "stele_pages"):
            s = src_web / sub
            d = packs_dir / sub
            if s.exists() and not d.exists():
                d.symlink_to(s.resolve())
        print(f"    完成！现在运行 `python web/serve_local.py` 即可离线浏览 21.7 万原图与 5,651 张整页拓片。", flush=True)

    if want_real:
        print("[2] 下载真迹域自适应微调集与 12 项评测集 (~2.3 GB)...", flush=True)
        dl = Path(
            snapshot_download(
                repo_id=args.repo,
                repo_type="dataset",
                allow_patterns=["train_real_artifacts/**"],
                local_dir=str(ROOT / "data" / "_hf_cache"),
            )
        )
        if not args.no_extract:
            extract_tars(dl / "train_real_artifacts", ROOT / "data" / "real_artifacts", remove_tar=args.remove_tar)

    if want_base:
        print("[3] 下载 123.5 万清洗多书体基础字库 (~15 GB)...", flush=True)
        dl = Path(
            snapshot_download(
                repo_id=args.repo,
                repo_type="dataset",
                allow_patterns=["train_clean_base/clean_base_*.tar", "train_clean_base/clean_base_meta.jsonl.gz", "train_clean_base/classes_styles_dynasties.json"],
                local_dir=str(ROOT / "data" / "_hf_cache"),
            )
        )
        src_base = dl / "train_clean_base"
        dst_base = ROOT / "data" / "clean_base"
        dst_base.mkdir(parents=True, exist_ok=True)
        for mf in ("clean_base_meta.jsonl.gz", "classes_styles_dynasties.json"):
            s, d = src_base / mf, dst_base / mf
            if s.exists() and not d.exists():
                d.symlink_to(s.resolve())
        if not args.no_extract:
            extract_tars(src_base, dst_base, remove_tar=args.remove_tar)

    if want_det:
        print("[4] 下载 CenterNet 多字检测器字池 (~237 MB)...", flush=True)
        dl = Path(
            snapshot_download(
                repo_id=args.repo,
                repo_type="dataset",
                allow_patterns=["train_clean_base/detector_pool.tar.gz"],
                local_dir=str(ROOT / "data" / "_hf_cache"),
            )
        )
        if not args.no_extract:
            extract_tars(dl / "train_clean_base", ROOT / "data", remove_tar=args.remove_tar)

    print("\n全部所选数据就绪！", flush=True)


if __name__ == "__main__":
    main()
