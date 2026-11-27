#!/usr/bin/env python3
"""
基于 CJKVI IDS 88,939 字形分解表，为汉火识字全部 12,368 个汉字构建多标签偏旁/部件结构监督词表
(checkpoints/classes_meta/ids_components.json)。

特性：
  - 递归展开多层 IDS 分解树（如「湖」-> {⿰, 氵, 胡, 古, 月, 十, 口}），既保留直接部件又保留底层原子部首；
  - 同步提取 12 种构字结构符（⿰左右、⿱上下、⿴全包围、⿸左上包围等），让模型显式感知字形拓扑；
  - 自动双向兼容简繁字头，过滤出现 <5 次的孤立罕见部件，产出紧凑高效的 ~650 维多标签监督表。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from opencc import OpenCC

SHIZI = Path(__file__).resolve().parent.parent
IDS_TXT = SHIZI / "data/cjkvi_ids.txt"
CLASSES_JSON = SHIZI / "checkpoints/classes_meta/classes.json"
OUT_JSON = SHIZI / "checkpoints/classes_meta/ids_components.json"

IDC_OPS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")
TAG_RE = re.compile(r"\[[^\]]*\]|\{[^}]*\}")


def parse_ids_file(path: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ch = parts[1].strip()
        if len(ch) != 1:
            continue
        # Prefer first/cleanest decomposition (strip region tags like [G], [T], [J])
        expr = TAG_RE.sub("", parts[2].split("\t")[0]).strip()
        if expr:
            table[ch] = expr
    return table


def decompose_recursive(ch: str, table: dict[str, str], max_depth: int = 3) -> set[str]:
    comps: set[str] = set()
    visited: set[str] = {ch}

    def dfs(c: str, depth: int):
        expr = table.get(c)
        if not expr or expr == c or depth >= max_depth:
            if c != ch:
                comps.add(c)
            return
        for tok in expr:
            if tok in IDC_OPS:
                if depth == 0:
                    comps.add(tok)  # Record top-level structural layout operator
                continue
            if ord(tok) < 0x2E80:
                continue
            comps.add(tok)
            if tok not in visited:
                visited.add(tok)
                dfs(tok, depth + 1)

    dfs(ch, 0)
    return comps


def main():
    ids_table = parse_ids_file(IDS_TXT)
    classes = json.loads(CLASSES_JSON.read_text(encoding="utf-8"))
    t2s, s2t = OpenCC("t2s"), OpenCC("s2t")

    uniq_chars = sorted({c.split("__", 1)[0] for c in classes})
    # Include both simplified and traditional forms in lookup table
    all_chars = sorted(
        set(uniq_chars)
        | {t2s.convert(c) for c in uniq_chars if len(t2s.convert(c)) == 1}
        | {s2t.convert(c) for c in uniq_chars if len(s2t.convert(c)) == 1}
    )

    raw_decomp: dict[str, set[str]] = {}
    comp_freq = Counter()
    for ch in all_chars:
        cs = decompose_recursive(ch, ids_table, max_depth=3)
        if not cs:
            tc = s2t.convert(ch)
            if len(tc) == 1 and tc != ch:
                cs = decompose_recursive(tc, ids_table, max_depth=3)
        if not cs:
            sc = t2s.convert(ch)
            if len(sc) == 1 and sc != ch:
                cs = decompose_recursive(sc, ids_table, max_depth=3)
        if not cs:
            # Atomic / pictographic base character (e.g. 永/馬/龍/龜/日/月) acts as its own radical!
            cs = {ch}
        raw_decomp[ch] = cs
        for tok in cs:
            comp_freq[tok] += 1

    # Keep all 12 IDC operators + recurring components appearing in >= 5 distinct characters
    vocab = sorted(IDC_OPS) + [tok for tok, cnt in comp_freq.most_common() if tok not in IDC_OPS and cnt >= 5]
    tok2id = {t: i for i, t in enumerate(vocab)}

    char2ids: dict[str, list[int]] = {}
    for ch, cs in raw_decomp.items():
        ids = sorted({tok2id[t] for t in cs if t in tok2id})
        if ids:
            char2ids[ch] = ids

    cov = sum(1 for c in uniq_chars if c in char2ids)
    avg_c = sum(len(v) for v in char2ids.values()) / max(1, len(char2ids))
    print(
        f"[IDS] Vocab={len(vocab)} components | Covered {cov:,}/{len(uniq_chars):,} ({100*cov/len(uniq_chars):.2f}%) "
        f"unique chars | avg {avg_c:.2f} components/char",
        flush=True,
    )
    for sample in ["永", "明", "湖", "蘭", "鼎", "龜", "龍", "馬"]:
        toks = [vocab[i] for i in char2ids.get(sample, [])]
        print(f"  {sample}: {toks}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"vocab": vocab, "char2ids": char2ids}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"[IDS] Saved -> {OUT_JSON} ({OUT_JSON.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
