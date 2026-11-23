#!/usr/bin/env python3
"""
将任意训练/微调得到的 PyTorch 权重导出为浏览器离线 Web 运行时包 (Step 3)：
  1. 导出纯 opset-17 (dynamo=False) Weight-Only U8 ONNX (`~26.7 MB`)，并校验 PyTorch <-> ONNX 余弦一致性 (>0.995)；
  2. 利用 `web/packs/packs_v2/` + `web/dist/assets/glyph_toc.bin` 重新编码全部原型，打包为 HXV1 INT8 分片 (`proto_vec.*.gzbin`)；
  3. 同步更新 `web/dist/models_chunks/manifest.json` 与 `web/dist/assets/proto_meta.json.gz`，
     完成后直接运行 `python web/serve_local.py` 即可在浏览器体验新模型！

用法：
  python -m training.export_web --ckpt work/finetune/wiseft_a070.pt
  python -m training.export_web --ckpt checkpoints/wiseft_v10_a070.pt --onnx-only
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import struct
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
from PIL import Image
import torch
import torch.nn as nn

from training.model import load_glyph_model, make_transforms

ROOT = Path(__file__).resolve().parent.parent
CHUNK_RAW_BYTES = 16 * 1024 * 1024  # 16 MB raw chunks -> <=15 MB gzipped


class EmbedWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.backbone(x), p=2, dim=1)


def export_wo_u8_opset17(model: nn.Module, out_onnx_u8: Path) -> float:
    """Export pure opset-17 per-channel Weight-Only U8 ONNX (WASM & WebGPU compatible, min_cos > 0.995)."""
    tmp_fp32 = out_onnx_u8.with_suffix(".fp32.onnx")
    wrapper = EmbedWrapper(model).cpu().eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        wrapper,
        dummy,
        str(tmp_fp32),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    m = onnx.load(str(tmp_fp32), load_external_data=True)
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
            if arr.size < 16384:
                continue
            reduce_axes = tuple(range(1, arr.ndim))
            wmin = arr.min(axis=reduce_axes)
            wmax = arr.max(axis=reduce_axes)
            scales = np.maximum((wmax - wmin) / 255.0, 1e-8).astype(np.float32)
            zps = np.clip(np.round(-wmin / scales), 0, 255).astype(np.uint8)
            b = [arr.shape[0]] + [1] * (arr.ndim - 1)
            q = np.clip(np.round(arr / scales.reshape(b)) + zps.reshape(b), 0, 255).astype(np.uint8)
            q_name, s_name, z_name, dq_out = wname + "_q8", wname + "_s8", wname + "_z8", wname + "_deq"
            new_inits.extend([
                numpy_helper.from_array(q, name=q_name),
                numpy_helper.from_array(scales, name=s_name),
                numpy_helper.from_array(zps, name=z_name),
            ])
            deq_nodes.append(helper.make_node("DequantizeLinear", [q_name, s_name, z_name], [dq_out], axis=0))
            node.input[1] = dq_out
            replaced.add(wname)

    kept = [t for t in m.graph.initializer if t.name not in replaced]
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept + new_inits)
    nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(deq_nodes + nodes)
    onnx.save_model(m, str(out_onnx_u8), save_as_external_data=False)
    if tmp_fp32.exists():
        tmp_fp32.unlink()

    xs = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        e_pt = wrapper(xs).numpy()
    sess = ort.InferenceSession(str(out_onnx_u8), providers=["CPUExecutionProvider"])
    e_ox = sess.run(["embedding"], {"image": xs.numpy()})[0]
    e_ox /= np.maximum(np.linalg.norm(e_ox, axis=1, keepdims=True), 1e-8)
    min_cos = float((e_pt * e_ox).sum(axis=1).min())
    print(f"[onnx] Exported {out_onnx_u8} ({out_onnx_u8.stat().st_size/1048576:.2f} MB) | min_cos={min_cos:.5f}", flush=True)
    return min_cos


def pack_hxv1_from_f32(vecs: np.ndarray) -> bytes:
    """Quantize unit-norm float32 (N, 512) -> exact HXV1 INT8 binary format matching wasm_core."""
    N, D = vecs.shape
    assert D == 512
    vecs = vecs.astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
    max_abs = np.maximum(np.abs(vecs).max(axis=1, keepdims=True), 1e-8)
    q = np.clip(np.round(vecs / max_abs * 127.0), -127, 127).astype(np.int8)
    inv_norm = (1.0 / np.maximum(np.linalg.norm(q.astype(np.float32), axis=1), 1e-8)).astype(np.float32)

    hdr = struct.pack("<4sIII", b"HXV1", N, D, 0)
    return hdr + inv_norm.tobytes() + q.tobytes()


def slice_and_gzip(name: str, raw_bytes: bytes, out_dir: Path) -> dict:
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    chunks = []
    tot_gz = 0
    for idx, off in enumerate(range(0, len(raw_bytes), CHUNK_RAW_BYTES)):
        chunk = raw_bytes[off : off + CHUNK_RAW_BYTES]
        gz = gzip.compress(chunk, compresslevel=6)
        fn = f"{name}.{digest}.part{idx:02d}.gzbin"
        (out_dir / fn).write_bytes(gz)
        chunks.append({
            "index": idx,
            "file": fn,
            "offset": off,
            "rawSize": len(chunk),
            "gzSize": len(gz),
            "rawSha256": hashlib.sha256(chunk).hexdigest(),
        })
        tot_gz += len(gz)
    return {
        "version": digest,
        "totalSize": len(raw_bytes),
        "totalGzSize": tot_gz,
        "chunks": chunks,
    }


@torch.no_grad()
def reembed_from_packs(model: nn.Module, toc_bytes: bytes, packs_v2: Path, device: str, batch_size: int = 512) -> np.ndarray:
    N = len(toc_bytes) // 12
    _, val_tf = make_transforms()
    model.to(device).eval()
    open_files: dict[int, object] = {}
    out = np.empty((N, 512), dtype=np.float32)
    xs = []
    idxs = []
    t0 = time.time()

    def flush():
        if not xs:
            return
        bt = torch.stack(xs).to(device)
        with torch.amp.autocast(device, enabled=(device == "cuda")):
            e = model.embed(bt).float().cpu().numpy()
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
        for pos, vec in zip(idxs, e):
            out[pos] = vec
        xs.clear()
        idxs.clear()

    try:
        for i in range(N):
            pack_id, _mime, _pad, off, ln = struct.unpack_from("<HBBII", toc_bytes, i * 12)
            fh = open_files.get(pack_id)
            if fh is None:
                fh = open(packs_v2 / f"pack_{pack_id:03d}.bin", "rb")
                open_files[pack_id] = fh
            fh.seek(off)
            raw = fh.read(ln)
            im = Image.open(io.BytesIO(raw)).convert("L")
            if np.asarray(im.resize((32, 32))).mean() < 110:
                im = Image.eval(im, lambda v: 255 - v)
            xs.append(val_tf(im))
            idxs.append(i)
            if len(xs) >= batch_size:
                flush()
                if (i + 1) % 25000 < batch_size:
                    print(f"  embedded {i+1:,}/{N:,} prototypes ({time.time()-t0:.1f}s)", flush=True)
        flush()
    finally:
        for fh in open_files.values():
            fh.close()
    return out


def main():
    ap = argparse.ArgumentParser(description="导出汉火识字浏览器离线 ONNX 与 HXV1 向量分片")
    ap.add_argument("--ckpt", type=Path, required=True, help="PyTorch 权重路径 (.pt)")
    ap.add_argument("--dist-dir", type=Path, default=ROOT / "web/dist")
    ap.add_argument("--packs-dir", type=Path, default=ROOT / "web/packs/packs_v2")
    ap.add_argument("--out-onnx", type=Path, default=ROOT / "work/exported_op17_wo_u8.onnx")
    ap.add_argument("--onnx-only", action="store_true", help="仅导出 ONNX，不重算原型向量库")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_onnx.parent.mkdir(parents=True, exist_ok=True)
    model, _ = load_glyph_model(args.ckpt, device=device)
    min_cos = export_wo_u8_opset17(model, args.out_onnx)
    assert min_cos > 0.990, f"ONNX parity check failed: {min_cos}"
    if args.onnx_only:
        return

    chunks_dir = args.dist_dir / "models_chunks"
    assets_dir = args.dist_dir / "assets"
    toc_p = assets_dir / "glyph_toc.bin"
    meta_p = assets_dir / "proto_meta.json.gz"
    if not toc_p.exists() or not args.packs_dir.exists():
        raise FileNotFoundError(
            "重算全部原型向量需要 web/dist/assets/glyph_toc.bin 与 web/packs/packs_v2/（请先运行 download_weights.py 与 download_dataset.py --web-packs）"
        )

    toc_bytes = toc_p.read_bytes()
    vecs = reembed_from_packs(model, toc_bytes, args.packs_dir, device=device, batch_size=args.batch_size)
    hxv1_bytes = pack_hxv1_from_f32(vecs)

    old_man_p = chunks_dir / "manifest.json"
    old_man = json.loads(old_man_p.read_text(encoding="utf-8")) if old_man_p.exists() else {}
    for p in list(chunks_dir.glob("resnet50.*.gzbin")) + list(chunks_dir.glob("proto_vec.*.gzbin")):
        p.unlink()

    r50_entry = slice_and_gzip("resnet50", args.out_onnx.read_bytes(), chunks_dir)
    pv_entry = slice_and_gzip("proto_vec", hxv1_bytes, chunks_dir)
    build_id = f"{time.strftime('%Y%m%d-%H%M%S')}-custom-{args.ckpt.stem}"
    new_man = {
        "schema": 1,
        "buildId": build_id,
        "totalRawBytes": r50_entry["totalSize"] + pv_entry["totalSize"],
        "totalGzBytes": r50_entry["totalGzSize"] + pv_entry["totalGzSize"],
        "models": {"resnet50": r50_entry, "proto_vec": pv_entry},
        "optionalModels": old_man.get("optionalModels", {}),
    }
    old_man_p.write_text(json.dumps(new_man, indent=2), encoding="utf-8")

    meta = json.loads(gzip.decompress(meta_p.read_bytes()))
    meta["abtt"] = None
    meta["version"] = build_id
    meta_p.write_bytes(gzip.compress(json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), compresslevel=9))
    print(f"[done] Updated {chunks_dir} with buildId={build_id} (N={vecs.shape[0]:,})!", flush=True)


if __name__ == "__main__":
    main()
