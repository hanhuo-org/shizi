#!/usr/bin/env python3
"""
汉火识字 (Hanfire Shizi) 零依赖本地离线 Web 服务器

特性：
  - 自动附加 COOP / COEP / CORP 跨域隔离头，启用浏览器多线程 SIMD WASM / WebGPU 加速；
  - 完整本地支持 `/glyph/<id>`（21.7 万无损代表原图）、`/api/glyph-context/<id>`（碑帖/简帛出处与红框坐标）、
    `/stele-page/<pid>`（5,651 张高清整页拓片）；
  - 若尚未下载 2.5 GB 离线原图包（`python scripts/download_dataset.py --web-packs`），可通过
    `--remote-fallback https://xing.hanhuo.org` 按需回源拉取缩略图，或纯离线识别（仅不显示代表原图）。

用法：
  python web/serve_local.py              # 默认监听 0.0.0.0:8788
  python web/serve_local.py 8080         # 指定端口
  python web/serve_local.py --packs-dir /path/to/web_packs
"""
from __future__ import annotations

import argparse
import gzip
import http.server
import json
import socket
import socketserver
import struct
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

WEB_DIR = Path(__file__).resolve().parent
DIST_DIR = WEB_DIR / "dist"
ASSETS_DIR = DIST_DIR / "assets"
MIME_BY_ID = ["image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"]


class LocalResources:
    def __init__(self, packs_root: Path, remote_fallback: str = ""):
        self.packs_v2 = packs_root / "packs_v2"
        if not self.packs_v2.exists() and (packs_root / "r2_packs_v2").exists():
            self.packs_v2 = packs_root / "r2_packs_v2"
        self.stele_pages = packs_root / "stele_pages"
        if not self.stele_pages.exists() and (packs_root / "r2_stele_pages").exists():
            self.stele_pages = packs_root / "r2_stele_pages"
        self.remote_fallback = remote_fallback.rstrip("/")

        gtoc_p = ASSETS_DIR / "glyph_toc.bin"
        stoc_p = ASSETS_DIR / "stele_page_toc.bin"
        sctx_p = ASSETS_DIR / "stele_context.json.gz"
        self.glyph_toc = gtoc_p.read_bytes() if gtoc_p.exists() else b""
        self.stele_toc = stoc_p.read_bytes() if stoc_p.exists() else b""
        self.stele_ctx = json.loads(gzip.decompress(sctx_p.read_bytes())) if sctx_p.exists() else None

    def read_glyph(self, gid: int) -> tuple[bytes, str] | None:
        off = gid * 12
        if off < 0 or off + 12 > len(self.glyph_toc):
            return None
        pack_id, mime_id, _pad, byte_off, length = struct.unpack_from("<HBBII", self.glyph_toc, off)
        pack_path = self.packs_v2 / f"pack_{pack_id:03d}.bin"
        if not pack_path.exists():
            return None
        with open(pack_path, "rb") as f:
            f.seek(byte_off)
            data = f.read(length)
        mime = MIME_BY_ID[mime_id] if mime_id < len(MIME_BY_ID) else "image/jpeg"
        return data, mime

    def read_stele_page(self, pid: int) -> tuple[bytes, str] | None:
        off = pid * 12
        if off < 0 or off + 12 > len(self.stele_toc):
            return None
        pack_id, mime_id, _pad, byte_off, length = struct.unpack_from("<HBBII", self.stele_toc, off)
        pack_path = self.stele_pages / f"pack_{pack_id:03d}.bin"
        if not pack_path.exists():
            return None
        with open(pack_path, "rb") as f:
            f.seek(byte_off)
            data = f.read(length)
        mime = MIME_BY_ID[mime_id] if mime_id < len(MIME_BY_ID) else "image/jpeg"
        return data, mime

    def read_glyph_context(self, gid: int) -> dict:
        ctx = self.stele_ctx
        if not ctx:
            return {"ok": True, "id": gid, "hasPage": False}
        idx = gid - int(ctx["baseId"])
        if idx < 0 or idx >= int(ctx["count"]):
            return {"ok": True, "id": gid, "hasPage": False}
        tid = ctx["titleIds"][idx]
        pid = ctx["pageIds"][idx]
        out = {
            "ok": True,
            "id": gid,
            "hasPage": pid >= 0,
            "title": ctx["titles"][tid] if 0 <= tid < len(ctx["titles"]) else "",
            "rawChar": ctx["rawChars"][idx],
            "ctx": ctx["ctxs"][idx],
        }
        if pid >= 0 and pid < len(ctx["pageDims"]):
            b0, b1, b2, b3 = ctx["boxes"][idx * 4 : idx * 4 + 4]
            out["pageId"] = pid
            out["pageUrl"] = f"/stele-page/{pid}"
            out["pageDims"] = ctx["pageDims"][pid]
            out["box"] = [round(b0 / 10000.0, 4), round(b1 / 10000.0, 4), round(b2 / 10000.0, 4), round(b3 / 10000.0, 4)]
        return out


def make_handler(res: LocalResources):
    class LocalHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(DIST_DIR), **kwargs)

        def end_headers(self):
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def guess_type(self, path):
            if path.endswith(".mjs"):
                return "text/javascript; charset=utf-8"
            if path.endswith(".wasm"):
                return "application/wasm"
            if path.endswith(".gzbin") or path.endswith(".gz"):
                return "application/octet-stream"
            return super().guess_type(path)

        def _send_bytes(self, data: bytes, ctype: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _try_remote(self, path: str) -> bool:
            if not res.remote_fallback:
                return False
            try:
                req = urllib.request.Request(
                    f"{res.remote_fallback}{path}",
                    headers={"User-Agent": "HanfireShiziLocal/1.0"},
                )
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = r.read()
                    ct = r.headers.get("Content-Type", "application/octet-stream")
                self._send_bytes(data, ct, 200)
                return True
            except Exception:
                return False

        def do_GET(self):
            p = urlparse(self.path).path
            if p.startswith("/glyph/"):
                try:
                    gid = int(p[7:])
                    hit = res.read_glyph(gid)
                    if hit:
                        self._send_bytes(hit[0], hit[1], 200)
                        return
                except Exception:
                    pass
                if self._try_remote(p):
                    return
                self.send_response(404)
                self.end_headers()
                return

            if p.startswith("/stele-page/"):
                try:
                    pid = int(p[12:])
                    hit = res.read_stele_page(pid)
                    if hit:
                        self._send_bytes(hit[0], hit[1], 200)
                        return
                except Exception:
                    pass
                if self._try_remote(p):
                    return
                self.send_response(404)
                self.end_headers()
                return

            if p.startswith("/api/glyph-context/"):
                try:
                    gid = int(p[19:])
                    body = json.dumps(res.read_glyph_context(gid), ensure_ascii=False).encode("utf-8")
                    self._send_bytes(body, "application/json; charset=utf-8", 200)
                    return
                except Exception:
                    self.send_response(404)
                    self.end_headers()
                    return

            if p in ("/api/community/vectors", "/api/feedback/public", "/api/help"):
                self._send_bytes(b'{"ok":true,"count":0,"total":0,"items":[]}', "application/json; charset=utf-8", 200)
                return

            self.path = p
            return super().do_GET()

    return LocalHandler


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description="汉火识字本地离线 Web 服务器")
    ap.add_argument("port", nargs="?", type=int, default=8788, help="监听端口 (默认 8788)")
    ap.add_argument("--host", default="0.0.0.0", help="绑定地址 (默认 0.0.0.0)")
    ap.add_argument("--packs-dir", type=Path, default=WEB_DIR / "packs", help="离线原图/整页包目录 (默认 web/packs)")
    ap.add_argument("--remote-fallback", default="", help="未下载离线原图包时的可选回源地址 (如 https://xing.hanhuo.org)")
    args = ap.parse_args()

    manifest = DIST_DIR / "models_chunks" / "manifest.json"
    if not manifest.exists():
        print("[!] 未找到 web/dist/models_chunks/manifest.json！", file=sys.stderr)
        print("    请先运行: python scripts/download_weights.py", file=sys.stderr)
        sys.exit(1)

    res = LocalResources(args.packs_dir, remote_fallback=args.remote_fallback)
    n_glyph_packs = len(list(res.packs_v2.glob("pack_*.bin"))) if res.packs_v2.exists() else 0
    n_stele_packs = len(list(res.stele_pages.glob("pack_*.bin"))) if res.stele_pages.exists() else 0

    srv = ThreadingHTTPServer((args.host, args.port), make_handler(res))
    lan_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print(f"Hanfire Shizi (汉火识字) Local Server listening on {args.host}:{args.port}", flush=True)
    print(f"  Local : http://127.0.0.1:{args.port}/", flush=True)
    print(f"  LAN   : http://{lan_ip}:{args.port}/", flush=True)
    print(f"  Packs : {n_glyph_packs} glyph packs, {n_stele_packs} stele-page packs (dir={args.packs_dir})", flush=True)
    if n_glyph_packs == 0 and not args.remote_fallback:
        print("  Tip   : 如需离线查看 21.7 万原图与整页拓片，运行 python scripts/download_dataset.py --web-packs", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
