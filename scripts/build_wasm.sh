#!/usr/bin/env bash
# 一键编译 Rust WASM 预处理与 INT8 HXV1 向量扫描内核 -> web/dist/vendor/hanzi_wasm/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/wasm_core"
wasm-pack build --target web --release --out-dir "$ROOT/web/dist/vendor/hanzi_wasm"
rm -f "$ROOT/web/dist/vendor/hanzi_wasm/.gitignore"
echo "WASM core built cleanly into $ROOT/web/dist/vendor/hanzi_wasm"
