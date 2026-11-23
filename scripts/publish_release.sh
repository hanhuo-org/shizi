#!/usr/bin/env bash
# 一键将 release_artifacts/ 中的 Clean V10 权重与 Web 模型包发布到 GitHub Release
# 用法: bash scripts/publish_release.sh [repo] [tag]
set -euo pipefail

REPO="${1:-hanhuo-org/shizi}"
TAG="${2:-v10}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/release_artifacts"

if [ ! -f "$DIR/shizi-v10-web-models.tar.gz" ]; then
  echo "未找到 $DIR/shizi-v10-web-models.tar.gz，请先打包 release_artifacts！" >&2
  exit 1
fi

echo "Publishing GitHub Release $TAG to $REPO ..."
gh release create "$TAG" \
  "$DIR/shizi-v10-web-models.tar.gz" \
  "$DIR/wiseft_v10_a070.pt" \
  "$DIR/wiseft_v10_a070_op17_wo_u8.onnx" \
  "$DIR/char_det_best.pt" \
  "$DIR/char_det_u8.onnx" \
  "$DIR/classes_meta.tar.gz" \
  "$DIR/SHA256SUMS.txt" \
  --repo "$REPO" \
  --title "Hanfire Shizi Clean V10 (216,988 Prototypes)" \
  --notes "Clean V10 Pre-Qin & Calligraphy Release: 216,988 multi-style prototypes covering Oracle Bone, Western Zhou Bronzes, Warring States Chu Slips, Stone Drums, Han Seals, and Ming-Tang-Jin Calligraphy. Companion dataset: https://huggingface.co/datasets/hanhuo-org/hanhuo-shizi-dataset"
