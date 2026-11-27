#!/usr/bin/env bash
# 一键将 release_artifacts/ 中的 Clean V12 权重与 Web 模型包发布到 GitHub Release
# 用法: bash scripts/publish_release.sh [repo] [tag]
set -euo pipefail

REPO="${1:-hanhuo-org/shizi}"
TAG="${2:-v12}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/release_artifacts"

if [ ! -f "$DIR/shizi-v12-web-models.tar.gz" ]; then
  echo "未找到 $DIR/shizi-v12-web-models.tar.gz，请先打包 release_artifacts！" >&2
  exit 1
fi

echo "Publishing GitHub Release $TAG to $REPO ..."
gh release create "$TAG" \
  "$DIR/shizi-v12-web-models.tar.gz" \
  "$DIR/convnext_tiny_v12_best.pt" \
  "$DIR/convnext_tiny_v12_op17_wo_u8.onnx" \
  "$DIR/char_det_best.pt" \
  "$DIR/char_det_u8.onnx" \
  "$DIR/classes_meta.tar.gz" \
  "$DIR/SHA256SUMS.txt" \
  --repo "$REPO" \
  --title "Hanfire Shizi Clean V12 (ConvNeXt-Tiny, 222,317 Prototypes)" \
  --notes "Clean V12 Upgrade: 88.6M ConvNeXt-Base Super-Teacher (Sub-center ArcFace K=3 + 1,217 IDS components) distilled into 28.6M ConvNeXt-Tiny (27.5 MB opset-17 WO-U8 ONNX, min_cos=0.99956) with 222,317 denoised HXV1 INT8 prototypes & ABTT k=6. Zero-leakage highlights: HanYin 82.80%/93.55%, 346-Hard 70.81%/89.60%, Jinwen 70.18%/85.09%, CHUBS 66.86%/81.52%, OBIMD 61.52%/80.06%. Live demo: https://xing.hanhuo.org/"
