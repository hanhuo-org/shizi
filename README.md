# 汉火识字 (Hanfire Shizi — Clean V10)

> **浏览器端 100% 本地离线运行的古文字、甲骨金文、战国楚简、汉代玺印与历代碑帖书法视觉检索引擎**（线上演示：[xing.hanhuo.org](https://xing.hanhuo.org/)）。

- **全量覆盖**：`12,368` 个汉字 × `18` 种书体 × **`216,988` 个多原型 INT8 向量** + `5,651` 张高清整页碑帖/甲骨/青铜器/石鼓文/汉印拓片红框溯源。
- **核心架构**：
  - 视觉编码器：`ResNet50 + ArcFace + WiSE-FT`（opset-17 WO-U8 ONNX，`26.7 MB`，PyTorch↔ONNX 余弦一致性 `>0.997`）
  - 多字检测与区域切字：`MobileNetV3-Small CenterNet`（`3.7 MB`，支持整页自动检测、手绘区域局部精切、暗底拓片双极性增强）
  - 浏览器加速内核：`Rust-WASM`（自适应二值化、去边框、倾斜拉正、HXV1 INT8 21.7 万原型极速扫描 + 混合 TTA `0.65·bw + 0.35·inv_gray` + 古籍二元语言模型重排）

---

## 一、零泄漏实测成绩（Clean V10，精确 HXV1 INT8）

| 零泄漏真实评测基准 | 纯二值 ALL (T1 / T5) | 线上混合 TTA (T1 / T5) | 专属书体 T5 |
| :--- | :---: | :---: | :---: |
| **战国楚简 `CHUBS_Real_Test`**（1,050 张未见简片） | `54.48% / 70.86%` | **`55.52% / 72.57%`** | `66.29%` (楚文字) |
| **汉代玺印 `HanYin_Real_Test`**（93 张，0/42 重叠字） | `66.67% / 88.17%` | **`68.82% / 88.17%`** | — |
| **西周青铜器 `Jinwen_Real_Test`**（114 张：虢季子白盘+何尊） | `57.02% / 78.07%` | **`62.28% / 82.46%`** | `79.82%` (金文) |
| **殷墟甲骨文 `OBIMD_Real_Test`**（998 张未见甲骨片） | `53.41% / 75.25%` | **`55.31% / 77.45%`** | `76.85%` (甲骨文) |
| **名碑帖疑难字 `346-Hard`** | **`66.18% / 86.13%`** | — | — |
| **《兰亭序》`L24` / `Gen(500)` / `Evo(500)`** | **`24/24 · 94.2% · 98.6%`** | — | — |

---

## 二、快速启动本地 Web 识字（2 步即用）

无需 Cloudflare / R2 / GPU，纯 Python 标准库即可启动支持多线程 WASM 的本地离线 Web 服务：

```bash
git clone https://github.com/hanhuo-org/shizi.git
cd shizi

# 1. 下载 Clean V10 Web 运行时模型包（约 136 MB，来自 GitHub Release）
python scripts/download_weights.py
# 若本地已有 release_artifacts/ 打包目录，可直接离线安装：
# python scripts/download_weights.py --local-dir release_artifacts

# 2. （可选）从 Hugging Face 下载 2.5 GB 离线原图与整页拓片包（用于离线浏览 21.7 万原图与红框溯源）
pip install huggingface_hub
python scripts/download_dataset.py --web-packs

# 3. 启动本地 Web 服务（自带 COOP/COEP 头，支持单字/多字/框选切字/缩放/撤销/导出对照图）
python web/serve_local.py
# 浏览器打开 http://127.0.0.1:8788/
# （若未下载 2.5 GB 原图包，也可附加 --remote-fallback https://xing.hanhuo.org 按需查看线上原图）
```

---

## 三、完整训练、微调与重新导出 Web 包

### 1. 安装 Python 训练依赖

```bash
cd shizi
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 按需下载权重与 Hugging Face 数据集

数据集已按模块托管至 Hugging Face：**[`hanhuo-org/hanhuo-shizi-dataset`](https://huggingface.co/datasets/hanhuo-org/hanhuo-shizi-dataset)**：

```bash
# 下载完整 PyTorch/ONNX 权重 + 词表（约 393 MB）
python scripts/download_weights.py --all

# 按需拉取数据模块：
python scripts/download_dataset.py --real      # ~2.3 GB：真迹微调 + 12 项零泄漏评测集（推荐快速微调）
python scripts/download_dataset.py --base      # ~15 GB ：123.5 万清洗多书体基础字库（从零训练骨干用）
python scripts/download_dataset.py --detector  # ~237 MB：CenterNet 多字检测器字池
# 或一步全部拉取：
# python scripts/download_dataset.py --all
```

### 3. 运行评测 / 域自适应微调 / 从零训练 / 导出浏览器模型

```bash
# A. 运行零泄漏真实文物评测（验证环境与基准分数）
python -m training.evaluate

# B. 基于 Clean V10 权重执行多流域自适应微调 + WiSE-FT 融合（如加入你自己的新碑帖/简帛数据）
python -m training.finetune \
  --base-ckpt checkpoints/wiseft_v10_a070.pt \
  --steps 3000 --alpha 0.70

# C. （可选）从零训练 ResNet50 + ArcFace 基础主干（需 --base 数据）
python -m training.train_backbone --epochs 15 --batch-size 256

# D. （可选）重新训练 CenterNet 多字框检测器（需 --detector 字池）
python -m training.train_detector

# E. 将微调后的 .pt 导出为 opset-17 WO-U8 ONNX 并重算全部 21.7 万 HXV1 INT8 原型包，直接写入 web/dist/
python -m training.export_web --ckpt work/finetune/wiseft_a070.pt
# 导出完成后直接运行 `python web/serve_local.py` 即可在浏览器体验你的新模型！
```

### 4. （可选）重新编译 Rust-WASM 内核

```bash
bash scripts/build_wasm.sh
```

---

## 四、目录结构

```text
shizi/
├── README.md
├── pyproject.toml / requirements.txt
├── scripts/
│   ├── download_weights.py      # 从 GitHub Release 拉取并校验模型权重
│   ├── download_dataset.py      # 从 HF (hanhuo-org/hanhuo-shizi-dataset) 按需拉取模块化数据集
│   ├── publish_release.sh       # 一键通过 gh release create 发布权重到 GitHub Release
│   └── build_wasm.sh            # 一键 wasm-pack 编译 Rust-WASM 内核
├── training/
│   ├── model.py                 # GlyphModel (ResNet50+ArcFace) 与拓片/裂纹/石花领域增强
│   ├── dataset.py               # 相对路径数据加载器与 6 流平衡对比采样器
│   ├── train_backbone.py        # 基础主干从零训练
│   ├── finetune.py              # 多流域自适应微调 + WiSE-FT 权重融合
│   ├── train_detector.py        # CenterNet (MobileNetV3-Small) 多字检测器训练
│   ├── export_web.py            # opset-17 WO-U8 ONNX + HXV1 INT8 向量分片导出
│   └── evaluate.py              # 零泄漏真实文物评测套件
├── wasm_core/                   # Rust-WASM 图像预处理与 INT8 向量扫描内核源码
└── web/
    ├── serve_local.py           # 零依赖本地离线 Web 服务器（含原图与整页红框溯源）
    ├── cf_worker.mjs            # 可选：Cloudflare Workers + R2 + D1 生产部署入口
    ├── wrangler.example.toml    # 脱敏 Wrangler 配置模板
    └── dist/                    # 浏览器端前端界面、Web Worker 与推理引擎
```

---

## 五、发布权重到 GitHub Release

所有待发布产物已预先打包并生成 `SHA256SUMS.txt` 位于 `release_artifacts/`（`.gitignore` 已忽略，不污染 Git 历史）。当你建好远程仓库并 `git push` 后，一行命令即可发布 Release：

```bash
bash scripts/publish_release.sh hanhuo-org/shizi v10
```
