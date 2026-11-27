# 汉火识字 (Hanfire Shizi — Clean V12)

> **浏览器端 100% 本地离线运行的古文字、甲骨金文、战国楚简、汉代玺印与历代碑帖书法视觉检索引擎**（线上演示：[xing.hanhuo.org](https://xing.hanhuo.org/)）。

- **全量覆盖**：`12,368` 个汉字 × `18` 种书体 × **`222,317` 个多原型 INT8 向量** + 高清整页碑帖/甲骨/青铜器/石鼓文/汉印拓片红框溯源。
- **核心原理（像“人脸识别”而不是传统 OCR）**：
  - 汉火识字**不是**传统分类式文字识别（OCR），而是类似**人脸识别**的 1:N 视觉检索：把每个古汉字当作一张“脸”提取 512 维特征向量，再去 22.23 万历代字形库里按相似度找最像的字与原碑拓片。
  - **为什么不做传统 OCR？** 古字异体字上万、狂草连笔且新出土甲骨楚简不断增加，传统固定类别 OCR 遇到冷僻字或新字形就直接失效且给不出依据；检索式方案只管“像不像”，能给出多个最接近候选并直接溯源原拓，新增字形也无需推倒重训模型。
- **硬件、算法与数据取舍**：
  - **硬件**：放弃昂贵云端显卡服务器，将压缩模型与 22.23 万 INT8 向量全部放进浏览器（`Rust-WASM` + 可选 `WebGPU`），首次缓存（约 `127 MB`）后纯离线毫秒级出结果，零服务器成本、照片绝不上传。
  - **算法**：`88.6M ConvNeXt-Base` 超级教师（多中心 ArcFace + 1,217 汉字偏旁 IDS 结构监督）→ 在线蒸馏 `28.6M ConvNeXt-Tiny` 轻量学生 → `opset-17 Weight-Only U8 ONNX`（仅 **`27.5 MB` 原始 / `25.4 MB` 传输**，与 FP32 余弦一致率 `0.9999+`），配合 `ABTT (k=6)` 去偏投影与古籍二元语言模型重排；多字切分另配 `3.7 MB` MobileNetV3-Small CenterNet。
  - **数据**：`123.5 万` 清洗多书体字库打底覆盖冷僻字 + `8.4 万` 真实甲骨/金文/楚简/汉印/名碑拓片专项域适应微调 + `WiSE-FT` 权重融合，兼顾字种广度与残损拓片抗噪能力。

---

## 一、零泄漏实测成绩（Clean V12 ConvNeXt-Tiny vs Clean V11 ResNet50，精确 HXV1 INT8）

| 零泄漏真实评测基准 | Clean V11 (ResNet50) | **Clean V12 (ConvNeXt-Tiny)** | 提升 (T1 / T5) |
| :--- | :---: | :---: | :---: |
| **名碑帖疑难字 `346-Hard`** | `65.32% / 84.97%` | **`70.81% / 89.60%`** | `+5.49% / +4.63%` |
| **汉代玺印 `HanYin_Real_Test`**（93 张，0 重叠字） | `65.59% / 87.10%` | **`82.80% / 93.55%`** | `+17.21% / +6.45%` |
| **西周青铜器 `Jinwen_Real_Test`**（114 张） | `62.28% / 82.46%` | **`70.18% / 85.09%`** | `+7.90% / +2.63%` |
| **战国楚简 `CHUBS_Real_Test`**（1,050 张） | `56.76% / 73.52%` | **`66.86% / 81.52%`**（专属 T5 `75.33%`） | `+10.10% / +8.00%` |
| **殷墟甲骨文 `OBIMD_Real_Test`**（998 张） | `55.11% / 77.15%` | **`61.52% / 80.06%`** | `+6.41% / +2.91%` |
| **极难残碑 `Wave-7 Gold`**（71 张） | `8.45% / 23.94%` | **`18.31% / 49.30%`** | `+9.86% / +25.36%` |
| **实拍三字 `r3` / 兰亭序 `L24` / `Gen` / `Evo`** | `[2,1,2] · 24/24 · 87.2% · 90.0%` | **`[1,1,1] · 24/24 · 90.0% · 90.6%`** | 全面领先 |

---

## 二、快速启动本地 Web 识字（2 步即用）

无需 Cloudflare / R2 / GPU，纯 Python 标准库即可启动支持多线程 WASM 的本地离线 Web 服务：

```bash
git clone https://github.com/hanhuo-org/shizi.git
cd shizi

# 1. 下载 Clean V12 Web 运行时模型包（约 142 MB，来自 GitHub Release）
python scripts/download_weights.py
# 若本地已有 release_artifacts/ 打包目录，可直接离线安装：
# python scripts/download_weights.py --local-dir release_artifacts

# 2. （可选）从 Hugging Face 下载离线原图与整页拓片包（用于离线浏览代表原图与红框溯源）
pip install huggingface_hub
python scripts/download_dataset.py --web-packs

# 3. 启动本地 Web 服务（自带 COOP/COEP 头，支持单字/多字/框选切字/缩放/撤销/导出对照图）
python web/serve_local.py
# 浏览器打开 http://127.0.0.1:8788/
# （若未下载本地原图包，也可附加 --remote-fallback https://xing.hanhuo.org 按需查看线上原图）
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
# 下载 V12 PyTorch/ONNX 权重 + 词表 + Web 模型包（约 654 MB）
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
# A. 运行零泄漏真实文物评测（默认加载 checkpoints/convnext_tiny_v12_best.pt + ABTT k=6）
python -m training.evaluate

# B. 域自适应微调 + WiSE-FT 融合（如加入你自己的新碑帖/简帛数据）
python -m training.finetune \
  --base-ckpt checkpoints/convnext_tiny_v12_best.pt \
  --steps 3000 --alpha 0.70

# C. （可选）从零训练 V12 ConvNeXt-Base 大教师与 ConvNeXt-Tiny 学生蒸馏（需 --base 数据）
python -m training.train_v12_distill --mode teacher --epochs 14
python -m training.train_v12_distill --mode student --epochs 14

# D. （可选）重新训练 CenterNet 多字框检测器（需 --detector 字池）
python -m training.train_detector

# E. 将微调后的 .pt 导出为 opset-17 WO-U8 ONNX 并重算全部 22.23 万 HXV1 INT8 原型包，直接写入 web/dist/
python -m training.export_web --ckpt checkpoints/convnext_tiny_v12_best.pt
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
│   ├── model.py                 # 基础模型加载、WiSE-FT 融合与拓片/裂纹/石花领域增强
│   ├── model_v12.py             # V12 ConvNeXt + 多中心 ArcFace (K=3 PartialFC) + U8 ONNX 导出
│   ├── train_v12_distill.py     # V12 超级教师训练 + 特征缓存 + 学生离线余弦/RKD 蒸馏
│   ├── sync_gdrive.py           # Kaggle/Colab 训练断点后台 rclone 同步辅助
│   ├── dataset.py               # 相对路径数据加载器与 6 流平衡对比采样器
│   ├── train_backbone.py        # ResNet50 基础主干从零训练
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
bash scripts/publish_release.sh hanhuo-org/shizi v12
```
