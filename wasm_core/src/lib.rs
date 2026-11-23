//! 汉火字形 (xing.hanhuo.org) 浏览器端 WASM 核心：
//! 1. 100% 复刻原 Rust 服务端 preprocess.rs（描漆彩色分割 / 自适应二值化 / 去边框 / 四点透视拉正 / 224 归一化）
//! 2. 16.7 万 × 512 维 INT8 原型向量库极速 Top-K 余弦检索（整数累加，无溢出，浏览器端 ~30-60ms）

mod preprocess;

use base64::Engine as _;
use image::{DynamicImage, GrayImage, ImageBuffer, Rgb};
use serde::Serialize;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use wasm_bindgen::prelude::*;

fn rgb_to_png_data_url(img: &DynamicImage, max_side: u32) -> Result<String, JsValue> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let resized = if w > max_side || h > max_side {
        let scale = (max_side as f32 / w as f32).min(max_side as f32 / h as f32);
        let nw = ((w as f32 * scale).round() as u32).max(1);
        let nh = ((h as f32 * scale).round() as u32).max(1);
        image::imageops::resize(&rgb, nw, nh, image::imageops::FilterType::Lanczos3)
    } else {
        rgb
    };
    let mut png = Vec::with_capacity(16384);
    DynamicImage::ImageRgb8(resized)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&png)
    ))
}

/// 将浏览器 Canvas/ImageBitmap 解码得到的 RGBA 像素合成为白底 RGB（透明区不发黑）
fn rgba_to_white_rgb(rgba: &[u8], w: u32, h: u32) -> Result<DynamicImage, JsValue> {
    let expected = (w as usize) * (h as usize) * 4;
    if rgba.len() != expected || w == 0 || h == 0 {
        return Err(JsValue::from_str("RGBA 缓冲尺寸不匹配"));
    }
    let out = ImageBuffer::from_fn(w, h, |x, y| {
        let idx = ((y * w + x) * 4) as usize;
        let a = rgba[idx + 3] as u32;
        if a == 255 {
            Rgb([rgba[idx], rgba[idx + 1], rgba[idx + 2]])
        } else {
            let mix = |c: u8| ((c as u32 * a + 255 * (255 - a)) / 255) as u8;
            Rgb([mix(rgba[idx]), mix(rgba[idx + 1]), mix(rgba[idx + 2])])
        }
    });
    Ok(DynamicImage::ImageRgb8(out))
}

fn gray_to_png_data_url(canvas: &GrayImage) -> Result<String, JsValue> {
    let mut png = Vec::with_capacity(8192);
    DynamicImage::ImageLuma8(canvas.clone())
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&png)
    ))
}

#[derive(Serialize)]
struct DetectOut {
    quad: [[f32; 2]; 4],
    preview: String,
    warning: String,
    multi_hint: bool,
    est_chars: u32,
}

/// 自动检测识别框 + 返回预处理 256×256 预览图（上传照片即刻本地完成，0 网络耗时）
#[wasm_bindgen]
pub fn detect_from_rgba(rgba: &[u8], w: u32, h: u32) -> Result<String, JsValue> {
    let img = rgba_to_white_rgb(rgba, w, h)?;
    let (canvas, warning, quad, multi_hint, est_chars) = preprocess::extract_glyph_canvas_with_multi_hint(&img);
    let preview = gray_to_png_data_url(&canvas)?;
    let out = DetectOut {
        quad: [
            [quad[0].0, quad[0].1],
            [quad[1].0, quad[1].1],
            [quad[2].0, quad[2].1],
            [quad[3].0, quad[3].1],
        ],
        preview,
        warning,
        multi_hint,
        est_chars,
    };
    serde_json::to_string(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub struct PreparedInput {
    tensor: Vec<f32>,
    tensor_alt: Vec<f32>,
    aspect_ratio: f32,
    preview: String,
    cropped_png: String,
    warning: String,
}

#[wasm_bindgen]
impl PreparedInput {
    #[wasm_bindgen(getter)]
    pub fn tensor(&self) -> Vec<f32> {
        self.tensor.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn tensor_alt(&self) -> Vec<f32> {
        self.tensor_alt.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn aspect_ratio(&self) -> f32 {
        self.aspect_ratio
    }
    #[wasm_bindgen(getter)]
    pub fn preview(&self) -> String {
        self.preview.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn cropped_png(&self) -> String {
        self.cropped_png.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn warning(&self) -> String {
        self.warning.clone()
    }
}

fn canvas_to_tensor(canvas: &GrayImage) -> Vec<f32> {
    let off = (256u32 - 224u32) / 2;
    let mut tensor = vec![0f32; 3 * 224 * 224];
    for y in 0..224u32 {
        for x in 0..224u32 {
            let p = canvas.get_pixel(x + off, y + off)[0];
            let v = (p as f32 / 255.0 - 0.5) / 0.5;
            let idx = (y as usize) * 224 + (x as usize);
            tensor[idx] = v;
            tensor[224 * 224 + idx] = v;
            tensor[2 * 224 * 224 + idx] = v;
        }
    }
    tensor
}

#[derive(Serialize)]
struct CropPreviewOut {
    preview: String,
    cropped_png: String,
    warning: String,
}

/// 仅执行透视拉正与黑白预览生成（不跑模型，用于用户点「确定裁剪」或发「帮看」时实时更新）
#[wasm_bindgen]
pub fn preview_crop_from_rgba(
    rgba: &[u8],
    w: u32,
    h: u32,
    quad_flat: &[f32],
) -> Result<String, JsValue> {
    let img = rgba_to_white_rgb(rgba, w, h)?;
    let user_cropped = quad_flat.len() == 8;
    let img = if user_cropped {
        let q = [
            (quad_flat[0], quad_flat[1]),
            (quad_flat[2], quad_flat[3]),
            (quad_flat[4], quad_flat[5]),
            (quad_flat[6], quad_flat[7]),
        ];
        match preprocess::perspective_crop(&img, q) {
            Some(cropped) => cropped,
            None => img,
        }
    } else {
        img
    };
    let (canvas, warning, _quad) = if user_cropped {
        preprocess::extract_glyph_canvas_user_crop(&img)
    } else {
        preprocess::extract_glyph_canvas(&img)
    };
    let preview = gray_to_png_data_url(&canvas)?;
    let cropped_png = rgb_to_png_data_url(&img, 512)?;
    let out = CropPreviewOut { preview, cropped_png, warning };
    serde_json::to_string(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// 多字模式区域透视拉正：按 4 角点把整碑/目标区域拉正裁切为高清彩色 PNG（最大边 1400px）
#[wasm_bindgen]
pub fn crop_quad_to_png(
    rgba: &[u8],
    w: u32,
    h: u32,
    quad_flat: &[f32],
) -> Result<String, JsValue> {
    let img = rgba_to_white_rgb(rgba, w, h)?;
    if quad_flat.len() != 8 {
        return Err(JsValue::from_str("quad 需要 8 个坐标值"));
    }
    let q = [
        (quad_flat[0], quad_flat[1]),
        (quad_flat[2], quad_flat[3]),
        (quad_flat[4], quad_flat[5]),
        (quad_flat[6], quad_flat[7]),
    ];
    let cropped = preprocess::perspective_crop(&img, q).unwrap_or(img);
    rgb_to_png_data_url(&cropped, 1400)
}

/// 按用户微调后的四角坐标做透视拉正 + 字形二值化归一化，一次返回 (3×224×224 f32 张量 + 黑白预览 + 透视拉正彩色裁剪图 + 警告)
#[wasm_bindgen]
pub fn prepare_input_from_rgba(
    rgba: &[u8],
    w: u32,
    h: u32,
    quad_flat: &[f32],
) -> Result<PreparedInput, JsValue> {
    let img = rgba_to_white_rgb(rgba, w, h)?;
    let user_cropped = quad_flat.len() == 8;
    let img = if user_cropped {
        let q = [
            (quad_flat[0], quad_flat[1]),
            (quad_flat[2], quad_flat[3]),
            (quad_flat[4], quad_flat[5]),
            (quad_flat[6], quad_flat[7]),
        ];
        match preprocess::perspective_crop(&img, q) {
            Some(cropped) => cropped,
            None => img,
        }
    } else {
        img
    };

    let (canvas, canvas_alt, aspect_ratio, warning, _quad) = if user_cropped {
        preprocess::extract_glyph_canvases_user_crop(&img)
    } else {
        preprocess::extract_glyph_canvases(&img)
    };
    let preview = gray_to_png_data_url(&canvas)?;
    let cropped_png = rgb_to_png_data_url(&img, 512)?;

    let tensor = canvas_to_tensor(&canvas);
    let tensor_alt = match canvas_alt {
        Some(ref c) => canvas_to_tensor(c),
        None => Vec::new(),
    };

    Ok(PreparedInput { tensor, tensor_alt, aspect_ratio, preview, cropped_png, warning })
}

/// 多字模式专用：按紧致矩形框 [x0, y0, x1, y1] 裁切单字并保留全部偏旁部件，输出 224 张量 + 黑白预览 + 彩色裁剪图
#[wasm_bindgen]
pub fn prepare_char_box_from_rgba(
    rgba: &[u8],
    w: u32,
    h: u32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) -> Result<PreparedInput, JsValue> {
    let img = rgba_to_white_rgb(rgba, w, h)?;
    let ix0 = (x0.floor().max(0.0) as u32).min(w.saturating_sub(1));
    let iy0 = (y0.floor().max(0.0) as u32).min(h.saturating_sub(1));
    let ix1 = (x1.ceil().min(w as f32) as u32).max(ix0 + 1).min(w);
    let iy1 = (y1.ceil().min(h as f32) as u32).max(iy0 + 1).min(h);
    let sub = img.crop_imm(ix0, iy0, ix1 - ix0, iy1 - iy0);

    // 用字框周围外扩 22% 的环形上下文估算底色亮度（避开字框紧贴笔画边缘误翻极性，如「九」「口」）
    let bw = ix1 - ix0;
    let bh = iy1 - iy0;
    let mx = ((bw as f32 * 0.22).round() as u32).clamp(4, 28);
    let my = ((bh as f32 * 0.22).round() as u32).clamp(4, 28);
    let rx0 = ix0.saturating_sub(mx);
    let ry0 = iy0.saturating_sub(my);
    let rx1 = (ix1 + mx).min(w);
    let ry1 = (iy1 + my).min(h);
    let full_rgb = img.to_rgb8();
    let mut ring_lumas: Vec<u8> = Vec::with_capacity(256);
    for x in (rx0..rx1).step_by(2) {
        for &y in &[ry0, ry1.saturating_sub(1)] {
            let p = full_rgb.get_pixel(x, y);
            ring_lumas.push(((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29) >> 8) as u8);
        }
    }
    for y in (ry0..ry1).step_by(2) {
        for &x in &[rx0, rx1.saturating_sub(1)] {
            let p = full_rgb.get_pixel(x, y);
            ring_lumas.push(((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29) >> 8) as u8);
        }
    }
    let bg_luma_hint = if !ring_lumas.is_empty() {
        ring_lumas.sort_unstable();
        ring_lumas[ring_lumas.len() / 2] as f32
    } else {
        -1.0
    };

    let (canvas, canvas_alt, aspect_ratio) = preprocess::extract_tight_char_box_canvases(&sub, bg_luma_hint);
    let preview = gray_to_png_data_url(&canvas)?;
    let cropped_png = rgb_to_png_data_url(&sub, 384)?;

    let tensor = canvas_to_tensor(&canvas);
    let tensor_alt = match canvas_alt {
        Some(ref c) => canvas_to_tensor(c),
        None => Vec::new(),
    };

    Ok(PreparedInput {
        tensor,
        tensor_alt,
        aspect_ratio,
        preview,
        cropped_png,
        warning: String::new(),
    })
}

#[derive(Copy, Clone)]
struct ScoredIdx {
    score: f32,
    idx: u32,
}

impl PartialEq for ScoredIdx {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.idx == other.idx
    }
}
impl Eq for ScoredIdx {}
impl PartialOrd for ScoredIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
// 最小堆（堆顶为当前 top-k 中最小分，方便 O(1) 剪枝淘汰）
impl Ord for ScoredIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        other.score.total_cmp(&self.score).then_with(|| self.idx.cmp(&other.idx))
    }
}

#[derive(Serialize)]
struct SearchHit {
    id: u32,
    score: f32,
}

/// 常驻内存的 INT8 原型库检索器（避免每次检索跨 JS/WASM 边界拷贝 86 MB！）
#[wasm_bindgen]
pub struct ProtoIndex {
    n: usize,
    dim: usize,
    inv_norms: Vec<f32>,
    vecs_i8: Vec<i8>,
    style_ids: Vec<u8>,
}

#[wasm_bindgen]
impl ProtoIndex {
    /// 从 HXV1 二进制（OPFS 整包直读）+ style_ids 初始化常驻索引
    #[wasm_bindgen(constructor)]
    pub fn new(vec_bin: &[u8], style_ids: &[u8]) -> Result<ProtoIndex, JsValue> {
        if vec_bin.len() < 16 || &vec_bin[0..4] != b"HXV1" {
            return Err(JsValue::from_str("proto_vec.bin 格式校验失败（非 HXV1）"));
        }
        let n = u32::from_le_bytes(vec_bin[4..8].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(vec_bin[8..12].try_into().unwrap()) as usize;
        let inv_bytes = n * 4;
        let vec_bytes = n * dim;
        if vec_bin.len() != 16 + inv_bytes + vec_bytes {
            return Err(JsValue::from_str("proto_vec.bin 长度不匹配"));
        }
        if style_ids.len() != n {
            return Err(JsValue::from_str("style_ids 长度与向量条数不一致"));
        }
        let mut inv_norms = Vec::with_capacity(n);
        let inv_slice = &vec_bin[16..16 + inv_bytes];
        for chunk in inv_slice.chunks_exact(4) {
            inv_norms.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let raw_u8 = &vec_bin[16 + inv_bytes..];
        let vecs_i8: Vec<i8> = raw_u8.iter().map(|&b| b as i8).collect();
        Ok(ProtoIndex {
            n,
            dim,
            inv_norms,
            vecs_i8,
            style_ids: style_ids.to_vec(),
        })
    }

    pub fn count(&self) -> usize {
        self.n
    }

    /// 极速 INT8 暴力余弦 Top-K 检索（filter_style < 0 表示不限书体）
    pub fn search(&self, query_f32: &[f32], filter_style: i32, topk: usize) -> Result<String, JsValue> {
        let mask = if filter_style >= 0 && (filter_style as u32) < 32 {
            1u32 << (filter_style as u32)
        } else {
            0u32
        };
        self.search_mask(query_f32, mask, topk)
    }

    /// 按书体位掩码（style_mask == 0 表示不限书体）执行 INT8 暴力余弦 Top-K 检索，支持书体大类（如篆/古文字/楷行草）精确过滤
    pub fn search_mask(&self, query_f32: &[f32], style_mask: u32, topk: usize) -> Result<String, JsValue> {
        if query_f32.len() != self.dim {
            return Err(JsValue::from_str("查询向量维度不匹配"));
        }
        let k = topk.clamp(1, 200);
        // 单位化查询向量并缩放至 i32 定点（scale = 4096 / max_abs，512*4096*127 = 2.66e8 < i32::MAX，零溢出、误差 < 1e-4）
        let mut qnorm2 = 0.0f32;
        for &v in query_f32 {
            qnorm2 += v * v;
        }
        let inv_qn = 1.0f32 / qnorm2.sqrt().max(1e-8);
        let mut q_unit = [0.0f32; 512];
        let mut q_max = 1e-8f32;
        for i in 0..self.dim {
            let u = query_f32[i] * inv_qn;
            q_unit[i] = u;
            let a = u.abs();
            if a > q_max {
                q_max = a;
            }
        }
        let q_scale = 4096.0f32 / q_max;
        let inv_q_scale = q_max / 4096.0f32;
        let mut q_i32 = [0i32; 512];
        for i in 0..self.dim {
            q_i32[i] = (q_unit[i] * q_scale).round() as i32;
        }

        let mut heap: BinaryHeap<ScoredIdx> = BinaryHeap::with_capacity(k + 1);
        let dim = self.dim;

        for i in 0..self.n {
            if style_mask != 0 {
                let sid = self.style_ids[i] as u32;
                if sid >= 32 || (style_mask & (1u32 << sid)) == 0 {
                    continue;
                }
            }
            let row = &self.vecs_i8[i * dim..(i + 1) * dim];
            // 8 路展开整数点积，LLVM 在 wasm32 下可高效流水线化/自动向量化
            let mut acc0 = 0i32;
            let mut acc1 = 0i32;
            let mut acc2 = 0i32;
            let mut acc3 = 0i32;
            let mut j = 0usize;
            while j < dim {
                acc0 += q_i32[j] * (row[j] as i32) + q_i32[j + 1] * (row[j + 1] as i32);
                acc1 += q_i32[j + 2] * (row[j + 2] as i32) + q_i32[j + 3] * (row[j + 3] as i32);
                acc2 += q_i32[j + 4] * (row[j + 4] as i32) + q_i32[j + 5] * (row[j + 5] as i32);
                acc3 += q_i32[j + 6] * (row[j + 6] as i32) + q_i32[j + 7] * (row[j + 7] as i32);
                j += 8;
            }
            let dot = (acc0 + acc1 + acc2 + acc3) as f32;
            let score = dot * inv_q_scale * self.inv_norms[i];
            if heap.len() < k {
                heap.push(ScoredIdx { score, idx: i as u32 });
            } else if let Some(&min_top) = heap.peek() {
                if score > min_top.score {
                    heap.pop();
                    heap.push(ScoredIdx { score, idx: i as u32 });
                }
            }
        }

        let mut results: Vec<ScoredIdx> = heap.into_vec();
        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        let hits: Vec<SearchHit> = results
            .into_iter()
            .map(|r| SearchHit { id: r.idx, score: r.score })
            .collect();
        serde_json::to_string(&hits).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
