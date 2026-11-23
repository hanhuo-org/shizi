//! 拍照预处理: 彩色描漆检测 / 自适应二值化 / 最大连通域裁剪 / 白底居中
//! 行为对齐 Python 版 glyph_common::preprocess_upload + val_tf(Resize256+CenterCrop224)

use image::{DynamicImage, GenericImage, GenericImageView, GrayImage, ImageBuffer, Luma};

/// 完整预处理: 任意照片 -> (3,224,224) f32 张量(已归一化)
pub fn photo_to_tensor(img: &DynamicImage) -> Vec<f32> {
    let (canvas, _warn, _quad) = extract_glyph_canvas(img); // 256x256 白底黑字
    let crop = center_crop_224(&canvas);    // 对齐 val_tf: Resize(256)+CenterCrop(224)
    let mut out = vec![0f32; 3 * 224 * 224];
    for (x, y, p) in crop.enumerate_pixels() {
        let v = p[0] as f32 / 255.0;
        let v = (v - 0.5) / 0.5;
        for c in 0..3 {
            out[c * 224 * 224 + y as usize * 224 + x as usize] = v;
        }
    }
    out
}

/// 四点透视校正: 把原图中 quad(左上/右上/右下/左下, 原图像素坐标) 拉正为矩形图。
/// 拍碑帖视角倾斜时, 用户拖出的梯形框由此变正。四点退化(共线)返回 None。
pub fn perspective_crop(img: &DynamicImage, quad: [(f32, f32); 4]) -> Option<DynamicImage> {
    use imageproc::geometric_transformations::Projection;
    use imageproc::geometric_transformations::{warp, Interpolation};
    let dist = |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    let (top, bot) = (dist(quad[0], quad[1]), dist(quad[3], quad[2]));
    let (lef, rig) = (dist(quad[0], quad[3]), dist(quad[1], quad[2]));
    let tw = ((top + bot) / 2.0).clamp(32.0, 1600.0) as u32;
    let th = ((lef + rig) / 2.0).clamp(32.0, 1600.0) as u32;
    let proj = Projection::from_control_points(
        quad,
        [(0.0, 0.0), (tw as f32, 0.0), (tw as f32, th as f32), (0.0, th as f32)],
    )?;
    let out = warp(
        &img.to_rgb8(),
        &proj,
        Interpolation::Bilinear,
        image::Rgb([255u8, 255, 255]),
    );
    Some(DynamicImage::ImageRgb8(out.view(0, 0, tw, th).to_image()))
}

/// 提取字形到 256x256 白底画布(黑字), 返回 (画布, 警告, 自动识别框)
/// 警告非空 = 提取质量可疑(贴了整页截图/多字横排/无边框检测失败)
/// quad = 原图像素坐标的识别框四角(左上/右上/右下/左下), 供前端绘制与微调
pub fn extract_glyph_canvas(img: &DynamicImage) -> (GrayImage, String, [(f32, f32); 4]) {
    let (canvas, warn, quad, _multi, _est) = extract_glyph_canvas_internal(img, false);
    (canvas, warn, quad)
}

/// 用户已手动拉正/裁剪单字框后的提取入口：保留框内全部偏旁部件，绝不退回单连通域误丢左右/上下结构另一半
pub fn extract_glyph_canvas_user_crop(img: &DynamicImage) -> (GrayImage, String, [(f32, f32); 4]) {
    let (canvas, warn, quad, _multi, _est) = extract_glyph_canvas_internal(img, true);
    (canvas, warn, quad)
}

/// 同 extract_glyph_canvas，额外在 <1ms 内基于二值化墨迹连通域判断是否包含多个汉字，
/// 供单字模式上传整碑/对联/多字图时智能提示一键切换「多字整碑识别」。
pub fn extract_glyph_canvas_with_multi_hint(img: &DynamicImage) -> (GrayImage, String, [(f32, f32); 4], bool, u32) {
    extract_glyph_canvas_internal(img, false)
}

fn extract_glyph_canvas_internal(img: &DynamicImage, user_cropped: bool) -> (GrayImage, String, [(f32, f32); 4], bool, u32) {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();

    // 1) 彩色描漆检测: 保留全部 d > 30 像素以防腐蚀细描漆笔画；
    //    同时严格要求彩色像素必须真正集中在图像中心核心区 [15%..85%]（core_frac >= 0.025 且 col_in_core_ratio >= 0.45），
    //    彻底排除四周红框/截图红边/角落小红印章误触发描漆通道导致整张灰度拓片被丢弃！
    let mut colored = ImageBuffer::from_pixel(w, h, Luma([0u8]));
    let mut colored_cnt = 0usize;
    let mut core_colored_cnt = 0usize;
    let mut sat_sum = 0.0f32;
    let mut border_colored_cnt = 0usize;
    let mut border_total = 0usize;
    let cx0 = (w as f32 * 0.15) as u32;
    let cx1 = (w as f32 * 0.85) as u32;
    let cy0 = (h as f32 * 0.15) as u32;
    let cy1 = (h as f32 * 0.85) as u32;
    let core_area = ((cx1.saturating_sub(cx0)).max(1) as usize) * ((cy1.saturating_sub(cy0)).max(1) as usize);
    for (x, y, p) in rgb.enumerate_pixels() {
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        let d = mx - mn;
        let is_border = x == 0 || y == 0 || x + 1 == w || y + 1 == h;
        if is_border {
            border_total += 1;
        }
        if d > 30 {
            colored.put_pixel(x, y, Luma([255u8]));
            colored_cnt += 1;
            if x >= cx0 && x < cx1 && y >= cy0 && y < cy1 {
                core_colored_cnt += 1;
            }
            sat_sum += d as f32 / (mx.max(1) as f32);
            if is_border {
                border_colored_cnt += 1;
            }
        }
    }
    let frac = colored_cnt as f32 / (w * h) as f32;
    let core_frac = core_colored_cnt as f32 / core_area as f32;
    let col_in_core_ratio = core_colored_cnt as f32 / colored_cnt.max(1) as f32;
    let mean_sat = if colored_cnt > 0 { sat_sum / colored_cnt as f32 } else { 0.0 };
    let border_colored_frac = border_colored_cnt as f32 / border_total.max(1) as f32;

    let used_lacquer = frac > 0.02
        && frac < 0.60
        && core_frac >= 0.025
        && col_in_core_ratio >= 0.45
        && border_colored_frac < 0.45
        && mean_sat >= 0.24;
    let bw: GrayImage; // 255=白(背景) 0=黑(笔画)
    if used_lacquer {
        // 描漆: 彩色即笔画, 闭运算连接断裂
        let mut mask = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                mask[(y * w + x) as usize] = colored.get_pixel(x, y)[0];
            }
        }
        morph_close(&mut mask, w, h, 2);
        bw = ImageBuffer::from_fn(w, h, |x, y| {
            Luma([if mask[(y * w + x) as usize] > 0 { 0 } else { 255 }])
        });
    } else {
        // 用 Otsu + 边界连通亮区联合判定黑底白字拓片极性：
        // - 既准确识别全黑底拓片（如碑拓原图）与白边黑底拓片（如魏三体石经「明」），
        // - 又绝不把占满画幅的粗笔白底黑字（如行书「明」）误判反色！
        // 黑底白字须先反相再做自适应阈值，彻底根治粗白笔画被掏空成空心双轮廓线的问题。
        let gray = img.to_luma8();
        let is_white_on_black = detect_white_on_black(&gray);
        let gray_norm = if is_white_on_black {
            ImageBuffer::from_fn(w, h, |x, y| Luma([255u8.saturating_sub(gray.get_pixel(x, y)[0])]))
        } else {
            gray
        };
        // 按图像尺寸自适应放大局部均值窗口（积分图 O(1) 零额外耗时），确保高分辨率照片里的粗笔画保持 100% 实心
        let blk = ((w.min(h) / 3) | 1).clamp(35, 161);
        let c_thr = if is_white_on_black { 18.0 } else { 12.0 };
        let bin = adaptive_threshold(&gray_norm, blk, c_thr);
        bw = ImageBuffer::from_fn(w, h, |x, y| {
            Luma([if bin[(y * w + x) as usize] == 0 { 0 } else { 255 }])
        });
    }

    // 2) 去边框: 框 = 上/下/左/右 >=3 个方向各有一条"几乎全宽的长线"(不限贴边——
    //    字典截图的框四周留白不均, 右侧可距边 8%)。满足框形态才整体擦; 单条长线
    //    ("一"字)不构成框不擦; 国/周等包围字框占宽 ~84% 不到 88% 阈值不触发。
    //    迭代剥层处理粗框/双层框的内层线。
    let (w, h) = bw.dimensions();
    let mut fg = bw.clone();
    let mut frame_removed = false;
    for _ in 0..4 {
        let mut rows_hit: Vec<u32> = Vec::new();
        let mut cols_hit: Vec<u32> = Vec::new();
        for y in 0..h {
            let cnt = (0..w).filter(|&x| fg.get_pixel(x, y)[0] == 0).count();
            if (cnt as f32) > (w as f32) * 0.88 {
                rows_hit.push(y);
            }
        }
        for x in 0..w {
            let cnt = (0..h).filter(|&y| fg.get_pixel(x, y)[0] == 0).count();
            if (cnt as f32) > (h as f32) * 0.88 {
                cols_hit.push(x);
            }
        }
        if rows_hit.is_empty() && cols_hit.is_empty() {
            break;
        }
        let mut dirs = 0; // 长线分布方向数(框至少占三边)
        if rows_hit.iter().any(|&y| (y as f32) < h as f32 * 0.35) { dirs += 1; }
        if rows_hit.iter().any(|&y| (y as f32) > h as f32 * 0.65) { dirs += 1; }
        if cols_hit.iter().any(|&x| (x as f32) < w as f32 * 0.35) { dirs += 1; }
        if cols_hit.iter().any(|&x| (x as f32) > w as f32 * 0.65) { dirs += 1; }
        if dirs < 3 {
            break;
        }
        for &y in &rows_hit {
            for x in 0..w {
                fg.put_pixel(x, y, Luma([255u8]));
            }
        }
        for &x in &cols_hit {
            for y in 0..h {
                fg.put_pixel(x, y, Luma([255u8]));
            }
        }
        frame_removed = true;
    }

    // 2.5) 清除纯贴边外缘杂带（如黑底拓片反相后残留的左右白纸窄边、四角小三角等完全位于中心核心区之外的贴边连通域）
    if !frame_removed {
        erase_outer_margin_strips(&mut fg);
    }

    // 3) 字形 bbox:
    //    - 用户手动确认的裁剪框：联合框内全部有效部件（绝不丢左右/上下结构另一半）
    //    - 有字典外框（已剥净）：多部件联合（收国/明等散件整字）
    //    - 首次自动识别无框图：基于主部件尺度合并同字临近偏旁（左右/上下/多笔字均完整收录，同时排除远处杂物）
    let (mut bx, mut by, mut bw_, mut bh_) = if user_cropped {
        tight_char_components_bbox(&fg, 0.03)
    } else if frame_removed {
        union_components_bbox(&fg, 0.05)
    } else {
        main_char_components_bbox(&fg)
    };
    // 兜底防退化：绝不让识别框坍缩成一个极小噪点
    if bw_ < 16 && bh_ < 16 && w >= 24 && h >= 24 {
        bx = 0;
        by = 0;
        bw_ = w;
        bh_ = h;
    }
    let pad = ((bw_.max(bh_) as f32 * 0.04).round() as u32).clamp(4, 16);
    let x0 = bx.saturating_sub(pad);
    let y0 = by.saturating_sub(pad);
    let x1 = (bx + bw_ + pad).min(w);
    let y1 = (by + bh_ + pad).min(h);
    // 提取质量检查: 细条(非"一"类的极端宽高比)或几乎全图 = 大概率贴错了图
    let mut warn = String::new();
    let ar = bw_ as f32 / bh_.max(1) as f32;
    let cover = (bw_ as f32 * bh_ as f32) / (w as f32 * h as f32);
    if !user_cropped && (ar > 4.0 || ar < 0.12) {
        warn = format!("提取到的是细长条(宽高比 {ar:.1})——可能贴了整页截图或多字横排, 请裁出单个字形; 若检索的就是\"一\"等扁字, 直接看结果即可");
    } else if !user_cropped && cover > 0.94 {
        warn = "提取区域几乎覆盖全图——可能没有清晰的字形主体, 请裁剪后重试".into();
    }
    let cropped = if x1 > x0 && y1 > y0 {
        fg.view(x0, y0, x1 - x0, y1 - y0).to_image() // 必须用擦框后的 fg, bw 会把框线带回来
    } else {
        fg.clone()
    };
    let quad = [
        (x0 as f32, y0 as f32),
        (x1 as f32, y0 as f32),
        (x1 as f32, y1 as f32),
        (x0 as f32, y1 as f32),
    ];

    // 3) 缩放贴到 256x256 白底居中（inner=212 保证后续 CenterCrop(224) 裁去四周 16px 后仍保留 6px 完整白边，高分辨率图绝不切掉笔画末梢）
    let canvas = render_canvas_256(&cropped, 212, None);

    // 4) 单字模式多字嗅探：检测图中是否存在多个独立字形墨迹簇（<1ms，兼容黑底拓片与白底碑帖）
    // 仅当彩色描漆占比 >= 5% 时按描漆通道嗅探，避免角落红印章/红水印掩盖整页黑白碑帖
    let (multi_hint, est_chars) = if user_cropped {
        (false, 1)
    } else {
        estimate_multi_char_hint(img, &fg, used_lacquer && frac >= 0.05)
    };
    (canvas, warn, quad, multi_hint, est_chars)
}

/// 快速判断图中是否包含多个独立汉字（<1ms）：
/// - 对非描漆灰度图先按中位数极性翻正再自适应二值化（彻底避免黑底拓片整页糊成一块）；
/// - 将相近偏旁聚为单字级墨迹簇（如「明/湖/蘭」自动合为 1 簇，绝不误报）；
/// - 检出整碑、对联或 2 字以上横/竖排时返回 (true, 估计字数)。
fn estimate_multi_char_hint(img: &DynamicImage, lacquer_fg: &GrayImage, used_lacquer: bool) -> (bool, u32) {
    let (w, h) = img.dimensions();
    if w < 24 || h < 24 {
        return (false, 1);
    }
    let max_side = 320u32;
    let ds = if w.max(h) > max_side {
        max_side as f32 / w.max(h) as f32
    } else {
        1.0f32
    };
    let sw = ((w as f32 * ds).round() as u32).max(16);
    let sh = ((h as f32 * ds).round() as u32).max(16);

    let small: GrayImage = if used_lacquer {
        if ds < 0.999 {
            image::imageops::resize(lacquer_fg, sw, sh, image::imageops::FilterType::Nearest)
        } else {
            lacquer_fg.clone()
        }
    } else {
        let gray_small = image::imageops::resize(&img.to_luma8(), sw, sh, image::imageops::FilterType::Triangle);
        // 用中心 80% 区域中位数判定黑底拓片极性（避开白纸页边干扰）
        let mut hist = [0u32; 256];
        let x0 = sw / 10;
        let x1 = (sw - x0).max(x0 + 1);
        let y0 = sh / 10;
        let y1 = (sh - y0).max(y0 + 1);
        let mut total = 0u32;
        for y in y0..y1 {
            for x in x0..x1 {
                hist[gray_small.get_pixel(x, y)[0] as usize] += 1;
                total += 1;
            }
        }
        let half = total / 2;
        let mut acc = 0u32;
        let mut med = 255u8;
        for (v, &c) in hist.iter().enumerate() {
            acc += c;
            if acc >= half {
                med = v as u8;
                break;
            }
        }
        let is_rubbing = med < 126;
        let norm = if is_rubbing {
            ImageBuffer::from_fn(sw, sh, |x, y| Luma([255u8.saturating_sub(gray_small.get_pixel(x, y)[0])]))
        } else {
            gray_small
        };
        let blk = ((sw.min(sh) / 8) | 1).clamp(15, 39);
        let bin = adaptive_threshold(&norm, blk, 16.0);
        ImageBuffer::from_fn(sw, sh, |x, y| {
            Luma([if bin[(y * sw + x) as usize] == 0 { 0 } else { 255 }])
        })
    };

    let mut visited = vec![false; (sw * sh) as usize];
    let mut queue = std::collections::VecDeque::new();
    let mut comps: Vec<(f32, f32, f32, f32, u32)> = Vec::new(); // (minx, miny, maxx, maxy, area)
    for sy in 0..sh {
        for sx in 0..sw {
            if small.get_pixel(sx, sy)[0] < 128 && !visited[(sy * sw + sx) as usize] {
                queue.clear();
                queue.push_back((sx, sy));
                visited[(sy * sw + sx) as usize] = true;
                let (mut minx, mut miny, mut maxx, mut maxy) = (sx, sy, sx, sy);
                let mut area = 0u32;
                while let Some((x, y)) = queue.pop_front() {
                    area += 1;
                    minx = minx.min(x); maxx = maxx.max(x);
                    miny = miny.min(y); maxy = maxy.max(y);
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= sw as i32 || ny >= sh as i32 { continue; }
                        let (nx, ny) = (nx as u32, ny as u32);
                        let i = (ny * sw + nx) as usize;
                        if !visited[i] && small.get_pixel(nx, ny)[0] < 128 {
                            visited[i] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
                let cw = (maxx - minx + 1) as f32;
                let ch = (maxy - miny + 1) as f32;
                let touches_border = minx <= 2 || miny <= 2 || maxx + 3 >= sw || maxy + 3 >= sh;
                let fill = area as f32 / (cw * ch).max(1.0);
                let is_hollow_frame = cw >= sw as f32 * 0.55 && ch >= sh as f32 * 0.55 && fill < 0.09;
                let is_long_rule = (cw >= sw as f32 * 0.45 && ch <= 14.0 && cw / ch.max(1.0) >= 8.0)
                    || (ch >= sh as f32 * 0.45 && cw <= 14.0 && ch / cw.max(1.0) >= 8.0);
                let is_border_strip = is_hollow_frame
                    || is_long_rule
                    || (cw >= sw as f32 * 0.65 && ch <= sh as f32 * 0.28 && touches_border)
                    || (ch >= sh as f32 * 0.65 && cw <= sw as f32 * 0.28 && touches_border)
                    || (cw >= sw as f32 * 0.82 && ch >= sh as f32 * 0.82);
                if area >= 8 && !is_border_strip {
                    comps.push((minx as f32, miny as f32, maxx as f32, maxy as f32, area));
                }
            }
        }
    }
    if comps.len() < 2 {
        return (false, 1);
    }

    // 找最大前景部件作为单字参考尺度；过滤极小噪点
    comps.sort_by(|a, b| b.4.cmp(&a.4));
    let top_area = comps[0].4;
    let min_comp_ink = ((top_area as f32 * 0.08) as u32).max(10);
    comps.retain(|c| c.4 >= min_comp_ink);
    if comps.len() < 2 {
        return (false, 1);
    }

    // 参考单字尺度：取前几大连通域最大边的中位数（防止细长一笔把参考尺度拉歪）
    let mut dims: Vec<f32> = comps.iter().take(6).map(|c| (c.2 - c.0 + 1.0).max(c.3 - c.1 + 1.0)).collect();
    dims.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let char_s = dims[dims.len() / 2].max(14.0);

    // 合并相邻偏旁（边界间距 <= 0.28 * char_s 且合并后仍不大于 1.55 * char_s，或中心距 <= 0.72 * char_s）
    let mut clusters: Vec<(f32, f32, f32, f32, u32)> = Vec::new();
    for c in comps {
        let mut merged = false;
        let ccx = (c.0 + c.2) * 0.5;
        let ccy = (c.1 + c.3) * 0.5;
        for cl in &mut clusters {
            let clcx = (cl.0 + cl.2) * 0.5;
            let clcy = (cl.1 + cl.3) * 0.5;
            let nx0 = cl.0.min(c.0);
            let ny0 = cl.1.min(c.1);
            let nx1 = cl.2.max(c.2);
            let ny1 = cl.3.max(c.3);
            let merged_dim = (nx1 - nx0 + 1.0).max(ny1 - ny0 + 1.0);
            let center_close = (ccx - clcx).abs() <= char_s * 0.78 && (ccy - clcy).abs() <= char_s * 0.78;
            let gap_x = (c.0 - cl.2).max(cl.0 - c.2).max(0.0);
            let gap_y = (c.1 - cl.3).max(cl.1 - c.3).max(0.0);
            let bbox_close = gap_x <= char_s * 0.28 && gap_y <= char_s * 0.28 && merged_dim <= char_s * 1.55;
            if center_close || bbox_close {
                cl.0 = nx0;
                cl.1 = ny0;
                cl.2 = nx1;
                cl.3 = ny1;
                cl.4 += c.4;
                merged = true;
                break;
            }
        }
        if !merged {
            clusters.push(c);
        }
    }

    if clusters.len() < 2 {
        return (false, 1);
    }
    clusters.sort_by(|a, b| b.4.cmp(&a.4));
    let main_ink = clusters[0].4;
    let main_dim = (clusters[0].2 - clusters[0].0 + 1.0).max(clusters[0].3 - clusters[0].1 + 1.0);

    // 若主簇自己就已经占满图像长短边（> 55% 短边），且周围只有一个小偏旁/印章，则判为单字
    let min_cl_ink = ((main_ink as f32 * 0.22) as u32).max(20);
    let min_cl_dim = (main_dim * 0.28).max(8.0);
    let valid: Vec<&(f32, f32, f32, f32, u32)> = clusters
        .iter()
        .filter(|cl| {
            let cw = cl.2 - cl.0 + 1.0;
            let ch = cl.3 - cl.1 + 1.0;
            cl.4 >= min_cl_ink && cw.max(ch) >= min_cl_dim
        })
        .collect();

    let count = valid.len() as u32;
    let extra_ink: u32 = valid.iter().skip(1).map(|cl| cl.4).sum();
    let is_multi = count >= 3 || (count == 2 && (extra_ink as f32) >= (main_ink as f32) * 0.50);
    (is_multi, count.min(36))
}

/// 当字形裁切框显著细长（如竖长篆书对联 ar < 0.72）或扁宽（如汉隶 ar > 1.40）时，
/// 计算双视图 TTA 第二视图的温和宽高比归一化目标值；极端单笔划（ar < 0.22 或 > 3.5）不拉伸。
fn compute_norm_target_ar(w: u32, h: u32) -> Option<f32> {
    let ar = w.max(1) as f32 / h.max(1) as f32;
    if ar < 0.72 && ar >= 0.22 {
        Some((ar * 1.85).min(0.85))
    } else if ar > 1.40 && ar <= 3.50 {
        Some((ar / 1.80).max(1.25))
    } else {
        None
    }
}

/// 将二值裁剪字形贴到 256x256 白底画布（inner=204 保证 CenterCrop(224) 仍有完整白边）
fn render_canvas_256(cropped: &GrayImage, inner: u32, target_ar: Option<f32>) -> GrayImage {
    let size = 256u32;
    let cw = cropped.width().max(1) as f32;
    let ch = cropped.height().max(1) as f32;
    let (nw, nh) = match target_ar {
        None => {
            let scale = (inner as f32 / cw).min(inner as f32 / ch);
            (
                ((cw * scale).round() as u32).clamp(1, size),
                ((ch * scale).round() as u32).clamp(1, size),
            )
        }
        Some(tar) => {
            if cw / ch < tar {
                let nh = inner.clamp(1, size);
                let nw = ((nh as f32 * tar).round() as u32).clamp(1, size);
                (nw, nh)
            } else {
                let nw = inner.clamp(1, size);
                let nh = ((nw as f32 / tar).round() as u32).clamp(1, size);
                (nw, nh)
            }
        }
    };
    let thumb = image::imageops::resize(cropped, nw, nh, image::imageops::FilterType::Lanczos3);
    let mut canvas = GrayImage::from_pixel(size, size, Luma([255u8]));
    let _ = canvas.copy_from(&thumb, (size - nw) / 2, (size - nh) / 2);
    canvas
}

/// 单字模式双视图提取：在原 extract_glyph_canvas 基础上，对显著瘦长/扁宽字形额外生成宽高比温和归一化的第二视图
pub fn extract_glyph_canvases(img: &DynamicImage) -> (GrayImage, Option<GrayImage>, f32, String, [(f32, f32); 4]) {
    extract_glyph_canvases_internal(img, false)
}

/// 用户手动裁剪单字框后的双视图提取：保留框内全部偏旁部件
pub fn extract_glyph_canvases_user_crop(img: &DynamicImage) -> (GrayImage, Option<GrayImage>, f32, String, [(f32, f32); 4]) {
    extract_glyph_canvases_internal(img, true)
}

fn extract_glyph_canvases_internal(img: &DynamicImage, user_cropped: bool) -> (GrayImage, Option<GrayImage>, f32, String, [(f32, f32); 4]) {
    let (canvas, warn, quad, _, _) = extract_glyph_canvas_internal(img, user_cropped);
    let gw = (quad[1].0 - quad[0].0).abs().max(1.0);
    let gh = (quad[3].1 - quad[0].1).abs().max(1.0);
    let ar = gw / gh;
    let alt = if let Some(tar) = compute_norm_target_ar(gw.round() as u32, gh.round() as u32) {
        // 从原 256x256 画布中找非白像素紧致边界再按目标宽高比拉伸到 inner=204
        let (mut mnx, mut mny, mut mxx, mut mxy) = (256u32, 256u32, 0u32, 0u32);
        for y in 0..256u32 {
            for x in 0..256u32 {
                if canvas.get_pixel(x, y)[0] < 200 {
                    mnx = mnx.min(x); mny = mny.min(y);
                    mxx = mxx.max(x); mxy = mxy.max(y);
                }
            }
        }
        if mnx < mxx && mny < mxy {
            let sub = canvas.view(mnx, mny, mxx - mnx + 1, mxy - mny + 1).to_image();
            Some(render_canvas_256(&sub, 204, Some(tar)))
        } else {
            None
        }
    } else {
        None
    };
    (canvas, alt, ar, warn, quad)
}

/// 多字模式专用：对检测器已定位好的单个紧致字框做二值化 + 多部件联合居中
/// 传入 full_img 周围上下文的极性参考 bg_luma_hint（若 < 0 则自动从框内边界估计），
/// 避免字框紧贴笔画边缘（如「九」「口」「日」）时把拓片极性误判翻转！
pub fn extract_tight_char_box_canvas(img: &DynamicImage, bg_luma_hint: f32) -> GrayImage {
    extract_tight_char_box_canvases(img, bg_luma_hint).0
}

/// 多字模式双视图提取：返回 (原生等比画布, 可选宽高比温和归一化画布, 紧致字形宽高比)
pub fn extract_tight_char_box_canvases(
    img: &DynamicImage,
    bg_luma_hint: f32,
) -> (GrayImage, Option<GrayImage>, f32) {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    if w < 4 || h < 4 {
        return (GrayImage::from_pixel(256, 256, Luma([255u8])), None, 1.0);
    }

    // 1) 彩色描漆检测: 同单字入口，要求彩色像素真正集中在中心核心区 [15%..85%]，防止边缘红框/小红印章误触发描漆反色
    let mut colored = vec![0u8; (w * h) as usize];
    let mut colored_cnt = 0usize;
    let mut core_colored_cnt = 0usize;
    let mut sat_sum = 0.0f32;
    let mut border_colored_cnt = 0usize;
    let mut border_total = 0usize;
    let cx0 = (w as f32 * 0.15) as u32;
    let cx1 = (w as f32 * 0.85) as u32;
    let cy0 = (h as f32 * 0.15) as u32;
    let cy1 = (h as f32 * 0.85) as u32;
    let core_area = ((cx1.saturating_sub(cx0)).max(1) as usize) * ((cy1.saturating_sub(cy0)).max(1) as usize);
    for (x, y, p) in rgb.enumerate_pixels() {
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        let d = mx - mn;
        let is_border = x == 0 || y == 0 || x + 1 == w || y + 1 == h;
        if is_border {
            border_total += 1;
        }
        if d > 32 {
            colored[(y * w + x) as usize] = 255;
            colored_cnt += 1;
            if x >= cx0 && x < cx1 && y >= cy0 && y < cy1 {
                core_colored_cnt += 1;
            }
            sat_sum += d as f32 / (mx.max(1) as f32);
            if is_border {
                border_colored_cnt += 1;
            }
        }
    }
    let frac = colored_cnt as f32 / (w * h) as f32;
    let core_frac = core_colored_cnt as f32 / core_area as f32;
    let col_in_core_ratio = core_colored_cnt as f32 / colored_cnt.max(1) as f32;
    let mean_sat = if colored_cnt > 0 { sat_sum / colored_cnt as f32 } else { 0.0 };
    let border_colored_frac = border_colored_cnt as f32 / border_total.max(1) as f32;

    let mut fg: GrayImage;
    if frac > 0.025 && frac < 0.65 && core_frac >= 0.025 && col_in_core_ratio >= 0.45 && border_colored_frac < 0.45 && mean_sat >= 0.24 {
        let r_close = if w.min(h) >= 48 { 2 } else { 1 };
        morph_close(&mut colored, w, h, r_close);
        fg = ImageBuffer::from_fn(w, h, |x, y| {
            Luma([if colored[(y * w + x) as usize] > 0 { 0 } else { 255 }])
        });
    } else {
        let gray = img.to_luma8();
        // 任何单个汉字在紧致字框内的笔画占比永远 < 48%，背景永远占 > 52%：
        // 直接取框内全部像素的中位数（p50）判定极性，既不怕笔画紧贴字框四边（如「九」），
        // 也不受拓片外圈白纸页边干扰（如角落里的「永」「之」「事」）！
        let _ = bg_luma_hint;
        let mut hist = [0u32; 256];
        for p in gray.pixels() {
            hist[p[0] as usize] += 1;
        }
        let half = (w * h) / 2;
        let mut acc = 0u32;
        let mut med = 255u8;
        for (v, &c) in hist.iter().enumerate() {
            acc += c;
            if acc >= half {
                med = v as u8;
                break;
            }
        }
        let is_white_on_black = med < 126;

        let gray_norm = if is_white_on_black {
            ImageBuffer::from_fn(w, h, |x, y| Luma([255u8.saturating_sub(gray.get_pixel(x, y)[0])]))
        } else {
            gray
        };
        let blk = ((w.min(h) / 2) | 1).clamp(17, 55);
        let bin = adaptive_threshold(&gray_norm, blk, 12.0);
        fg = ImageBuffer::from_fn(w, h, |x, y| {
            Luma([if bin[(y * w + x) as usize] == 0 { 0 } else { 255 }])
        });
    }

    // 2) 轻微剥除紧贴边界的界格残线（若某边前 2px 行/列覆盖 > 85% 黑色则擦除）
    for y in [0, 1, h.saturating_sub(2), h.saturating_sub(1)] {
        if y < h {
            let cnt = (0..w).filter(|&x| fg.get_pixel(x, y)[0] == 0).count();
            if (cnt as f32) > (w as f32) * 0.85 {
                for x in 0..w { fg.put_pixel(x, y, Luma([255u8])); }
            }
        }
    }
    for x in [0, 1, w.saturating_sub(2), w.saturating_sub(1)] {
        if x < w {
            let cnt = (0..h).filter(|&y| fg.get_pixel(x, y)[0] == 0).count();
            if (cnt as f32) > (h as f32) * 0.85 {
                for y in 0..h { fg.put_pixel(x, y, Luma([255u8])); }
            }
        }
    }

    // 3) 联合框内全部有效部件（紧致字框无字典外框，严禁用 ring_wall 误杀占满字框的连笔大字如「暮/稽/陰」！）
    let (bx, by, bw_, bh_) = tight_char_components_bbox(&fg, 0.03);
    let cropped = if bw_ >= 4 && bh_ >= 4 {
        let pad = ((bw_.max(bh_) as f32 * 0.06).round() as u32).clamp(2, 10);
        let x0 = bx.saturating_sub(pad);
        let y0 = by.saturating_sub(pad);
        let x1 = (bx + bw_ + pad).min(w);
        let y1 = (by + bh_ + pad).min(h);
        fg.view(x0, y0, x1 - x0, y1 - y0).to_image()
    } else {
        fg
    };

    // 4) 等比居中贴到 256x256 白底（inner=204 保证后续 CenterCrop(224) 裁去四周 16px 后仍保留 10px 完整白边，绝不切掉「九/之/永」的笔画末梢！）
    let cw = cropped.width().max(1);
    let ch = cropped.height().max(1);
    let ar = cw as f32 / ch as f32;
    let canvas = render_canvas_256(&cropped, 204, None);
    let canvas_alt = compute_norm_target_ar(cw, ch).map(|tar| render_canvas_256(&cropped, 204, Some(tar)));
    (canvas, canvas_alt, ar)
}

fn center_crop_224(img: &GrayImage) -> GrayImage {
    let off = (256 - 224) / 2;
    img.view(off, off, 224, 224).to_image()
}

/// 积分图 box 均值自适应阈值, 返回 0/255
fn adaptive_threshold(gray: &GrayImage, block: u32, c: f32) -> Vec<u8> {
    let (w, h) = gray.dimensions();
    let mut integral = vec![0u64; (w + 1) as usize * (h + 1) as usize];
    for y in 0..h {
        let mut rowsum = 0u64;
        for x in 0..w {
            rowsum += gray.get_pixel(x, y)[0] as u64;
            integral[((y + 1) * (w + 1) + (x + 1)) as usize] =
                integral[(y * (w + 1) + (x + 1)) as usize] + rowsum;
        }
    }
    let r = block / 2;
    let mut out = vec![255u8; (w * h) as usize];
    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let area = ((x1 - x0 + 1) * (y1 - y0 + 1)) as u64;
            let s = integral[((y1 + 1) * (w + 1) + (x1 + 1)) as usize]
                + integral[(y0 * (w + 1) + x0) as usize]
                - integral[((y1 + 1) * (w + 1) + x0) as usize]
                - integral[(y0 * (w + 1) + x1 + 1) as usize];
            let mean = s as f32 / area as f32;
            out[(y * w + x) as usize] = if (gray.get_pixel(x, y)[0] as f32) < mean - c { 0 } else { 255 };
        }
    }
    out
}

/// 有框时提取字: 丢弃 bbox 覆盖 >=85% 全图的部件(粗框/断口框的环状残余, 永远接近
/// 全图; 字在框内必小于此), 剩余面积 >= frac*最大部件 的连通域联合 bbox(收国/明散件)
fn union_components_bbox(bw: &GrayImage, frac: f32) -> (u32, u32, u32, u32) {
    let (w, h) = bw.dimensions();
    let mut visited = vec![false; (w * h) as usize];
    let mut queue = std::collections::VecDeque::new();
    let mut comps: Vec<(u32, u32, u32, u32, u32)> = Vec::new(); // (minx,miny,maxx,maxy,area)
    for sy in 0..h {
        for sx in 0..w {
            if bw.get_pixel(sx, sy)[0] == 0 && !visited[(sy * w + sx) as usize] {
                queue.clear();
                queue.push_back((sx, sy));
                visited[(sy * w + sx) as usize] = true;
                let (mut minx, mut miny, mut maxx, mut maxy) = (sx, sy, sx, sy);
                let mut area = 0u32;
                while let Some((x, y)) = queue.pop_front() {
                    area += 1;
                    minx = minx.min(x); maxx = maxx.max(x);
                    miny = miny.min(y); maxy = maxy.max(y);
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                        let (nx, ny) = (nx as u32, ny as u32);
                        let i = (ny * w + nx) as usize;
                        if !visited[i] && bw.get_pixel(nx, ny)[0] == 0 {
                            visited[i] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
                comps.push((minx, miny, maxx, maxy, area));
            }
        }
    }
    if comps.is_empty() {
        return (0, 0, 0, 0);
    }
    // 框残余过滤(框线已擦, 剩下的壁残余两种形态):
    // 1) bbox 宽高都 >=85% 全图 = 完整环状残壁
    // 2) 最大部件面积 >3x次大 且 fill<0.13 = 断口框的 U 形/线状壁
    //    (字内部件如"国"的囗与玉面积同量级, 不触发; 全滤掉则退回全量防纯框图空集)
    let (fw, fh) = (w as f32 * 0.85, h as f32 * 0.85);
    let mut sorted = comps.clone();
    sorted.sort_by(|a, b| b.4.cmp(&a.4));
    let ring_wall = |c: &(u32, u32, u32, u32, u32)| {
        (c.2 - c.0 + 1) as f32 >= fw && (c.3 - c.1 + 1) as f32 >= fh
    };
    let thin_line = |c: &(u32, u32, u32, u32, u32)| {
        let fill = c.4 as f32 / (((c.2 - c.0 + 1) * (c.3 - c.1 + 1)) as f32);
        (((c.3 - c.1 + 1) as f32) < h as f32 * 0.02 || ((c.2 - c.0 + 1) as f32) < w as f32 * 0.02)
            && fill > 0.8
    };
    let line_wall = sorted.len() >= 2
        && sorted[1].4 > 500
        && sorted[0].4 > sorted[1].4 * 3
        && (sorted[0].4 as f32
            / (((sorted[0].2 - sorted[0].0 + 1) * (sorted[0].3 - sorted[0].1 + 1)) as f32))
            < 0.13;
    let filtered: Vec<(u32, u32, u32, u32, u32)> = sorted
        .iter()
        .enumerate()
        .filter(|(i, c)| !(ring_wall(c) || thin_line(c) || (*i == 0 && line_wall)))
        .map(|(_, c)| *c)
        .collect();
    let cand = if filtered.is_empty() { comps.clone() } else { filtered };
    let best_area = cand.iter().map(|c| c.4).max().unwrap_or(0);
    if best_area == 0 {
        return (0, 0, 0, 0);
    }
    let thr = (best_area as f32 * frac) as u32;
    let (mut mnx, mut mny, mut mxx, mut mxy) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for &(a, b, c, d, area) in &cand {
        if area >= thr {
            mnx = mnx.min(a); mny = mny.min(b);
            mxx = mxx.max(c); mxy = mxy.max(d);
        }
    }
    (mnx, mny, mxx - mnx + 1, mxy - mny + 1)
}

/// 紧致单字框专用多部件联合 bbox：
/// 绝不执行字典方框的 ring_wall 剔除（否则会把占满 85%+ 紧致框的连笔大字「暮/稽/陰」整体误删！），
/// 仅剔除极小噪点与远离中心且贴边的邻字侵入碎点。
fn tight_char_components_bbox(bw: &GrayImage, frac: f32) -> (u32, u32, u32, u32) {
    let (w, h) = bw.dimensions();
    let mut visited = vec![false; (w * h) as usize];
    let mut queue = std::collections::VecDeque::new();
    let mut comps: Vec<(u32, u32, u32, u32, u32)> = Vec::new();
    for sy in 0..h {
        for sx in 0..w {
            if bw.get_pixel(sx, sy)[0] == 0 && !visited[(sy * w + sx) as usize] {
                queue.clear();
                queue.push_back((sx, sy));
                visited[(sy * w + sx) as usize] = true;
                let (mut minx, mut miny, mut maxx, mut maxy) = (sx, sy, sx, sy);
                let mut area = 0u32;
                while let Some((x, y)) = queue.pop_front() {
                    area += 1;
                    minx = minx.min(x); maxx = maxx.max(x);
                    miny = miny.min(y); maxy = maxy.max(y);
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                        let (nx, ny) = (nx as u32, ny as u32);
                        let i = (ny * w + nx) as usize;
                        if !visited[i] && bw.get_pixel(nx, ny)[0] == 0 {
                            visited[i] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
                comps.push((minx, miny, maxx, maxy, area));
            }
        }
    }
    if comps.is_empty() {
        return (0, 0, 0, 0);
    }
    let best_area = comps.iter().map(|c| c.4).max().unwrap_or(0);
    if best_area == 0 {
        return (0, 0, 0, 0);
    }
    let thr = ((best_area as f32 * frac) as u32).max(4);
    let edge_thr = (best_area as f32 * 0.14) as u32;
    let cx_lo = (w as f32 * 0.14) as u32;
    let cx_hi = (w as f32 * 0.86) as u32;
    let cy_lo = (h as f32 * 0.14) as u32;
    let cy_hi = (h as f32 * 0.86) as u32;

    let (mut mnx, mut mny, mut mxx, mut mxy) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for &(a, b, c, d, area) in &comps {
        if area < thr {
            continue;
        }
        // 仅在某小碎片完全贴边且不落入中心核心区时视为邻字边缘侵入，避免扩偏框
        let touches_edge = a <= 1 || b <= 1 || c + 2 >= w || d + 2 >= h;
        let outside_core = c < cx_lo || a > cx_hi || d < cy_lo || b > cy_hi;
        if touches_edge && outside_core && area < edge_thr {
            continue;
        }
        mnx = mnx.min(a);
        mny = mny.min(b);
        mxx = mxx.max(c);
        mxy = mxy.max(d);
    }
    if mnx == u32::MAX {
        return (0, 0, w, h);
    }
    (mnx, mny, mxx - mnx + 1, mxy - mny + 1)
}

/// 极性判别：区分「白底黑字（含占满画幅的粗笔字与欠曝灰纸黑墨）」与「黑底白字拓片（含光照不均拓片与带白边的拓片截图）」
fn detect_white_on_black(gray: &GrayImage) -> bool {
    let (w, h) = gray.dimensions();
    if w < 8 || h < 8 {
        return false;
    }
    let percentile_from_hist = |hist: &[u64; 256], tot: u64, pct: f64| -> u8 {
        let target = ((tot as f64) * pct).ceil() as u64;
        let mut acc = 0u64;
        for (v, &c) in hist.iter().enumerate() {
            acc += c;
            if acc >= target.max(1) {
                return v as u8;
            }
        }
        255
    };

    // 1) 统计中心 80% 与外缘 6% 亮度分布（中位数 + p05/p95）
    let x0 = w / 10;
    let x1 = (w - x0).max(x0 + 1);
    let y0 = h / 10;
    let y1 = (h - y0).max(y0 + 1);
    let mut int_hist = [0u64; 256];
    let mut int_tot = 0u64;
    for y in y0..y1 {
        for x in x0..x1 {
            int_hist[gray.get_pixel(x, y)[0] as usize] += 1;
            int_tot += 1;
        }
    }
    let int_med = percentile_from_hist(&int_hist, int_tot, 0.50);
    let int_p05 = percentile_from_hist(&int_hist, int_tot, 0.05);
    let int_p95 = percentile_from_hist(&int_hist, int_tot, 0.95);

    let m = (w.min(h) / 16).max(2);
    let mut brd_hist = [0u64; 256];
    let mut brd_tot = 0u64;
    for y in 0..h {
        for x in 0..w {
            if y < m || y + m >= h || x < m || x + m >= w {
                brd_hist[gray.get_pixel(x, y)[0] as usize] += 1;
                brd_tot += 1;
            }
        }
    }
    let border_med = percentile_from_hist(&brd_hist, brd_tot, 0.50);

    // 情形 1：外缘与中心主背景均偏暗（典型整张黑底白字拓片，即使光照不均也稳定），
    // 且亮部笔画明显高于暗底（排除欠曝灰纸黑墨）
    if border_med < 142 && int_med < 142 {
        let hi_span = int_p95.saturating_sub(int_med) as f32;
        let lo_span = int_med.saturating_sub(int_p05) as f32;
        if hi_span >= 28.0 && hi_span >= lo_span * 0.65 {
            return true;
        }
    }

    // 情形 2：外围有亮白边但中心大面积偏暗（区分「带白边的黑底拓片如魏三体石经明」vs「占满框的粗笔行书明」）
    if int_med >= 126 {
        return false;
    }
    let mut hist = [0u64; 256];
    let total = (w as u64) * (h as u64);
    for p in gray.pixels() {
        hist[p[0] as usize] += 1;
    }
    let mut sum_all = 0.0f64;
    for (i, &c) in hist.iter().enumerate() {
        sum_all += (i as f64) * (c as f64);
    }
    let mut w_b = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut best_var = -1.0f64;
    let mut thr = 128u8;
    for i in 0..255usize {
        w_b += hist[i] as f64;
        if w_b == 0.0 { continue; }
        let w_f = (total as f64) - w_b;
        if w_f == 0.0 { break; }
        sum_b += (i as f64) * (hist[i] as f64);
        let m_b = sum_b / w_b;
        let m_f = (sum_all - sum_b) / w_f;
        let diff = m_b - m_f;
        let var = w_b * w_f * diff * diff;
        if var > best_var {
            best_var = var;
            thr = (i as u8).clamp(32, 224);
        }
    }

    let mut int_dark = 0u64;
    for y in y0..y1 {
        for x in x0..x1 {
            if gray.get_pixel(x, y)[0] < thr { int_dark += 1; }
        }
    }
    let int_dark_frac = int_dark as f32 / int_tot.max(1) as f32;
    if int_dark_frac >= 0.60 {
        let mut ext_bright = vec![false; (w * h) as usize];
        let mut q = std::collections::VecDeque::new();
        for x in 0..w {
            for &y in &[0u32, h - 1] {
                if gray.get_pixel(x, y)[0] >= thr {
                    let i = (y * w + x) as usize;
                    if !ext_bright[i] { ext_bright[i] = true; q.push_back((x, y)); }
                }
            }
        }
        for y in 1..h.saturating_sub(1) {
            for &x in &[0u32, w - 1] {
                if gray.get_pixel(x, y)[0] >= thr {
                    let i = (y * w + x) as usize;
                    if !ext_bright[i] { ext_bright[i] = true; q.push_back((x, y)); }
                }
            }
        }
        while let Some((x, y)) = q.pop_front() {
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                let (nx, ny) = (nx as u32, ny as u32);
                let i = (ny * w + nx) as usize;
                if !ext_bright[i] && gray.get_pixel(nx, ny)[0] >= thr {
                    ext_bright[i] = true;
                    q.push_back((nx, ny));
                }
            }
        }
        let mut int_bright_tot = 0u64;
        let mut int_bright_enclosed = 0u64;
        for y in y0..y1 {
            for x in x0..x1 {
                if gray.get_pixel(x, y)[0] >= thr {
                    int_bright_tot += 1;
                    if !ext_bright[(y * w + x) as usize] {
                        int_bright_enclosed += 1;
                    }
                }
            }
        }
        if int_bright_tot > 0 && (int_bright_enclosed as f32 / int_bright_tot as f32) >= 0.45 {
            return true;
        }
    }
    false
}

/// 清除完全位于图像外缘 15% 区域且贴边的伪前景（例如黑底拓片反相后残留的窄白纸边、四角小三角），
/// 保留任何触及中心核心区 [15%..85%] 的真实笔画部件。
fn erase_outer_margin_strips(fg: &mut GrayImage) {
    let (w, h) = fg.dimensions();
    if w < 16 || h < 16 {
        return;
    }
    let cx_lo = (w as f32 * 0.15) as u32;
    let cx_hi = (w as f32 * 0.85) as u32;
    let cy_lo = (h as f32 * 0.15) as u32;
    let cy_hi = (h as f32 * 0.85) as u32;

    let mut visited = vec![false; (w * h) as usize];
    let mut queue = std::collections::VecDeque::new();
    let mut to_erase: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut core_count = 0usize;

    for sy in 0..h {
        for sx in 0..w {
            if fg.get_pixel(sx, sy)[0] == 0 && !visited[(sy * w + sx) as usize] {
                queue.clear();
                queue.push_back((sx, sy));
                visited[(sy * w + sx) as usize] = true;
                let (mut minx, mut miny, mut maxx, mut maxy) = (sx, sy, sx, sy);
                let mut pixels: Vec<(u32, u32)> = Vec::new();
                while let Some((x, y)) = queue.pop_front() {
                    pixels.push((x, y));
                    minx = minx.min(x); maxx = maxx.max(x);
                    miny = miny.min(y); maxy = maxy.max(y);
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                        let (nx, ny) = (nx as u32, ny as u32);
                        let i = (ny * w + nx) as usize;
                        if !visited[i] && fg.get_pixel(nx, ny)[0] == 0 {
                            visited[i] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
                let touches_edge = minx <= 1 || miny <= 1 || maxx + 2 >= w || maxy + 2 >= h;
                let outside_core = maxx < cx_lo || minx > cx_hi || maxy < cy_lo || miny > cy_hi;
                if touches_edge && outside_core {
                    to_erase.push(pixels);
                } else if pixels.len() >= 8 {
                    core_count += 1;
                }
            }
        }
    }
    if core_count > 0 {
        for comp in to_erase {
            for (x, y) in comp {
                fg.put_pixel(x, y, Luma([255u8]));
            }
        }
    }
}

/// 单字无框图专用多部件联合 bbox：
/// 以中心最大笔画为种子，按同字偏旁尺度迭代合并临近部件（完整保留左右结构、上下结构、三点水等散件汉字），
/// 同时排除远离主字的角落噪点与无关背景杂物。
fn main_char_components_bbox(bw: &GrayImage) -> (u32, u32, u32, u32) {
    let (w, h) = bw.dimensions();
    let mut visited = vec![false; (w * h) as usize];
    let mut queue = std::collections::VecDeque::new();
    let mut comps: Vec<(u32, u32, u32, u32, u32)> = Vec::new();
    for sy in 0..h {
        for sx in 0..w {
            if bw.get_pixel(sx, sy)[0] == 0 && !visited[(sy * w + sx) as usize] {
                queue.clear();
                queue.push_back((sx, sy));
                visited[(sy * w + sx) as usize] = true;
                let (mut minx, mut miny, mut maxx, mut maxy) = (sx, sy, sx, sy);
                let mut area = 0u32;
                while let Some((x, y)) = queue.pop_front() {
                    area += 1;
                    minx = minx.min(x); maxx = maxx.max(x);
                    miny = miny.min(y); maxy = maxy.max(y);
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                        let (nx, ny) = (nx as u32, ny as u32);
                        let i = (ny * w + nx) as usize;
                        if !visited[i] && bw.get_pixel(nx, ny)[0] == 0 {
                            visited[i] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
                if area >= 6 {
                    comps.push((minx, miny, maxx, maxy, area));
                }
            }
        }
    }
    if comps.is_empty() {
        return (0, 0, w, h);
    }
    // 排除空心全图残框（覆盖 >= 88% 宽高且填充率 < 10%）
    let non_ring: Vec<(u32, u32, u32, u32, u32)> = comps
        .iter()
        .copied()
        .filter(|&(a, b, c, d, area)| {
            let cw = (c - a + 1) as f32;
            let ch = (d - b + 1) as f32;
            let fill = area as f32 / (cw * ch).max(1.0);
            !(cw >= w as f32 * 0.88 && ch >= h as f32 * 0.88 && fill < 0.10)
        })
        .collect();
    let pool = if non_ring.is_empty() { comps } else { non_ring };

    let cx_lo = (w as f32 * 0.14) as u32;
    let cx_hi = (w as f32 * 0.86) as u32;
    let cy_lo = (h as f32 * 0.14) as u32;
    let cy_hi = (h as f32 * 0.86) as u32;

    // 选中心加权最大的主笔画作为种子
    let mut seed_idx = 0usize;
    let mut seed_score = -1.0f32;
    for (i, &(a, b, c, d, area)) in pool.iter().enumerate() {
        let touches_edge = a <= 1 || b <= 1 || c + 2 >= w || d + 2 >= h;
        let outside_core = c < cx_lo || a > cx_hi || d < cy_lo || b > cy_hi;
        let sc = area as f32 * if touches_edge && outside_core { 0.05 } else { 1.0 };
        if sc > seed_score {
            seed_score = sc;
            seed_idx = i;
        }
    }

    let (mut mnx, mut mny, mut mxx, mut mxy, seed_area) = pool[seed_idx];
    let seed_scale = ((mxx - mnx + 1).max(mxy - mny + 1) as f32).max(16.0);
    let is_single_photo = seed_scale >= (w.min(h) as f32) * 0.34;
    let max_gap = if is_single_photo { seed_scale * 0.60 } else { seed_scale * 0.30 };
    let max_span = if is_single_photo { seed_scale * 1.95 } else { seed_scale * 1.48 };
    let min_part_area = ((seed_area as f32 * 0.025) as u32).max(12);

    let mut used = vec![false; pool.len()];
    used[seed_idx] = true;
    loop {
        let mut changed = false;
        for (i, &(a, b, c, d, area)) in pool.iter().enumerate() {
            if used[i] || area < min_part_area {
                continue;
            }
            let touches_edge = a <= 1 || b <= 1 || c + 2 >= w || d + 2 >= h;
            let outside_core = c < cx_lo || a > cx_hi || d < cy_lo || b > cy_hi;
            if touches_edge && outside_core && area < (seed_area as f32 * 0.18) as u32 {
                continue;
            }
            let gap_x = if a > mxx { (a - mxx) as f32 } else if mnx > c { (mnx - c) as f32 } else { 0.0 };
            let gap_y = if b > mxy { (b - mxy) as f32 } else if mny > d { (mny - d) as f32 } else { 0.0 };
            if gap_x > max_gap || gap_y > max_gap {
                continue;
            }
            let nx0 = mnx.min(a);
            let ny0 = mny.min(b);
            let nx1 = mxx.max(c);
            let ny1 = mxy.max(d);
            let nw = (nx1 - nx0 + 1) as f32;
            let nh = (ny1 - ny0 + 1) as f32;
            if nw <= max_span && nh <= max_span {
                mnx = nx0;
                mny = ny0;
                mxx = nx1;
                mxy = ny1;
                used[i] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    (mnx, mny, mxx - mnx + 1, mxy - mny + 1)
}

/// 5x5 膨胀+腐蚀(闭运算), r=2
fn morph_close(mask: &mut [u8], w: u32, h: u32, r: u32) {    let dilate = |src: &[u8], on: u8, off: u8| -> Vec<u8> {
        let mut dst = vec![off; src.len()];
        for y in 0..h {
            for x in 0..w {
                let mut hit = false;
                for dy in -(r as i32)..=(r as i32) {
                    for dx in -(r as i32)..=(r as i32) {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                        if src[(ny as u32 * w + nx as u32) as usize] == on { hit = true; break; }
                    }
                    if hit { break; }
                }
                dst[(y * w + x) as usize] = if hit { on } else { off };
            }
        }
        dst
    };
    let d = dilate(mask, 255, 0);
    let e = dilate(&d, 0, 255);
    mask.copy_from_slice(&e);
}

