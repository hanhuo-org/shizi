/**
 * 全端统一前端 WebP 压缩模块 (webp_compress.mjs)
 * - Chrome / Edge / Firefox / Android：优先使用原生 Canvas toBlob('image/webp')（0 KB 额外开销、5ms）
 * - iOS / macOS Safari（原生无 WebP 编码器，canvas.toBlob 会静默回落成超大 PNG）：自动按需懒加载本地 vendor/webp/ (@jsquash/webp libwebp WASM) 真 WebP 压缩！
 */
let _nativeWebpSupported = null;
let _webpEncModulePromise = null;

async function checkNativeWebpEncode() {
  if (_nativeWebpSupported !== null) return _nativeWebpSupported;
  try {
    const c = document.createElement("canvas");
    c.width = 2;
    c.height = 2;
    const blob = await new Promise((res) => c.toBlob(res, "image/webp", 0.8));
    _nativeWebpSupported = Boolean(blob && blob.type === "image/webp");
  } catch (_) {
    _nativeWebpSupported = false;
  }
  return _nativeWebpSupported;
}

async function getWasmWebpEncoder() {
  if (!_webpEncModulePromise) {
    _webpEncModulePromise = (async () => {
      const mod = await import("./vendor/webp/webp_enc.js");
      const factory = mod.default;
      const wasmUrl = new URL("./vendor/webp/webp_enc.wasm", import.meta.url).href;
      return await factory({ locateFile: () => wasmUrl });
    })();
  }
  return _webpEncModulePromise;
}

/**
 * 将 data URL / Image 压缩为真 WebP Blob（最大边 <= maxSide，默认 512px，quality 0~1）
 * 返回 { blob, dataUrl, width, height, bytes, encoder }
 */
export async function compressDataUrlToWebp(srcDataUrl, { maxSide = 512, quality = 0.82 } = {}) {
  const img = await new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("图片解码失败"));
    im.src = srcDataUrl;
  });

  const scale = Math.min(1, maxSide / Math.max(img.naturalWidth || 1, img.naturalHeight || 1));
  const w = Math.max(1, Math.round((img.naturalWidth || 1) * scale));
  const h = Math.max(1, Math.round((img.naturalHeight || 1) * scale));

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);

  let blob = null;
  let encoder = "native-canvas";
  if (await checkNativeWebpEncode()) {
    blob = await new Promise((res) => canvas.toBlob(res, "image/webp", quality));
  }
  if (!blob || blob.type !== "image/webp") {
    try {
      const enc = await getWasmWebpEncoder();
      const imgData = ctx.getImageData(0, 0, w, h);
      const qInt = Math.round(Math.max(10, Math.min(100, quality * 100)));
      const opts = {
        quality: qInt,
        target_size: 0,
        target_PSNR: 0,
        method: 4,
        sns_strength: 50,
        filter_strength: 60,
        filter_sharpness: 0,
        filter_type: 1,
        partitions: 0,
        segments: 4,
        pass: 1,
        show_compressed: 0,
        preprocessing: 0,
        autofilter: 0,
        partition_limit: 0,
        alpha_compression: 1,
        alpha_filtering: 1,
        alpha_quality: 100,
        lossless: 0,
        exact: 0,
        image_hint: 0,
        emulate_jpeg_size: 0,
        thread_level: 0,
        low_memory: 0,
        near_lossless: 100,
        use_delta_palette: 0,
        use_sharp_yuv: 0,
      };
      const outU8 = enc.encode(imgData.data, w, h, opts);
      blob = new Blob([outU8], { type: "image/webp" });
      encoder = "jsquash-webp-wasm";
    } catch (_) {
      // 极端环境兜底为 JPEG，确保绝不会上传超大未压缩 PNG
      blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", quality));
      encoder = "jpeg-fallback";
    }
  }

  const dataUrl = await new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(new Error("编码读取失败"));
    r.readAsDataURL(blob);
  });

  return { blob, dataUrl, width: w, height: h, bytes: blob.size, encoder };
}
