/**
 * 汉火字形 (xing.hanhuo.org) 浏览器推理与检索 Web Worker
 * - 零网络本地 Rust-WASM 自动框选检测 (detect_from_rgba) & 透视拉正预处理 (prepare_input_from_rgba)
 * - ConvNeXt-Tiny V12 Weight-Only U8 ONNX 推理 (默认多核 WASM / 可选 WebGPU 显卡加速)
 * - 16.7 万多原型 INT8 向量库常驻 WASM 极速检索 (OPFS 零合并整包加载)
 */
import {
  fetchManifest,
  inspectModelCache,
  loadChunkedModel,
  clearModelCache,
  saveCommunityVectors,
  loadCommunityVectors,
} from "./model_cache.mjs?v=20261121c";

const ASSET_VER = new URL(import.meta.url).searchParams.get("v") || "20261127a";

// 书体大类定义（用于 #styleSelect 大类筛选 & 多字同碑书体家族共识）
const SCRIPT_GROUPS = {
  "group:seal_core": ["zhuanshu", "xiaozhuan", "jinwen", "zhuanke"],
  "group:ancient_seal": [
    "zhuanshu", "xiaozhuan", "jinwen", "zhuanke",
    "zhanguo", "chunqiu", "qinwenzi", "chuwenzi", "jiaguwen", "jinshi",
  ],
  "group:brush_modern": [
    "kaishu", "xingshu", "caoshu", "zhangcao", "weibei", "lishu", "jiandu",
  ],
};
// 单书体同源软扩展：篆刻/汉印(zhuanke)与大小篆(zhuanshu/xiaozhuan)字形高度同源，
// 防止纯 zhuanke 字种稀疏导致真实汉印/官印漏检（实测 HanYin T5: 3.23% -> 88.17%）
const STYLE_SOFT_ALIASES = {
  zhuanke: ["zhuanke", "zhuanshu", "xiaozhuan"],
};
const SEAL_CORE_SET = new Set(SCRIPT_GROUPS["group:seal_core"]);
const ANCIENT_SEAL_SET = new Set(SCRIPT_GROUPS["group:ancient_seal"]);
const BRUSH_MODERN_SET = new Set(SCRIPT_GROUPS["group:brush_modern"]);

function resolveStyleFilter(meta, styleStr) {
  if (!styleStr) return { mask: 0, allowedSet: null, preferredStyle: null };
  const expanded = SCRIPT_GROUPS[styleStr] || STYLE_SOFT_ALIASES[styleStr];
  if (expanded) {
    const allowedSet = new Set(expanded);
    let mask = 0;
    const styles = meta?.styles || [];
    for (let i = 0; i < styles.length && i < 32; i++) {
      if (allowedSet.has(styles[i])) mask |= (1 << i);
    }
    return { mask: mask >>> 0, allowedSet, preferredStyle: STYLE_SOFT_ALIASES[styleStr] ? styleStr : null };
  }
  const idx = (meta?.styles || []).indexOf(styleStr);
  if (idx >= 0 && idx < 32) {
    return { mask: (1 << idx) >>> 0, allowedSet: new Set([styleStr]), preferredStyle: null };
  }
  return { mask: 0, allowedSet: null, preferredStyle: null };
}

function l2NormalizeInPlace(vec) {
  let n2 = 0;
  for (let i = 0; i < vec.length; i++) n2 += vec[i] * vec[i];
  const inv = 1.0 / Math.max(Math.sqrt(n2), 1e-8);
  for (let i = 0; i < vec.length; i++) vec[i] *= inv;
  return vec;
}

function blendNormalizedEmbs(embOrig, embAlt, wOrig = 0.45) {
  const a = l2NormalizeInPlace(new Float32Array(embOrig));
  const b = l2NormalizeInPlace(new Float32Array(embAlt));
  const wAlt = 1.0 - wOrig;
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * wOrig + b[i] * wAlt;
  return l2NormalizeInPlace(out);
}

function applyAbtt(embF32, meta) {
  // Clean V6 ABTT (All-But-The-Top): 减去全局均值并投影剥离前 K=6 个书体/拓片主成分，将特征空间去偏为纯字形结构度量空间
  const abtt = meta?.abtt;
  if (!abtt || !abtt.mean || !abtt.pcs) return embF32;
  const D = embF32.length;
  const out = new Float32Array(D);
  const mean = abtt.mean;
  for (let i = 0; i < D; i++) out[i] = embF32[i] - mean[i];
  const pcs = abtt.pcs;
  for (let k = 0; k < pcs.length; k++) {
    const pc = pcs[k];
    let dot = 0;
    for (let i = 0; i < D; i++) dot += out[i] * pc[i];
    for (let i = 0; i < D; i++) out[i] -= dot * pc[i];
  }
  return l2NormalizeInPlace(out);
}
const _isWorker = typeof self !== "undefined" && typeof self.postMessage === "function";
const ortDistUrl = new URL("./vendor/ort/", import.meta.url).href;
const NUM_WASM_THREADS = _isWorker && self.crossOriginIsolated
  ? Math.min(8, Math.max(2, navigator.hardwareConcurrency || 4))
  : 1;

let wasmMod = null;
let ort = null;
let _ortRuntimePrepared = false;
let _cachedGpuProbe = null;
let protoMeta = null;
let protoMetaVer = "";
let protoIndex = null;
let activeProtoVer = "v2";
let session = null;
let detSession = null;
let detBackend = null;
let bigramLm = null;
let bigramLmPromise = null;
let activeBackend = null;
let activeBackendPref = "wasm";
let gpuHardwareAvailable = false;
let gpuVramOk = false;
let gpuVramReason = "";
let activeGpuName = "";
let activeGpuDiag = "";
let lastCacheInfo = null;
let communityVecs = []; // [{ cid, char, style, dynasty, image_url, q_i8: Int8Array(512), inv_norm: number }]

function quantizeFloat512ToB64(embF32) {
  let maxAbs = 1e-8;
  for (let i = 0; i < embF32.length; i++) {
    const a = Math.abs(embF32[i]);
    if (a > maxAbs) maxAbs = a;
  }
  const u8 = new Uint8Array(512);
  let norm2 = 0;
  for (let i = 0; i < 512; i++) {
    const q = Math.max(-127, Math.min(127, Math.round((embF32[i] / maxAbs) * 127)));
    u8[i] = q & 0xff;
    norm2 += q * q;
  }
  const invNorm = 1.0 / Math.max(Math.sqrt(norm2), 1e-8);
  let bin = "";
  for (let i = 0; i < 512; i++) bin += String.fromCharCode(u8[i]);
  return { embedding_b64: btoa(bin), inv_norm: invNorm };
}

function hydrateCommunityVecs(rawItems, meta = protoMeta) {
  const out = [];
  const hasAbtt = Boolean(meta?.abtt?.mean && meta?.abtt?.pcs);
  for (const it of rawItems || []) {
    if (!it || !it.embedding_b64 || !it.char || !(it.inv_norm > 0)) continue;
    try {
      const bin = atob(it.embedding_b64);
      if (bin.length !== 512) continue;
      let q = new Int8Array(512);
      let invNorm = Number(it.inv_norm);
      for (let i = 0; i < 512; i++) {
        const b = bin.charCodeAt(i);
        q[i] = b > 127 ? b - 256 : b;
      }
      // D1 永久保留原始 ep14.pt 空间向量，载入内存时按当前 proto_meta.abtt 实时映射到检索空间
      if (hasAbtt) {
        const rawF32 = new Float32Array(512);
        for (let i = 0; i < 512; i++) rawF32[i] = q[i] * invNorm;
        const proj = applyAbtt(rawF32, meta);
        let maxAbs = 1e-8;
        for (let i = 0; i < 512; i++) {
          const a = Math.abs(proj[i]);
          if (a > maxAbs) maxAbs = a;
        }
        const qProj = new Int8Array(512);
        let norm2 = 0;
        for (let i = 0; i < 512; i++) {
          const qi = Math.max(-127, Math.min(127, Math.round((proj[i] / maxAbs) * 127)));
          qProj[i] = qi;
          norm2 += qi * qi;
        }
        q = qProj;
        invNorm = 1.0 / Math.max(Math.sqrt(norm2), 1e-8);
      }
      out.push({
        cid: it.cid,
        char: it.char,
        style: it.style || "unknown",
        dynasty: it.dynasty || "众包收录",
        image_url: it.image_url || "",
        q_i8: q,
        inv_norm: invNorm,
      });
    } catch (_) {}
  }
  communityVecs = out;
  return out.length;
}

function rerankByCharConsensus(rawHits, topk) {
  // 温度 LogSumExp 多原型字符聚合共识（动态读取 proto_meta.lse_tau，默认 0.044）：
  const TAU = Number(protoMeta?.lse_tau) || 0.044;
  const byChar = new Map();
  for (const h of rawHits) {
    let entry = byChar.get(h.char);
    if (!entry) {
      entry = { best: h, scores: [h.score] };
      byChar.set(h.char, entry);
    } else {
      if (h.score > entry.best.score) entry.best = h;
      if (entry.scores.length < 6) entry.scores.push(h.score);
    }
  }
  const ranked = [];
  for (const entry of byChar.values()) {
    const scs = entry.scores;
    let maxS = scs[0] || 0;
    for (let i = 1; i < scs.length; i++) {
      if (scs[i] > maxS) maxS = scs[i];
    }
    let sumExp = 0;
    for (let i = 0; i < scs.length; i++) {
      sumExp += Math.exp((scs[i] - maxS) / TAU);
    }
    const lseScore = maxS + Math.log(sumExp) * TAU;
    ranked.push({
      ...entry.best,
      rawScore: maxS,
      score: Math.min(0.999, lseScore),
    });
  }
  ranked.sort((a, b) => b.score - a.score);
  return ranked.slice(0, topk);
}

function searchCommunityVecs(embF32, allowedStyleSet, topk) {
  if (!communityVecs.length) return [];
  let qn2 = 0;
  for (let i = 0; i < 512; i++) qn2 += embF32[i] * embF32[i];
  const invQn = 1.0 / Math.max(Math.sqrt(qn2), 1e-8);
  const hits = [];
  for (const cv of communityVecs) {
    if (allowedStyleSet && !allowedStyleSet.has(cv.style)) continue;
    let dot = 0;
    const q = cv.q_i8;
    for (let i = 0; i < 512; i++) dot += embF32[i] * q[i];
    const score = dot * invQn * cv.inv_norm;
    hits.push({
      id: cv.cid,
      char: cv.char,
      style: cv.style,
      dynasty: cv.dynasty,
      score,
      isCommunity: true,
      imageUrl: cv.image_url,
    });
  }
  hits.sort((a, b) => b.score - a.score);
  return hits.slice(0, topk);
}

async function ensureWasmCore() {
  if (wasmMod) return wasmMod;
  const mod = await import(`./vendor/hanzi_wasm/hanzi_wasm.js?v=${ASSET_VER}`);
  await mod.default(new URL(`./vendor/hanzi_wasm/hanzi_wasm_bg.wasm?v=${ASSET_VER}`, import.meta.url));
  wasmMod = mod;
  return wasmMod;
}

async function gunzipArrayBuffer(buf) {
  const u8 = new Uint8Array(buf);
  if (u8.length < 2 || u8[0] !== 0x1f || u8[1] !== 0x8b) return buf;
  const ds = new DecompressionStream("gzip");
  return await new Response(new Blob([buf]).stream().pipeThrough(ds)).arrayBuffer();
}

async function fetchArrayBufferWithRetry(url, { retries = 2, timeoutMs = 25000, cache = "force-cache" } = {}) {
  let lastErr = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
    try {
      const resp = await fetch(url, { cache, signal: ctrl ? ctrl.signal : undefined });
      if (!resp.ok) {
        if (timer) clearTimeout(timer);
        lastErr = new Error(`HTTP ${resp.status} (${url})`);
      } else {
        const buf = await resp.arrayBuffer();
        if (timer) clearTimeout(timer);
        return buf;
      }
    } catch (err) {
      if (timer) clearTimeout(timer);
      lastErr = err;
    }
    if (attempt < retries) await new Promise((r) => setTimeout(r, 350 * (attempt + 1)));
  }
  throw lastErr || new Error(`拉取失败: ${url}`);
}

async function ensureProtoMeta(protoVer = "", forceFresh = false) {
  const targetVer = protoVer || protoMetaVer || ASSET_VER;
  if (!forceFresh && protoMeta && (!protoVer || protoMetaVer === protoVer)) {
    return protoMeta;
  }
  const qs = forceFresh ? `v=${encodeURIComponent(targetVer)}&ts=${Date.now()}` : `v=${encodeURIComponent(targetVer)}`;
  const cacheMode = forceFresh ? "no-store" : (protoVer ? "no-cache" : "force-cache");
  const buf = await fetchArrayBufferWithRetry(`./assets/proto_meta.json.gz?${qs}`, { cache: cacheMode });
  const raw = await gunzipArrayBuffer(buf);
  protoMeta = JSON.parse(new TextDecoder("utf-8").decode(new Uint8Array(raw)));
  protoMetaVer = targetVer;
  return protoMeta;
}

async function detectWebGPU() {
  if (_cachedGpuProbe) return _cachedGpuProbe;
  if (typeof navigator === "undefined" || !navigator.gpu) {
    _cachedGpuProbe = { available: false, vramOk: false, vramReason: "浏览器未开启 WebGPU", adapterName: "", diag: "no navigator.gpu" };
    return _cachedGpuProbe;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      _cachedGpuProbe = { available: false, vramOk: false, vramReason: "未检测到可用显卡", adapterName: "", diag: "null adapter" };
      return _cachedGpuProbe;
    }
    const info = adapter.info || {};
    const adapterName = [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "WebGPU";
    const maxBufMB = Math.floor((adapter.limits?.maxBufferSize || 0) / (1024 * 1024));
    if (maxBufMB < 256) {
      _cachedGpuProbe = { available: true, vramOk: false, vramReason: `显存上限偏低 (${maxBufMB}MB)`, adapterName, diag: `maxBuf=${maxBufMB}MB` };
      return _cachedGpuProbe;
    }
    const device = await adapter.requestDevice();
    device.pushErrorScope("out-of-memory");
    const probeBuf = device.createBuffer({ size: 128 * 1024 * 1024, usage: GPUBufferUsage.STORAGE });
    const oom = await device.popErrorScope();
    probeBuf.destroy();
    device.destroy();
    if (oom) {
      _cachedGpuProbe = { available: true, vramOk: false, vramReason: "可用显存不足，已自动使用处理器", adapterName, diag: "OOM on 128MB probe" };
      return _cachedGpuProbe;
    }
    _cachedGpuProbe = { available: true, vramOk: true, vramReason: "", adapterName, diag: `OK (${adapterName}, maxBuf=${maxBufMB}MB)` };
    return _cachedGpuProbe;
  } catch (err) {
    _cachedGpuProbe = { available: false, vramOk: false, vramReason: "显卡探测异常，已使用处理器", adapterName: "", diag: String(err?.message || err) };
    return _cachedGpuProbe;
  }
}

async function ensureOrtModule() {
  if (ort) return ort;
  ort = await import("./vendor/ort/ort.webgpu.bundle.min.mjs");
  ort.env.wasm.wasmPaths = ortDistUrl;
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads = NUM_WASM_THREADS;
  return ort;
}

async function prepareOrtWasmBlobRuntime() {
  await ensureOrtModule();
  if (_ortRuntimePrepared || !_isWorker || typeof URL.createObjectURL !== "function") return;
  try {
    const mjsUrl = new URL("./vendor/ort/ort-wasm-simd-threaded.asyncify.mjs", import.meta.url).href;
    const wasmGzUrl = new URL("./vendor/ort/ort-wasm-simd-threaded.asyncify.wasm.gzbin", import.meta.url).href;
    const wasmPlainUrl = new URL("./vendor/ort/ort-wasm-simd-threaded.asyncify.wasm", import.meta.url).href;
    const [mjsResp, wasmResp] = await Promise.all([
      fetch(mjsUrl, { credentials: "same-origin", cache: "force-cache" }),
      fetch(wasmGzUrl, { credentials: "same-origin", cache: "force-cache" }).then((r) =>
        r.ok ? r : fetch(wasmPlainUrl, { credentials: "same-origin", cache: "force-cache" })
      ),
    ]);
    if (!mjsResp.ok || !wasmResp.ok) throw new Error(`ORT runtime fetch failed`);
    const [mjsText, wasmRawOrGz] = await Promise.all([mjsResp.text(), wasmResp.arrayBuffer()]);
    const wasmBuf = await gunzipArrayBuffer(wasmRawOrGz);
    const mjsBlobUrl = URL.createObjectURL(new Blob([mjsText], { type: "text/javascript; charset=utf-8" }));
    ort.env.wasm.wasmPaths = { mjs: mjsBlobUrl, wasm: wasmPlainUrl };
    ort.env.wasm.wasmBinary = wasmBuf;
    _ortRuntimePrepared = true;
  } catch (err) {
    console.warn("[Hanzi Worker] ORT Blob runtime fallback:", err);
  }
}

async function loadEngine(preferredMode = activeBackendPref || "wasm") {
  const pref = preferredMode === "webgpu" ? "webgpu" : "wasm";
  const wantGpuChanged = activeBackendPref !== pref;
  if (!wantGpuChanged && session && protoIndex) {
    return { backend: activeBackend, backendPref: activeBackendPref, cacheInfo: lastCacheInfo };
  }
  activeBackendPref = pref;
  const reloadModelOnly = Boolean(protoIndex);
  if (session) {
    try { await session.release(); } catch (_) {}
    session = null;
  }

  const ortPrepPromise = prepareOrtWasmBlobRuntime();
  const [wasm, gpu, manifest] = await Promise.all([
    ensureWasmCore(),
    detectWebGPU(),
    fetchManifest("./models_chunks/manifest.json"),
  ]);
  const pvVer = manifest.models?.proto_vec?.version || ASSET_VER;
  let meta = await ensureProtoMeta(pvVer);
  gpuHardwareAvailable = gpu.available;
  gpuVramOk = gpu.vramOk;
  gpuVramReason = gpu.vramReason || "";
  activeGpuName = gpu.adapterName || "";
  activeGpuDiag = gpu.diag || "";
  const allowWebGPU = pref === "webgpu" && gpu.available && gpu.vramOk;

  const order = reloadModelOnly
    ? [["resnet50", "视觉编码模型"]]
    : [["resnet50", "视觉编码模型 (1/2)"], ["proto_vec", "22.23 万字形特征向量库 (2/2)"]];
  const totalAllGz = order.reduce((acc, [k]) => acc + (manifest.models[k]?.totalGzSize || 0), 0);
  const totalAllChunks = order.reduce((acc, [k]) => acc + (manifest.models[k]?.chunks?.length || 0), 0);
  let doneGzBase = 0, doneCached = 0, doneNet = 0, opfsHits = 0, allFromCache = true;
  const t0 = performance.now();

  for (const [key, label] of order) {
    const mInfo = manifest.models[key];
    let lastReportAt = 0;
    const res = await loadChunkedModel(key, mInfo, "./models_chunks/", {
      concurrency: 5,
      retries: 3,
      onProgress: (p) => {
        const now = performance.now();
        if (p.phase !== "done" && now - lastReportAt < 80) return;
        lastReportAt = now;
        const curLoaded = doneGzBase + p.loadedBytes;
        const pct = Math.min(99, Math.round((curLoaded / Math.max(1, totalAllGz)) * 99));
        const srcTag = p.phase === "cache" ? "正在读取本地字形库" : "正在下载字形库";
        self.postMessage({
          type: "download_progress",
          pct,
          loadedBytes: curLoaded,
          totalBytes: totalAllGz,
          cachedChunks: doneCached + p.cachedChunks,
          downloadedChunks: doneNet + p.downloadedChunks,
          totalChunks: totalAllChunks,
          speedMBps: p.speedMBps,
          phase: p.phase,
          stage: `${srcTag} · ${label}`,
        });
      },
    });
    doneGzBase += mInfo.totalGzSize || 0;
    doneCached += res.cachedChunks;
    doneNet += res.downloadedChunks;
    if (res.fromOpfs) opfsHits += 1;
    if (!res.fromCacheOnly) allFromCache = false;

    if (key === "resnet50") {
      await ortPrepPromise;
      self.postMessage({ type: "status", stage: "正在初始化视觉识别引擎..." });
      let buf = res.bytes;
      const wasmOpts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
      const webgpuOpts = { executionProviders: ["webgpu"], graphOptimizationLevel: "all" };
      if (allowWebGPU) {
        try {
          session = await ort.InferenceSession.create(buf, webgpuOpts);
          activeBackend = "webgpu";
        } catch (err) {
          console.warn("[Hanzi Worker] WebGPU init failed, falling back to WASM:", err);
          session = await ort.InferenceSession.create(buf, wasmOpts);
          activeBackend = "wasm";
        }
      } else {
        session = await ort.InferenceSession.create(buf, wasmOpts);
        activeBackend = "wasm";
      }
      buf = null;
    } else if (key === "proto_vec") {
      // 自愈校验：若旧缓存/旧标签页的 proto_meta 与刚下载的 proto_vec 向量条数不一致，强制绕过缓存重拉最新 proto_meta.json.gz
      const vecCount = res.bytes.byteLength >= 16
        ? new DataView(res.bytes.buffer, res.bytes.byteOffset, 16).getUint32(4, true)
        : 0;
      if (!meta || !meta.style_ids || (vecCount > 0 && meta.style_ids.length !== vecCount)) {
        meta = await ensureProtoMeta(pvVer, true);
      }
      self.postMessage({
        type: "meta_ready",
        totalPrototypes: meta.total_prototypes,
        totalOriginalGlyphs: meta.total_original_glyphs,
        distinctChars: meta.distinct_chars,
        styles: meta.styles,
        styleCounts: meta.style_counts,
        styleOrigCounts: meta.style_orig_counts,
      });
      self.postMessage({ type: "status", stage: `正在装载 ${(meta.total_prototypes / 10000).toFixed(1)} 万干净字形检索索引...` });
      if (protoIndex) { try { protoIndex.free(); } catch (_) {} protoIndex = null; }
      activeProtoVer = (manifest.models?.proto_vec?.version || ASSET_VER).slice(0, 8);
      const styleU8 = new Uint8Array(meta.style_ids);
      protoIndex = new wasm.ProtoIndex(res.bytes, styleU8);
    }
  }

  if (!reloadModelOnly || !lastCacheInfo) {
    lastCacheInfo = {
      fromCacheOnly: allFromCache,
      opfsHits,
      cachedChunks: doneCached,
      downloadedChunks: doneNet,
      totalChunks: totalAllChunks,
      elapsedSec: +((performance.now() - t0) / 1000).toFixed(2),
      buildId: manifest.buildId || "",
    };
  }
  console.log("[Hanzi Worker] Engine ready:", { backend: activeBackend, threads: NUM_WASM_THREADS, cacheInfo: lastCacheInfo });
  return { backend: activeBackend, backendPref: activeBackendPref, cacheInfo: lastCacheInfo };
}

const DET_IMG_SIZE = 512;
const DET_FM_SIZE = 128;
const DET_STRIDE = 4;

async function ensureCharDetSession(preferredMode = activeBackendPref || "wasm") {
  const pref = preferredMode === "webgpu" ? "webgpu" : "wasm";
  if (detSession && detBackend === pref) return detSession;
  if (detSession) {
    try { await detSession.release(); } catch (_) {}
    detSession = null;
    detBackend = null;
  }
  const [manifest, gpu] = await Promise.all([
    fetchManifest("./models_chunks/manifest.json"),
    detectWebGPU(),
    prepareOrtWasmBlobRuntime(),
  ]);
  const detInfo = manifest.optionalModels?.char_det || manifest.models?.char_det;
  if (!detInfo) {
    throw new Error("未找到多字单字检测模型配置 (char_det)");
  }
  self.postMessage({ type: "status", stage: "正在按需加载轻量 AI 切字模型 (~3 MB)..." });
  const res = await loadChunkedModel("char_det", detInfo, "./models_chunks/", {
    concurrency: 3,
    retries: 3,
    onProgress: (p) => {
      const pct = Math.min(99, Math.round((p.loadedBytes / Math.max(1, detInfo.totalGzSize || 1)) * 99));
      self.postMessage({
        type: "status",
        stage: p.phase === "cache" ? `正在读取本地 AI 切字模型 (${pct}%)...` : `正在下载轻量 AI 切字模型 (${pct}%)...`,
      });
    },
  });
  const allowWebGPU = pref === "webgpu" && gpu.available && gpu.vramOk;
  const wasmOpts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  const webgpuOpts = { executionProviders: ["webgpu"], graphOptimizationLevel: "all" };
  if (allowWebGPU) {
    try {
      detSession = await ort.InferenceSession.create(res.bytes, webgpuOpts);
      detBackend = "webgpu";
    } catch (_) {
      detSession = await ort.InferenceSession.create(res.bytes, wasmOpts);
      detBackend = "wasm";
    }
  } else {
    detSession = await ort.InferenceSession.create(res.bytes, wasmOpts);
    detBackend = "wasm";
  }
  return detSession;
}

function letterboxRgbaTo512(rgba, srcW, srcH) {
  // 四周预留 24px 保护灰边，让紧贴图像顶/底/侧边的残破字（如顶角「皇」）拥有完整卷积感受野，显著提升边角召回
  const DET_MARGIN = 24;
  const inner = DET_IMG_SIZE - 2 * DET_MARGIN;
  const scale = inner / Math.max(1, srcW, srcH);
  const nw = Math.max(1, Math.round(srcW * scale));
  const nh = Math.max(1, Math.round(srcH * scale));
  const ox = Math.floor((DET_IMG_SIZE - nw) / 2);
  const oy = Math.floor((DET_IMG_SIZE - nh) / 2);
  const plane = DET_IMG_SIZE * DET_IMG_SIZE;
  const out = new Float32Array(3 * plane);
  // 中灰填充 (128/255 - 0.5)/0.5 ≈ 0.0039
  out.fill(0.0);

  for (let y = 0; y < nh; y++) {
    const sy = Math.min(srcH - 1, (y + 0.5) / scale - 0.5);
    const sy0 = Math.max(0, Math.floor(sy));
    const sy1 = Math.min(srcH - 1, sy0 + 1);
    const fy = Math.max(0, sy - sy0);
    const dy = (oy + y) * DET_IMG_SIZE;
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(srcW - 1, (x + 0.5) / scale - 0.5);
      const sx0 = Math.max(0, Math.floor(sx));
      const sx1 = Math.min(srcW - 1, sx0 + 1);
      const fx = Math.max(0, sx - sx0);

      const i00 = (sy0 * srcW + sx0) * 4;
      const i10 = (sy0 * srcW + sx1) * 4;
      const i01 = (sy1 * srcW + sx0) * 4;
      const i11 = (sy1 * srcW + sx1) * 4;

      const w00 = (1 - fx) * (1 - fy);
      const w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy;
      const w11 = fx * fy;

      const r = rgba[i00] * w00 + rgba[i10] * w10 + rgba[i01] * w01 + rgba[i11] * w11;
      const g = rgba[i00 + 1] * w00 + rgba[i10 + 1] * w10 + rgba[i01 + 1] * w01 + rgba[i11 + 1] * w11;
      const b = rgba[i00 + 2] * w00 + rgba[i10 + 2] * w10 + rgba[i01 + 2] * w01 + rgba[i11 + 2] * w11;

      const dst = dy + ox + x;
      out[dst] = r / 127.5 - 1.0;
      out[plane + dst] = g / 127.5 - 1.0;
      out[2 * plane + dst] = b / 127.5 - 1.0;
    }
  }
  return { tensorData: out, scale, ox, oy };
}

function boxIouXYXY(a, b) {
  const ix0 = Math.max(a.x0, b.x0);
  const iy0 = Math.max(a.y0, b.y0);
  const ix1 = Math.min(a.x1, b.x1);
  const iy1 = Math.min(a.y1, b.y1);
  const inter = Math.max(0, ix1 - ix0) * Math.max(0, iy1 - iy0);
  if (inter <= 0) return 0;
  const ua = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
  return inter / Math.max(1e-6, ua);
}

// 从 RGBA 区域构造「自动反色+对比度拉伸」224×224 灰度张量（实测金文 Top-1 +6.1%、甲骨 Top-5 +2.1%、碑帖 Top-1 +0.9%）
function makeAutoInvGrayTensorFromRgba(rgba, srcW, srcH, box, manualInvert = false, threshBias = 0) {
  const x0 = Math.max(0, Math.min(srcW - 1, Math.floor(box ? box.x0 : 0)));
  const y0 = Math.max(0, Math.min(srcH - 1, Math.floor(box ? box.y0 : 0)));
  const x1 = Math.max(x0 + 1, Math.min(srcW, Math.ceil(box ? box.x1 : srcW)));
  const y1 = Math.max(y0 + 1, Math.min(srcH, Math.ceil(box ? box.y1 : srcH)));
  const cw = x1 - x0;
  const ch = y1 - y0;
  const n = cw * ch;
  const gray = new Uint8Array(n);
  const hist = new Uint32Array(256);
  let sum = 0;
  for (let y = 0; y < ch; y++) {
    let sp = ((y0 + y) * srcW + x0) * 4;
    let dp = y * cw;
    for (let x = 0; x < cw; x++, sp += 4, dp++) {
      const v = Math.round(0.299 * rgba[sp] + 0.587 * rgba[sp + 1] + 0.114 * rgba[sp + 2]);
      gray[dp] = v;
      sum += v;
    }
  }
  const autoInv = (sum / Math.max(1, n)) < 127.0;
  const doInv = manualInvert ? !autoInv : autoInv;
  for (let i = 0; i < n; i++) {
    const v = doInv ? (255 - gray[i]) : gray[i];
    gray[i] = v;
    hist[v]++;
  }
  // 5% ~ 96% 分位对比度拉伸（保留残边与淡笔画灰度层次）
  const loCnt = Math.floor(n * 0.05);
  const hiCnt = Math.floor(n * 0.96);
  let acc = 0, lo = 0, hi = 255;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (acc >= loCnt) { lo = v; break; }
  }
  acc = 0;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (acc >= hiCnt) { hi = v; break; }
  }
  const span = hi - lo;
  const biasShift = Number(threshBias) || 0;
  if (span > 10 || biasShift !== 0) {
    const invSpan = span > 10 ? 255.0 / span : 1.0;
    for (let i = 0; i < n; i++) {
      let f = span > 10 ? (gray[i] - lo) * invSpan : gray[i];
      if (biasShift !== 0) f -= biasShift * 0.65;
      gray[i] = f <= 0 ? 0 : (f >= 255 ? 255 : Math.round(f));
    }
  }
  // 居中贴到 224×224 白底（inner=204，与训练分布一致）
  const S = 224;
  const plane = S * S;
  const out = new Float32Array(3 * plane);
  out.fill(1.0);
  const scale = Math.min(204 / cw, 204 / ch);
  const nw = Math.max(1, Math.min(S, Math.round(cw * scale)));
  const nh = Math.max(1, Math.min(S, Math.round(ch * scale)));
  const ox = Math.floor((S - nw) / 2);
  const oy = Math.floor((S - nh) / 2);
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(ch - 1, (y + 0.5) / scale - 0.5);
    const sy0 = Math.max(0, Math.floor(sy));
    const sy1 = Math.min(ch - 1, sy0 + 1);
    const fy = Math.max(0, sy - sy0);
    const dy = (oy + y) * S + ox;
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(cw - 1, (x + 0.5) / scale - 0.5);
      const sx0 = Math.max(0, Math.floor(sx));
      const sx1 = Math.min(cw - 1, sx0 + 1);
      const fx = Math.max(0, sx - sx0);
      const v =
        gray[sy0 * cw + sx0] * (1 - fx) * (1 - fy) +
        gray[sy0 * cw + sx1] * fx * (1 - fy) +
        gray[sy1 * cw + sx0] * (1 - fx) * fy +
        gray[sy1 * cw + sx1] * fx * fy;
      const nv = v / 127.5 - 1.0;
      const dst = dy + x;
      out[dst] = nv;
      out[plane + dst] = nv;
      out[2 * plane + dst] = nv;
    }
  }
  return out;
}

function quadToBoundingBox(quadArr, srcW, srcH) {
  if (!quadArr || quadArr.length !== 8) return { x0: 0, y0: 0, x1: srcW, y1: srcH };
  let x0 = srcW, y0 = srcH, x1 = 0, y1 = 0;
  for (let i = 0; i < 8; i += 2) {
    if (quadArr[i] < x0) x0 = quadArr[i];
    if (quadArr[i] > x1) x1 = quadArr[i];
    if (quadArr[i + 1] < y0) y0 = quadArr[i + 1];
    if (quadArr[i + 1] > y1) y1 = quadArr[i + 1];
  }
  if (x1 - x0 < 8 || y1 - y0 < 8) return { x0: 0, y0: 0, x1: srcW, y1: srcH };
  return { x0, y0, x1, y1 };
}

// 单字模式用户手动「反转黑白 / 调节二值化深浅」时，实时生成所见即所得 224 张量 + 预览图
async function buildManualTunedBw(rgba, srcW, srcH, quadArr, manualInvert, threshBias) {
  const box = quadToBoundingBox(quadArr, srcW, srcH);
  const x0 = Math.max(0, Math.min(srcW - 1, Math.floor(box.x0)));
  const y0 = Math.max(0, Math.min(srcH - 1, Math.floor(box.y0)));
  const x1 = Math.max(x0 + 1, Math.min(srcW, Math.ceil(box.x1)));
  const y1 = Math.max(y0 + 1, Math.min(srcH, Math.ceil(box.y1)));
  const cw = x1 - x0, ch = y1 - y0, n = cw * ch;
  const gray = new Uint8Array(n);
  const hist = new Uint32Array(256);
  let sum = 0;
  for (let y = 0; y < ch; y++) {
    let sp = ((y0 + y) * srcW + x0) * 4;
    let dp = y * cw;
    for (let x = 0; x < cw; x++, sp += 4, dp++) {
      const v = Math.round(0.299 * rgba[sp] + 0.587 * rgba[sp + 1] + 0.114 * rgba[sp + 2]);
      gray[dp] = v;
      sum += v;
    }
  }
  const autoInv = (sum / Math.max(1, n)) < 127.0;
  const doInv = manualInvert ? !autoInv : autoInv;
  let totalMean = 0;
  for (let i = 0; i < n; i++) {
    const v = doInv ? (255 - gray[i]) : gray[i];
    gray[i] = v;
    hist[v]++;
    totalMean += v;
  }
  // Otsu 自动阈值 + 用户手工深浅偏移 threshBias（正值=笔画更粗黑，负值=过滤更多杂点）
  let wB = 0, sumB = 0, maxVar = -1, otsuT = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = n - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (totalMean - sumB) / wF;
    const v = wB * wF * (mB - mF) * (mB - mF);
    if (v > maxVar) { maxVar = v; otsuT = t; }
  }
  const thr = Math.max(8, Math.min(248, otsuT + (Number(threshBias) || 0)));
  const bw = new Uint8Array(n);
  let mnx = cw, mny = ch, mxx = -1, mxy = -1;
  for (let y = 0; y < ch; y++) {
    const row = y * cw;
    for (let x = 0; x < cw; x++) {
      const isInk = gray[row + x] <= thr;
      bw[row + x] = isInk ? 0 : 255;
      if (isInk) {
        if (x < mnx) mnx = x;
        if (x > mxx) mxx = x;
        if (y < mny) mny = y;
        if (y > mxy) mxy = y;
      }
    }
  }
  if (mxx <= mnx || mxy <= mny) { mnx = 0; mny = 0; mxx = cw - 1; mxy = ch - 1; }
  const subW = mxx - mnx + 1, subH = mxy - mny + 1;

  // 生成 224×224 模型张量
  const S = 224, plane = S * S;
  const tensor = new Float32Array(3 * plane);
  tensor.fill(1.0);
  const scale = Math.min(184 / subW, 184 / subH);
  const nw = Math.max(1, Math.min(S, Math.round(subW * scale)));
  const nh = Math.max(1, Math.min(S, Math.round(subH * scale)));
  const ox = Math.floor((S - nw) / 2);
  const oy = Math.floor((S - nh) / 2);
  const previewGray = new Uint8Array(S * S);
  previewGray.fill(255);
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(subH - 1, Math.max(0, Math.floor((y + 0.5) / scale)));
    const dy = (oy + y) * S + ox;
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(subW - 1, Math.max(0, Math.floor((x + 0.5) / scale)));
      const v = bw[(mny + sy) * cw + (mnx + sx)];
      previewGray[dy + x] = v;
      const nv = v / 127.5 - 1.0;
      tensor[dy + x] = nv;
      tensor[plane + dy + x] = nv;
      tensor[2 * plane + dy + x] = nv;
    }
  }
  let preview = "";
  if (typeof OffscreenCanvas !== "undefined") {
    try {
      const oc = new OffscreenCanvas(S, S);
      const octx = oc.getContext("2d");
      const imgData = octx.createImageData(S, S);
      const d = imgData.data;
      for (let i = 0; i < S * S; i++) {
        const v = previewGray[i];
        d[i * 4] = v; d[i * 4 + 1] = v; d[i * 4 + 2] = v; d[i * 4 + 3] = 255;
      }
      octx.putImageData(imgData, 0, 0);
      const blob = await oc.convertToBlob({ type: "image/png" });
      const ab = await blob.arrayBuffer();
      const u8 = new Uint8Array(ab);
      let bin = "";
      for (let i = 0; i < u8.length; i += 8192) {
        bin += String.fromCharCode(...u8.subarray(i, Math.min(u8.length, i + 8192)));
      }
      preview = "data:image/png;base64," + btoa(bin);
    } catch (_) {}
  }
  return { tensor, preview };
}

function decodeCenterNetBoxes(hm, whReg, scale, ox, oy, srcW, srcH, minThresh = 0.20, nmsIou = 0.45, maxDets = 64) {
  const FM = DET_FM_SIZE;
  const plane = FM * FM;
  const raw = [];
  for (let y = 0; y < FM; y++) {
    const rowOff = y * FM;
    for (let x = 0; x < FM; x++) {
      const idx = rowOff + x;
      const sc = hm[idx];
      if (sc < minThresh) continue;
      // 3x3 局部极大值判断
      let isPeak = true;
      for (let dy = -1; dy <= 1 && isPeak; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= FM) continue;
        const nRow = ny * FM;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= FM || (dx === 0 && dy === 0)) continue;
          if (hm[nRow + nx] > sc) { isPeak = false; break; }
        }
      }
      if (!isPeak) continue;
      const bw = whReg[idx] * DET_STRIDE;
      const bh = whReg[plane + idx] * DET_STRIDE;
      const dx = whReg[2 * plane + idx];
      const dy = whReg[3 * plane + idx];
      const cx = (x + dx) * DET_STRIDE;
      const cy = (y + dy) * DET_STRIDE;
      // 逆 letterbox 映回原图坐标，外扩 4% 给笔画末端留余量
      const padW = bw * 0.04;
      const padH = bh * 0.04;
      const x0 = Math.max(0, Math.min(srcW - 1, (cx - bw * 0.5 - padW - ox) / scale));
      const y0 = Math.max(0, Math.min(srcH - 1, (cy - bh * 0.5 - padH - oy) / scale));
      const x1 = Math.max(x0 + 1, Math.min(srcW, (cx + bw * 0.5 + padW - ox) / scale));
      const y1 = Math.max(y0 + 1, Math.min(srcH, (cy + bh * 0.5 + padH - oy) / scale));
      if (x1 - x0 < 10 || y1 - y0 < 10) continue;
      raw.push({ x0, y0, x1, y1, score: sc });
    }
  }
  raw.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of raw) {
    let dup = false;
    for (const k of kept) {
      if (boxIouXYXY(c, k) >= nmsIou) { dup = true; break; }
    }
    if (!dup) {
      kept.push(c);
      if (kept.length >= maxDets) break;
    }
  }
  // 拓片残破白边细长毛边假框防护：仅抑制紧邻且侵入正常宽字框边缘、极细窄且低分的碎片条（绝不误伤独立窄字）
  if (kept.length >= 4) {
    const ws = kept.map((b) => b.x1 - b.x0).sort((a, b) => a - b);
    const medW = ws[Math.floor(ws.length / 2)] || 1;
    const filtered = [];
    for (const b of kept) {
      const bw = b.x1 - b.x0;
      const bh = b.y1 - b.y0;
      if (b.score < 0.45 && bw < medW * 0.45 && bw / Math.max(1, bh) < 0.46) {
        const overlapsNormal = kept.some(
          (k) =>
            k !== b &&
            k.score > b.score &&
            (k.x1 - k.x0) >= medW * 0.65 &&
            Math.min(b.x1, k.x1) > Math.max(b.x0, k.x0) &&
            Math.min(b.y1, k.y1) > Math.max(b.y0, k.y0)
        );
        if (overlapsNormal) continue;
      }
      filtered.push(b);
    }
    return filtered;
  }
  return kept;
}

// ===================== 古汉语三元+分域语言模型 (HXB2/HXB1) 与二阶 Viterbi 上下文纠偏 =====================
// Domain IDs in HXB2: 0=oracle(甲骨卜辞), 1=bronze(青铜器金文), 2=seal(汉印官印), 3=stele(碑帖/简帛/诗文)
const TRI_STRIDE1 = 16384 * 16384; // vocab <= 16384 -> 3*14=42 bits < 53-bit IEEE-754 exact integer!
const TRI_STRIDE2 = 16384;
const DOM_BI_STRIDE0 = 65536 * 65536; // 4 * 2^32 < 2^53 exact
const DOM_TRI_STRIDE0 = 4 * 16384 * 16384 * 16384; // wait: 4*2^42 = 2^44 < 2^53 exact! Use 2^42 stride:
const DOM_TRI_S0 = 16384 * 16384 * 16384; // 2^42 = 4,398,046,511,104 (d * 2^42 + id1 * 2^28 + id2 * 2^14 + id3 < 2^45 < 2^53!)

function inferArtifactDomain(items, styleStr = "") {
  if (styleStr === "jiaguwen") return 0;
  if (styleStr === "jinwen" || styleStr === "chunqiu" || styleStr === "zhanguo" || styleStr === "qinwenzi") return 1;
  if (styleStr === "zhuanke" || styleStr === "zhuanshu" || styleStr === "xiaozhuan" || styleStr === "group:seal_core") return 2;
  if (styleStr && styleStr !== "group:ancient_seal") return 3;
  // Auto-detect from Top-5 visual styles across the multi-character sequence
  const scores = [0, 0, 0, 0];
  const RANK_W = [1.0, 0.78, 0.62, 0.50, 0.40];
  for (const it of items || []) {
    const top5 = (it._viterbiPool || it.hits || []).slice(0, 5);
    for (let r = 0; r < top5.length; r++) {
      const h = top5[r];
      const w = Math.max(0, Number(h.score)) * RANK_W[r];
      const st = h.style || "";
      if (st === "jiaguwen") scores[0] += w;
      else if (st === "jinwen" || st === "chunqiu" || st === "zhanguo" || st === "qinwenzi") scores[1] += w;
      else if (st === "zhuanke" || st === "zhuanshu" || st === "xiaozhuan") scores[2] += w;
      else if (st !== "unknown") scores[3] += w;
    }
  }
  let bestD = 3;
  let bestV = scores[3];
  for (let d = 0; d < 3; d++) {
    if (scores[d] > bestV) {
      bestV = scores[d];
      bestD = d;
    }
  }
  const tot = scores[0] + scores[1] + scores[2] + scores[3];
  return tot > 1e-6 && bestV / tot >= 0.34 ? bestD : 3;
}

async function ensureClassicalBigram() {
  if (bigramLm) return bigramLm;
  if (bigramLmPromise) return bigramLmPromise;
  bigramLmPromise = (async () => {
    try {
      const buf = await fetchArrayBufferWithRetry(`./assets/classical_bigram.bin.gz?v=${ASSET_VER}`);
      const raw = await gunzipArrayBuffer(buf);
      const u8 = new Uint8Array(raw);
      const dv = new DataView(raw);
      const isHxb2 = u8.length >= 24 && u8[0] === 0x48 && u8[1] === 0x58 && u8[2] === 0x42 && u8[3] === 0x32;
      const isHxb1 = u8.length >= 12 && u8[0] === 0x48 && u8[1] === 0x58 && u8[2] === 0x42 && u8[3] === 0x31;
      if (!isHxb2 && !isHxb1) {
        throw new Error("Invalid HXB1/HXB2 LM magic");
      }
      const nVocab = dv.getUint32(4, true);
      const nBi = dv.getUint32(8, true);
      const nTri = isHxb2 ? dv.getUint32(12, true) : 0;
      const nDomBi = isHxb2 ? dv.getUint32(16, true) : 0;
      const nDomTri = isHxb2 ? dv.getUint32(20, true) : 0;
      let off = isHxb2 ? 24 : 12;

      const vocabCp = new Uint32Array(nVocab);
      for (let i = 0; i < nVocab; i++, off += 4) vocabCp[i] = dv.getUint32(off, true);
      const uniQ8 = u8.slice(off, off + nVocab);
      off += nVocab;

      const biKeys = new Uint32Array(nBi);
      const biScores = new Uint8Array(nBi);
      for (let i = 0; i < nBi; i++, off += 5) {
        const id1 = dv.getUint16(off, true);
        const id2 = dv.getUint16(off + 2, true);
        biKeys[i] = ((id1 << 16) | id2) >>> 0;
        biScores[i] = u8[off + 4];
      }

      const triKeys = new Float64Array(nTri);
      const triScores = new Uint8Array(nTri);
      for (let i = 0; i < nTri; i++, off += 7) {
        const id1 = dv.getUint16(off, true);
        const id2 = dv.getUint16(off + 2, true);
        const id3 = dv.getUint16(off + 4, true);
        triKeys[i] = id1 * TRI_STRIDE1 + id2 * TRI_STRIDE2 + id3;
        triScores[i] = u8[off + 6];
      }

      const domBiKeys = new Float64Array(nDomBi);
      const domBiScores = new Uint8Array(nDomBi);
      for (let i = 0; i < nDomBi; i++, off += 6) {
        const d = u8[off];
        const id1 = dv.getUint16(off + 1, true);
        const id2 = dv.getUint16(off + 3, true);
        domBiKeys[i] = d * DOM_BI_STRIDE0 + id1 * 65536 + id2;
        domBiScores[i] = u8[off + 5];
      }

      const domTriKeys = new Float64Array(nDomTri);
      const domTriScores = new Uint8Array(nDomTri);
      for (let i = 0; i < nDomTri; i++, off += 8) {
        const d = u8[off];
        const id1 = dv.getUint16(off + 1, true);
        const id2 = dv.getUint16(off + 3, true);
        const id3 = dv.getUint16(off + 5, true);
        domTriKeys[i] = d * DOM_TRI_S0 + id1 * TRI_STRIDE1 + id2 * TRI_STRIDE2 + id3;
        domTriScores[i] = u8[off + 7];
      }

      bigramLm = {
        version: isHxb2 ? 2 : 1,
        nVocab, nBi, nTri, nDomBi, nDomTri,
        vocabCp, uniQ8,
        biKeys, biScores,
        triKeys, triScores,
        domBiKeys, domBiScores,
        domTriKeys, domTriScores,
      };
      console.log(
        `[Hanzi Worker] Classical LM HXB${bigramLm.version} ready: vocab=${nVocab}, bi=${nBi}, tri=${nTri}, domBi=${nDomBi}, domTri=${nDomTri}`
      );
      return bigramLm;
    } catch (err) {
      console.warn("[Hanzi Worker] Classical LM unavailable:", err);
      bigramLmPromise = null;
      return null;
    }
  })();
  return bigramLmPromise;
}

function lmLookupCharId(lm, ch) {
  if (!lm || !ch) return -1;
  const cp = ch.codePointAt(0) || 0;
  let lo = 0, hi = lm.nVocab - 1;
  const arr = lm.vocabCp;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const v = arr[mid];
    if (v === cp) return mid;
    if (v < cp) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

function bsearchScore(keys, scores, len, key) {
  let lo = 0, hi = len - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const v = keys[mid];
    if (v === key) return scores[mid] / 255.0;
    if (v < key) lo = mid + 1;
    else hi = mid - 1;
  }
  return 0;
}

function lmBigramNorm(lm, id1, id2, domId = -1) {
  if (!lm || id1 < 0 || id2 < 0) return 0;
  let sc = bsearchScore(lm.biKeys, lm.biScores, lm.nBi, ((id1 << 16) | id2) >>> 0);
  if (domId >= 0 && lm.nDomBi > 0) {
    const dsc = bsearchScore(lm.domBiKeys, lm.domBiScores, lm.nDomBi, domId * DOM_BI_STRIDE0 + id1 * 65536 + id2);
    if (dsc > 0) sc = Math.max(sc, Math.min(1.0, dsc * 1.15));
  }
  return sc;
}

function lmTrigramNorm(lm, id1, id2, id3, domId = -1) {
  if (!lm || id1 < 0 || id2 < 0 || id3 < 0) return 0;
  let sc = lm.nTri > 0 ? bsearchScore(lm.triKeys, lm.triScores, lm.nTri, id1 * TRI_STRIDE1 + id2 * TRI_STRIDE2 + id3) : 0;
  if (domId >= 0 && lm.nDomTri > 0) {
    const dsc = bsearchScore(
      lm.domTriKeys,
      lm.domTriScores,
      lm.nDomTri,
      domId * DOM_TRI_S0 + id1 * TRI_STRIDE1 + id2 * TRI_STRIDE2 + id3
    );
    if (dsc > 0) sc = Math.max(sc, Math.min(1.0, dsc * 1.25));
  }
  return sc;
}

function applyWholeSteleScriptConsistency(items, topk, opts = {}) {
  const boostVal = Number.isFinite(opts.consistencyBoost) ? Number(opts.consistencyBoost) : 0.05;
  const sealCoreExtra = Number.isFinite(opts.sealCoreExtra) ? Number(opts.sealCoreExtra) : 0.015;
  if (!items || items.length < 3 || boostVal <= 0) {
    return { family: null, ancientShare: 0, sealCoreShare: 0, brushShare: 0 };
  }
  const RANK_W = [1.0, 0.82, 0.68, 0.56, 0.46];
  let ancientMass = 0;
  let sealCoreMass = 0;
  let brushMass = 0;
  let totalMass = 0;
  for (const it of items) {
    const top5 = (it._viterbiPool || it.hits || []).slice(0, 5);
    for (let r = 0; r < top5.length; r++) {
      const h = top5[r];
      const w = Math.max(0, Number(h.score)) * RANK_W[r];
      totalMass += w;
      if (ANCIENT_SEAL_SET.has(h.style)) ancientMass += w;
      if (SEAL_CORE_SET.has(h.style)) sealCoreMass += w;
      if (BRUSH_MODERN_SET.has(h.style)) brushMass += w;
    }
  }
  const denom = Math.max(1e-6, totalMass);
  const ancientShare = ancientMass / denom;
  const sealCoreShare = sealCoreMass / denom;
  const brushShare = brushMass / denom;

  let family = null;
  let targetSet = null;
  let subSet = null;
  let subExtra = 0;
  if (ancientShare >= 0.55 && (ancientShare - brushShare) >= 0.20) {
    family = "ancient_seal";
    targetSet = ANCIENT_SEAL_SET;
    if (sealCoreShare >= 0.30 && sealCoreExtra > 0) {
      subSet = SEAL_CORE_SET;
      subExtra = sealCoreExtra;
    }
  } else if (brushShare >= 0.55 && (brushShare - ancientShare) >= 0.20) {
    family = "brush_modern";
    targetSet = BRUSH_MODERN_SET;
  }

  if (targetSet) {
    for (const it of items) {
      const pool = it._viterbiPool || it.hits || [];
      for (const h of pool) {
        let b = targetSet.has(h.style) ? boostVal : 0;
        if (subSet && subSet.has(h.style)) b += subExtra;
        h.familyBoost = b;
      }
      pool.sort((a, b) => (Number(b.score) + (b.familyBoost || 0)) - (Number(a.score) + (a.familyBoost || 0)));
      it.hits = pool.slice(0, topk);
    }
  }
  return {
    family,
    ancientShare: +ancientShare.toFixed(3),
    sealCoreShare: +sealCoreShare.toFixed(3),
    brushShare: +brushShare.toFixed(3),
  };
}

function detectSequenceBreaks(items, orderMode = "vertical-rl") {
  const N = items.length;
  const breaks = new Uint8Array(N);
  if (N < 2) return breaks;
  const ws = [];
  const hs = [];
  for (const it of items) {
    const b = it.box;
    if (b && b.x1 > b.x0 && b.y1 > b.y0) {
      ws.push(b.x1 - b.x0);
      hs.push(b.y1 - b.y0);
    }
  }
  if (!ws.length) return breaks;
  ws.sort((a, b) => a - b);
  hs.sort((a, b) => a - b);
  const medW = ws[Math.floor(ws.length / 2)] || 40;
  const medH = hs[Math.floor(hs.length / 2)] || 40;

  for (let i = 1; i < N; i++) {
    const prev = items[i - 1].box;
    const cur = items[i].box;
    if (!prev || !cur || prev.x1 <= prev.x0 || cur.x1 <= cur.x0) continue;
    const pcx = (prev.x0 + prev.x1) * 0.5;
    const pcy = (prev.y0 + prev.y1) * 0.5;
    const ccx = (cur.x0 + cur.x1) * 0.5;
    const ccy = (cur.y0 + cur.y1) * 0.5;
    if (orderMode === "vertical-rl") {
      if (Math.abs(ccx - pcx) > 0.9 * medW || ccy < pcy - 0.5 * medH) breaks[i] = 1;
    } else if (orderMode === "horizontal-lr") {
      if (Math.abs(ccy - pcy) > 0.9 * medH || ccx < pcx - 0.5 * medW) breaks[i] = 1;
    } else if (orderMode === "horizontal-rl") {
      if (Math.abs(ccy - pcy) > 0.9 * medH || ccx > pcx + 0.5 * medW) breaks[i] = 1;
    }
  }
  return breaks;
}

function viterbiRerankMulti(items, lm, opts = {}) {
  if (!lm || !items || items.length < 2) {
    return (items || []).map((it) => ({
      lmBestChar: it.hits?.[0]?.char || "",
      lmBestHitId: it.hits?.[0]?.id,
      lmRescued: false,
    }));
  }
  const N = items.length;
  const orderMode = opts.orderMode || "vertical-rl";
  const enableBreakGuard = opts.enableBreakGuard !== false;
  const enableConfGuard = opts.enableConfGuard !== false;
  const domId = lm.version >= 2 ? inferArtifactDomain(items, opts.styleStr || "") : -1;
  const BI_W = 0.22;
  const TRI_W = lm.version >= 2 ? 0.24 : 0.0;
  const UNI_W = 0.025;
  const VOTE_W = 0.018; // 同字多原型 Top-12 投票共识加分（防单个偶发异常原型压倒 4 个高分真字原型）
  const UNENC_PEN = 0.06;
  const MAX_VIS_GAP = 0.26; // 严格视觉底线：与首名视觉差距超过 0.26 的候选绝不被语境强行拉起，防幻觉
  const LOW_CONF_THR = 0.36;

  const breaks = enableBreakGuard ? detectSequenceBreaks(items, orderMode) : new Uint8Array(N);

  // 每个位置抽取去重后的最多 12 个视觉候选字（二阶 Viterbi 12*12=144 状态，<1ms）
  const rawTop1s = new Float32Array(N);
  const layers = items.map((it, idx) => {
    const pool = it._viterbiPool || it.hits || [];
    const rankScore = (h) => Number(h.score) + Number(h.familyBoost || 0);
    const top1Rank = pool.length ? rankScore(pool[0]) : 0;
    let maxRaw = 0;
    for (let pi = 0; pi < Math.min(5, pool.length); pi++) {
      const rs = Number(pool[pi].score);
      if (rs > maxRaw) maxRaw = rs;
    }
    rawTop1s[idx] = maxRaw;
    const isLowConf = enableConfGuard && maxRaw < LOW_CONF_THR;
    const uniW = isLowConf ? 0.004 : UNI_W;
    const visGap = isLowConf ? 0.16 : MAX_VIS_GAP;

    const top12Counts = new Map();
    for (let pi = 0; pi < Math.min(12, pool.length); pi++) {
      const ch = pool[pi]?.char;
      if (ch) top12Counts.set(ch, (top12Counts.get(ch) || 0) + 1);
    }
    const seen = new Set();
    const cands = [];
    for (const h of pool) {
      if (!h.char || seen.has(h.char)) continue;
      const sc = rankScore(h);
      if (cands.length > 0 && top1Rank - sc > visGap) continue;
      seen.add(h.char);
      const cid = lmLookupCharId(lm, h.char);
      const cp = h.char.codePointAt(0) || 0;
      const isUnenc = cp >= 0x40000 || (cp >= 0xE000 && cp <= 0xF8FF);
      const uniNorm = cid >= 0 ? lm.uniQ8[cid] / 255.0 : 0;
      const votes = Math.min(3, Math.max(0, (top12Counts.get(h.char) || 1) - 1));
      cands.push({
        char: h.char,
        hit: h,
        cid,
        emit: sc + VOTE_W * votes + uniW * uniNorm - (isUnenc ? UNENC_PEN : 0),
      });
      if (cands.length >= 12) break;
    }
    if (!cands.length) cands.push({ char: "", hit: null, cid: -1, emit: 0 });
    return cands;
  });

  const biTrans = (cPrev, cCur, isBreak, prevLow, curLow) => {
    if (isBreak) return 0;
    const biRaw = lmBigramNorm(lm, cPrev.cid, cCur.cid, domId);
    if (biRaw <= 0) return 0;
    if (prevLow && curLow) return biRaw >= 0.26 ? BI_W * 0.80 * Math.pow(biRaw, 0.72) : 0;
    if (prevLow || curLow) return biRaw >= 0.14 ? BI_W * Math.pow(biRaw, 0.62) : 0;
    return BI_W * Math.pow(biRaw, 0.60);
  };

  const triTrans = (cPrev2, cPrev1, cCur, b1, b2, low2, low1, low0) => {
    if (TRI_W <= 0 || b1 || b2) return 0;
    const triRaw = lmTrigramNorm(lm, cPrev2.cid, cPrev1.cid, cCur.cid, domId);
    if (triRaw <= 0) return 0;
    const nLow = (low2 ? 1 : 0) + (low1 ? 1 : 0) + (low0 ? 1 : 0);
    if (nLow >= 2) return triRaw >= 0.28 ? TRI_W * 0.75 * Math.pow(triRaw, 0.72) : 0;
    return TRI_W * Math.pow(triRaw, 0.60);
  };

  // 二阶 Viterbi DP: 状态为 (prev_idx, cur_idx) 展平为 prev_idx * curLen + cur_idx
  // i = 0 -> 1 初始化
  const L0 = layers[0].length;
  const L1 = layers[1].length;
  const dp2 = new Array(N);
  const back2 = new Array(N);
  dp2[1] = new Float64Array(L0 * L1);
  back2[1] = new Int16Array(L0 * L1);
  {
    const isBreak = breaks[1] === 1;
    const low0 = enableConfGuard && rawTop1s[0] < LOW_CONF_THR;
    const low1 = enableConfGuard && rawTop1s[1] < LOW_CONF_THR;
    for (let k = 0; k < L0; k++) {
      for (let j = 0; j < L1; j++) {
        dp2[1][k * L1 + j] = layers[0][k].emit + biTrans(layers[0][k], layers[1][j], isBreak, low0, low1) + layers[1][j].emit;
        back2[1][k * L1 + j] = k;
      }
    }
  }

  for (let i = 2; i < N; i++) {
    const Lp2 = layers[i - 2].length;
    const Lp1 = layers[i - 1].length;
    const Lc = layers[i].length;
    const curDp = new Float64Array(Lp1 * Lc);
    const curBack = new Int16Array(Lp1 * Lc);
    const b1 = breaks[i - 1] === 1;
    const b2 = breaks[i] === 1;
    const low2 = enableConfGuard && rawTop1s[i - 2] < LOW_CONF_THR;
    const low1 = enableConfGuard && rawTop1s[i - 1] < LOW_CONF_THR;
    const low0 = enableConfGuard && rawTop1s[i] < LOW_CONF_THR;
    const prevDp = dp2[i - 1];

    for (let k = 0; k < Lp1; k++) {
      const ck = layers[i - 1][k];
      for (let j = 0; j < Lc; j++) {
        const cj = layers[i][j];
        const bScore = biTrans(ck, cj, b2, low1, low0);
        let bestVal = -1e12;
        let bestM = 0;
        for (let m = 0; m < Lp2; m++) {
          const tScore = triTrans(layers[i - 2][m], ck, cj, b1, b2, low2, low1, low0);
          const val = prevDp[m * Lp1 + k] + bScore + tScore + cj.emit;
          if (val > bestVal) {
            bestVal = val;
            bestM = m;
          }
        }
        curDp[k * Lc + j] = bestVal;
        curBack[k * Lc + j] = bestM;
      }
    }
    dp2[i] = curDp;
    back2[i] = curBack;
  }

  // 回溯最优状态链
  const lastDp = dp2[N - 1];
  const lastLp1 = layers[N - 2].length;
  const lastLc = layers[N - 1].length;
  let bestPair = 0;
  for (let s = 1; s < lastDp.length; s++) {
    if (lastDp[s] > lastDp[bestPair]) bestPair = s;
  }
  const chosenIdx = new Int16Array(N);
  chosenIdx[N - 2] = Math.floor(bestPair / lastLc);
  chosenIdx[N - 1] = bestPair % lastLc;
  for (let i = N - 1; i >= 2; i--) {
    const Lc = layers[i].length;
    chosenIdx[i - 2] = back2[i][chosenIdx[i - 1] * Lc + chosenIdx[i]];
  }
  const path = Array.from(chosenIdx, (ci, i) => layers[i][ci]);

  return path.map((chosen, i) => {
    const top1 = items[i].hits?.[0];
    const rescued = Boolean(chosen.hit && top1 && chosen.char !== top1.char);
    if (rescued && chosen.hit && !items[i].hits.some((h) => h.id === chosen.hit.id)) {
      items[i].hits.push(chosen.hit);
    }
    return {
      lmBestChar: chosen.char || top1?.char || "",
      lmBestHitId: chosen.hit?.id ?? top1?.id,
      lmRescued: rescued,
    };
  });
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.cmd === "inspect_env") {
      // 首屏极轻并行探测：加载 200KB 书体元数据 + 282KB Rust WASM 预处理核 + WebGPU 探测 + OPFS 缓存扫描
      const metaPromise = ensureProtoMeta().then((m) => {
        self.postMessage({
          type: "meta_ready",
          totalPrototypes: m.total_prototypes,
          totalOriginalGlyphs: m.total_original_glyphs,
          distinctChars: m.distinct_chars,
          styles: m.styles,
          styleCounts: m.style_counts,
          styleOrigCounts: m.style_orig_counts,
        });
        return m;
      });
      const wasmPromise = ensureWasmCore();
      const gpuPromise = Promise.race([
        detectWebGPU(),
        new Promise((resolve) => setTimeout(() => resolve({ available: false, vramOk: false, vramReason: "显卡探测超时", adapterName: "", diag: "timeout" }), 2000)),
      ]);
      const cachePromise = (async () => {
        try {
          const manifest = await fetchManifest("./models_chunks/manifest.json");
          return await inspectModelCache(manifest);
        } catch (_) {
          return {
            totalGzBytes: 0, cachedGzBytes: 0, missingGzBytes: 0,
            totalChunks: 0, cachedChunks: 0, allCached: false, opfsModels: 0,
            hasUpdate: false, staleModels: [], buildId: "",
          };
        }
      })();

      const [, , gpu, cacheProbe, localComm] = await Promise.all([
        metaPromise,
        wasmPromise,
        gpuPromise,
        cachePromise,
        loadCommunityVectors().catch(() => []),
      ]);
      const communityCount = hydrateCommunityVecs(localComm);
      gpuHardwareAvailable = gpu.available;
      gpuVramOk = gpu.vramOk;
      gpuVramReason = gpu.vramReason || "";
      activeGpuName = gpu.adapterName || "";
      activeGpuDiag = gpu.diag || "";
      self.postMessage({
        type: "env_inspected",
        gpuHardwareAvailable: gpu.available,
        gpuVramOk: gpu.vramOk,
        gpuVramReason: gpu.vramReason || "",
        gpuAdapter: gpu.adapterName || "",
        gpuDiag: gpu.diag || "",
        threads: NUM_WASM_THREADS,
        crossOriginIsolated: !!self.crossOriginIsolated,
        cacheProbe,
        communityCount,
      });

      // 启动时后台轻量探针检查 D1 社区字库是否有新增向量（有新字形则推送红角标提醒用户点击同步；若仅删减则静默清理）
      const checkCommunityVersion = async () => {
        try {
          let fbMax = 0, helpMax = 0;
          for (const cv of communityVecs) {
            const cid = String(cv.cid || "");
            if (cid.startsWith("fb_")) {
              const n = parseInt(cid.slice(3), 10);
              if (Number.isFinite(n) && n > fbMax) fbMax = n;
            } else if (cid.startsWith("help_")) {
              const n = parseInt(cid.slice(5), 10);
              if (Number.isFinite(n) && n > helpMax) helpMax = n;
            }
          }
          const localCount = communityVecs.length;
          const localSig = `${fbMax}_${helpMax}_${localCount}`;
          const resp = await fetch("./api/community/version", { cache: "no-store" });
          if (!resp.ok) return;
          const ver = await resp.json();
          if (!ver || !ver.ok) return;
          const remoteCount = Number(ver.count || 0);
          const remoteFbMax = Number(ver.fb_max_id || 0);
          const remoteHelpMax = Number(ver.help_max_id || 0);
          const hasNew = remoteCount > 0 && (remoteFbMax > fbMax || remoteHelpMax > helpMax || remoteCount > localCount);
          if (hasNew) {
            self.postMessage({
              type: "community_update_available",
              localCount,
              remoteCount,
              delta: Math.max(1, remoteCount - localCount),
            });
          } else if (ver.sig && ver.sig !== localSig) {
            // 仅后台删减了旧条目，静默对齐本地缓存而不弹“新”角标
            const vResp = await fetch("./api/community/vectors", { cache: "no-cache" });
            if (!vResp.ok) return;
            const vData = await vResp.json();
            if (!vData || !vData.ok || !Array.isArray(vData.items)) return;
            await saveCommunityVectors(vData.items);
            const count = hydrateCommunityVecs(vData.items);
            self.postMessage({ type: "community_synced", count, silent: true });
          }
        } catch (_) {}
      };
      self._checkCommunityVersion = checkCommunityVersion;
      checkCommunityVersion();
    } else if (msg.cmd === "check_community_version") {
      if (typeof self._checkCommunityVersion === "function") {
        await self._checkCommunityVersion();
      }
    } else if (msg.cmd === "sync_community") {
      const [meta, resp] = await Promise.all([
        ensureProtoMeta(),
        fetch("./api/community/vectors", { cache: "no-cache" }),
      ]);
      if (!resp.ok) throw new Error(`同步社区字库失败 (HTTP ${resp.status})`);
      const data = await resp.json();
      if (!data.ok) throw new Error(data.error || "同步失败");
      await saveCommunityVectors(data.items || []);
      const count = hydrateCommunityVecs(data.items || [], meta);
      self.postMessage({ type: "community_synced", count, silent: false });
    } else if (msg.cmd === "detect") {
      const t0 = performance.now();
      const wasm = await ensureWasmCore();
      const rgba = new Uint8Array(msg.rgbaBuffer);
      const rawJson = wasm.detect_from_rgba(rgba, msg.width, msg.height);
      const parsed = JSON.parse(rawJson);
      self.postMessage({
        type: "detect_result",
        reqId: msg.reqId || 0,
        quad: parsed.quad,
        preview: parsed.preview,
        warning: parsed.warning,
        multiHint: Boolean(parsed.multi_hint),
        estChars: Number(parsed.est_chars) || 1,
        elapsedMs: Math.round(performance.now() - t0),
      });
    } else if (msg.cmd === "preview_crop") {
      // 用户拖动四角后点击「确定裁剪」：立即重新生成透视拉正彩色裁剪图 + 黑白预处理图
      const t0 = performance.now();
      const wasm = await ensureWasmCore();
      const rgba = new Uint8Array(msg.rgbaBuffer);
      const quadArr = msg.quad && msg.quad.length === 8 ? new Float32Array(msg.quad) : new Float32Array(0);
      const parsed = JSON.parse(wasm.preview_crop_from_rgba(rgba, msg.width, msg.height, quadArr));
      self.postMessage({
        type: "crop_preview_result",
        reqId: msg.reqId || 0,
        preview: parsed.preview,
        croppedPng: parsed.cropped_png,
        warning: parsed.warning,
        elapsedMs: Math.round(performance.now() - t0),
      });
    } else if (msg.cmd === "preload" || msg.cmd === "set_backend") {
      const info = await loadEngine(msg.backendPref || "wasm");
      self.postMessage({
        type: "ready",
        backend: info.backend,
        backendPref: info.backendPref,
        gpuHardwareAvailable,
        gpuVramOk,
        gpuVramReason,
        gpuAdapter: activeGpuName,
        threads: NUM_WASM_THREADS,
        cacheInfo: info.cacheInfo,
      });
    } else if (msg.cmd === "preview_crop_tuned") {
      const wasm = await ensureWasmCore();
      const rgba = new Uint8Array(msg.rgbaBuffer);
      const quadArr = msg.quad && msg.quad.length === 8 ? new Float32Array(msg.quad) : new Float32Array(0);
      const manualInvert = Boolean(msg.manualInvert);
      const threshBias = Number(msg.threshBias) || 0;
      let preview = "";
      let croppedPng = "";
      if (manualInvert || threshBias !== 0) {
        const tuned = await buildManualTunedBw(rgba, msg.width, msg.height, quadArr, manualInvert, threshBias);
        preview = tuned.preview;
        if (!preview) {
          const prep = wasm.prepare_input_from_rgba(rgba, msg.width, msg.height, quadArr);
          preview = prep.preview;
          croppedPng = prep.cropped_png;
          prep.free();
        }
      } else {
        const prep = wasm.prepare_input_from_rgba(rgba, msg.width, msg.height, quadArr);
        preview = prep.preview;
        croppedPng = prep.cropped_png;
        prep.free();
      }
      self.postMessage({
        type: "crop_preview_result",
        reqId: msg.reqId || 0,
        preview,
        croppedPng,
      });
    } else if (msg.cmd === "search") {
      if (!session || !protoIndex) {
        await loadEngine(msg.backendPref || activeBackendPref || "wasm");
      }
      const t0 = performance.now();
      const wasm = await ensureWasmCore();
      const meta = await ensureProtoMeta();
      const rgba = new Uint8Array(msg.rgbaBuffer);
      const quadArr = msg.quad && msg.quad.length === 8 ? new Float32Array(msg.quad) : new Float32Array(0);
      const manualInvert = Boolean(msg.manualInvert);
      const threshBias = Number(msg.threshBias) || 0;
      const prep = wasm.prepare_input_from_rgba(rgba, msg.width, msg.height, quadArr);
      let tensorData = prep.tensor;
      let tensorAlt = prep.tensor_alt;
      let preview = prep.preview;
      const croppedPng = prep.cropped_png;
      const warning = prep.warning;
      prep.free();

      if (manualInvert || threshBias !== 0) {
        const tuned = await buildManualTunedBw(rgba, msg.width, msg.height, quadArr, manualInvert, threshBias);
        tensorData = tuned.tensor;
        tensorAlt = null;
        if (tuned.preview) preview = tuned.preview;
      }

      // 准确率 #1：构造「自动反色+对比度拉伸灰度图」视图，与二值化视图按 0.65:0.35 融合
      const enableGrayTta = msg.enableGrayTta !== false;
      const grayWeight = Number.isFinite(msg.grayTtaWeight) ? Number(msg.grayTtaWeight) : 0.35;
      const grayBox = quadToBoundingBox(quadArr, msg.width, msg.height);
      const tensorGray = enableGrayTta
        ? makeAutoInvGrayTensorFromRgba(rgba, msg.width, msg.height, grayBox, manualInvert, threshBias)
        : null;
      const t1 = performance.now();

      const PLANE = 3 * 224 * 224;
      const useAlt = msg.enableTta !== false && tensorAlt && tensorAlt.length === PLANE;
      const useTta = useAlt || Boolean(tensorGray);
      const ttaOrigWeight = Number.isFinite(msg.ttaOrigWeight) ? Number(msg.ttaOrigWeight) : 0.45;
      const views = [tensorData];
      const altIdx = useAlt ? views.push(tensorAlt) - 1 : -1;
      const grayIdx = tensorGray ? views.push(tensorGray) - 1 : -1;
      const batchBuf = new Float32Array(views.length * PLANE);
      for (let vi = 0; vi < views.length; vi++) batchBuf.set(views[vi], vi * PLANE);

      const inputTensor = new ort.Tensor("float32", batchBuf, [views.length, 3, 224, 224]);
      const outputs = await session.run({ image: inputTensor });
      const embAll = new Float32Array(outputs.embedding.data);
      inputTensor.dispose();
      outputs.embedding.dispose();

      let bwEmb = l2NormalizeInPlace(new Float32Array(embAll.subarray(0, 512)));
      if (altIdx >= 0) {
        bwEmb = blendNormalizedEmbs(bwEmb, embAll.subarray(altIdx * 512, (altIdx + 1) * 512), ttaOrigWeight);
      }
      let emb = bwEmb;
      if (grayIdx >= 0 && grayWeight > 0) {
        const grayEmb = embAll.subarray(grayIdx * 512, (grayIdx + 1) * 512);
        emb = blendNormalizedEmbs(bwEmb, grayEmb, 1.0 - grayWeight);
      }
      const t2 = performance.now();

      const styleFilter = resolveStyleFilter(meta, msg.style || "");
      const topk = Math.max(1, Math.min(60, Number(msg.topk) || 24));
      const scanK = Math.max(200, topk * 6);
      const searchEmb = applyAbtt(emb, meta);
      const rawHits = JSON.parse(protoIndex.search_mask(searchEmb, styleFilter.mask, scanK));
      const baseHits = rawHits.map((h) => ({
        id: h.id,
        char: meta.chars[h.id],
        style: meta.styles[meta.style_ids[h.id]],
        dynasty: meta.dynasties[meta.dynasty_ids[h.id]] || "unknown",
        score: h.score,
        isCommunity: false,
        imageUrl: `./glyph/${h.id}?v=${activeProtoVer}`,
      }));
      const commHits = searchCommunityVecs(searchEmb, styleFilter.allowedSet, topk);
      const mergedAll = baseHits.concat(commHits).sort((a, b) => b.score - a.score);
      const hits = rerankByCharConsensus(mergedAll, topk);
      // D1 众包回传保留原始 ep14.pt 空间向量，跨版本永不失配
      const vecPacked = quantizeFloat512ToB64(emb);
      const t3 = performance.now();

      self.postMessage({
        type: "search_result",
        preview,
        croppedPng,
        warning,
        hits,
        ttaUsed: useTta,
        embedding_b64: vecPacked.embedding_b64,
        inv_norm: vecPacked.inv_norm,
        communityCount: communityVecs.length,
        backend: activeBackend,
        timings: {
          preprocessMs: Math.round(t1 - t0),
          inferMs: Math.round(t2 - t1),
          searchMs: Math.round(t3 - t2),
          totalMs: Math.round(t3 - t0),
        },
      });
    } else if (msg.cmd === "crop_roi") {
      const wasm = await ensureWasmCore();
      const rgba = new Uint8Array(msg.rgbaBuffer);
      const quadArr = new Float32Array(msg.quad || []);
      const pngDataUrl = wasm.crop_quad_to_png(rgba, msg.width, msg.height, quadArr);
      self.postMessage({ type: "roi_cropped", reqId: msg.reqId || 0, pngDataUrl });
    } else if (msg.cmd === "detect_multi") {
      const t0 = performance.now();
      // 并行预热 1.17MB 古汉语二元字频表，供批量识别 0ms 语境纠偏
      ensureClassicalBigram();
      const det = await ensureCharDetSession(msg.backendPref || activeBackendPref || "wasm");
      const rgba = new Uint8Array(msg.rgbaBuffer);
      const srcW = msg.width | 0;
      const srcH = msg.height | 0;
      // 估计全图亮度：深底白字拓片自动开启双极性切字增强
      let lumaSum = 0, lumaCnt = 0;
      for (let y = 0; y < srcH; y += 4) {
        const row = y * srcW * 4;
        for (let x = 0; x < srcW; x += 4) {
          const p = row + x * 4;
          lumaSum += 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
          lumaCnt++;
        }
      }
      const meanLuma = lumaSum / Math.max(1, lumaCnt);

      async function runFullDet(bufRgba) {
        const { tensorData, scale, ox, oy } = letterboxRgbaTo512(bufRgba, srcW, srcH);
        const inp = new ort.Tensor("float32", tensorData, [1, 3, DET_IMG_SIZE, DET_IMG_SIZE]);
        const out = await det.run({ image: inp });
        const hm = new Float32Array(out.hm.data);
        const whReg = new Float32Array(out.wh_reg.data);
        inp.dispose();
        out.hm.dispose();
        out.wh_reg.dispose();
        return decodeCenterNetBoxes(hm, whReg, scale, ox, oy, srcW, srcH, 0.15, 0.45, 64);
      }

      const t1 = performance.now();
      let rawCands = await runFullDet(rgba);
      const dualPolarity = meanLuma < 122;
      if (dualPolarity) {
        const invRgba = new Uint8Array(rgba.length);
        for (let i = 0; i < rgba.length; i += 4) {
          invRgba[i] = 255 - rgba[i];
          invRgba[i + 1] = 255 - rgba[i + 1];
          invRgba[i + 2] = 255 - rgba[i + 2];
          invRgba[i + 3] = 255;
        }
        rawCands = rawCands.concat(await runFullDet(invRgba));
        rawCands.sort((a, b) => b.score - a.score);
      }
      const candidates = [];
      for (const b of rawCands) {
        const areaB = Math.max(1, (b.x1 - b.x0) * (b.y1 - b.y0));
        let dup = false;
        for (const k of candidates) {
          const ix = Math.max(0, Math.min(b.x1, k.x1) - Math.max(b.x0, k.x0));
          const iy = Math.max(0, Math.min(b.y1, k.y1) - Math.max(b.y0, k.y0));
          const inter = ix * iy;
          if (inter <= 0) continue;
          const areaK = Math.max(1, (k.x1 - k.x0) * (k.y1 - k.y0));
          if (inter / Math.max(1, areaB + areaK - inter) > 0.42 || inter / Math.min(areaB, areaK) > 0.72) {
            dup = true;
            break;
          }
        }
        if (!dup) {
          candidates.push(b);
          if (candidates.length >= 64) break;
        }
      }
      const t2 = performance.now();
      self.postMessage({
        type: "multi_detect_result",
        reqId: msg.reqId || 0,
        candidates,
        dualPolarity,
        detBackend,
        elapsedMs: Math.round(t2 - t0),
        inferMs: Math.round(t2 - t1),
      });
    } else if (msg.cmd === "detect_multi_region") {
      const t0 = performance.now();
      ensureClassicalBigram();
      const det = await ensureCharDetSession(msg.backendPref || activeBackendPref || "wasm");
      const rgba = new Uint8Array(msg.rgbaBuffer);
      const srcW = msg.width | 0;
      const srcH = msg.height | 0;
      const reg = msg.region || { x0: 0, y0: 0, x1: srcW, y1: srcH };
      const ux0 = Math.max(0, Math.min(srcW - 1, Math.floor(Math.min(reg.x0, reg.x1))));
      const uy0 = Math.max(0, Math.min(srcH - 1, Math.floor(Math.min(reg.y0, reg.y1))));
      const ux1 = Math.max(ux0 + 1, Math.min(srcW, Math.ceil(Math.max(reg.x0, reg.x1))));
      const uy1 = Math.max(uy0 + 1, Math.min(srcH, Math.ceil(Math.max(reg.y0, reg.y1))));
      const pad = 8;
      const rx0 = Math.max(0, ux0 - pad);
      const ry0 = Math.max(0, uy0 - pad);
      const rx1 = Math.min(srcW, ux1 + pad);
      const ry1 = Math.min(srcH, uy1 + pad);
      const subW = Math.max(1, rx1 - rx0);
      const subH = Math.max(1, ry1 - ry0);

      // 提取高分辨率子区域 RGBA 并估计平均亮度
      const subRgba = new Uint8Array(subW * subH * 4);
      let lumaSum = 0;
      for (let y = 0; y < subH; y++) {
        const srcRow = ((ry0 + y) * srcW + rx0) * 4;
        const dstRow = y * subW * 4;
        subRgba.set(rgba.subarray(srcRow, srcRow + subW * 4), dstRow);
        for (let x = 0; x < subW; x += 2) {
          const p = dstRow + x * 4;
          lumaSum += 0.299 * subRgba[p] + 0.587 * subRgba[p + 1] + 0.114 * subRgba[p + 2];
        }
      }
      const meanLuma = lumaSum / Math.max(1, Math.ceil(subW / 2) * subH);

      async function runDetOnPatch(patchRgba) {
        const { tensorData, scale, ox, oy } = letterboxRgbaTo512(patchRgba, subW, subH);
        const inp = new ort.Tensor("float32", tensorData, [1, 3, DET_IMG_SIZE, DET_IMG_SIZE]);
        const out = await det.run({ image: inp });
        const hm = new Float32Array(out.hm.data);
        const whReg = new Float32Array(out.wh_reg.data);
        inp.dispose();
        out.hm.dispose();
        out.wh_reg.dispose();
        return decodeCenterNetBoxes(hm, whReg, scale, ox, oy, subW, subH, 0.14, 0.45, 64);
      }

      const t1 = performance.now();
      let rawList = await runDetOnPatch(subRgba);

      // 深底白字拓片（甲骨/金文/碑拓）自动叠加反色双极性检测，大幅提升细小拓片字召回率
      if (meanLuma < 122) {
        const invRgba = new Uint8Array(subRgba.length);
        for (let i = 0; i < subRgba.length; i += 4) {
          invRgba[i] = 255 - subRgba[i];
          invRgba[i + 1] = 255 - subRgba[i + 1];
          invRgba[i + 2] = 255 - subRgba[i + 2];
          invRgba[i + 3] = 255;
        }
        const invList = await runDetOnPatch(invRgba);
        rawList = rawList.concat(invList);
      }

      // 映回全图坐标，仅保留中心点落在用户框选区域内的候选，并做跨极性 IoU NMS 去重
      const mapped = [];
      for (const b of rawList) {
        const gx0 = Math.max(0, Math.min(srcW - 1, b.x0 + rx0));
        const gy0 = Math.max(0, Math.min(srcH - 1, b.y0 + ry0));
        const gx1 = Math.max(gx0 + 1, Math.min(srcW, b.x1 + rx0));
        const gy1 = Math.max(gy0 + 1, Math.min(srcH, b.y1 + ry0));
        const cx = (gx0 + gx1) * 0.5;
        const cy = (gy0 + gy1) * 0.5;
        if (cx < ux0 || cx > ux1 || cy < uy0 || cy > uy1) continue;
        mapped.push({ x0: gx0, y0: gy0, x1: gx1, y1: gy1, score: b.score });
      }
      mapped.sort((a, b) => b.score - a.score);
      const candidates = [];
      for (const b of mapped) {
        const areaB = Math.max(1, (b.x1 - b.x0) * (b.y1 - b.y0));
        let dup = false;
        for (const k of candidates) {
          const ix = Math.max(0, Math.min(b.x1, k.x1) - Math.max(b.x0, k.x0));
          const iy = Math.max(0, Math.min(b.y1, k.y1) - Math.max(b.y0, k.y0));
          const inter = ix * iy;
          if (inter <= 0) continue;
          const areaK = Math.max(1, (k.x1 - k.x0) * (k.y1 - k.y0));
          const iou = inter / Math.max(1, areaB + areaK - inter);
          if (iou > 0.42 || inter / Math.min(areaB, areaK) > 0.72) {
            dup = true;
            break;
          }
        }
        if (!dup) {
          candidates.push(b);
          if (candidates.length >= 48) break;
        }
      }
      const t2 = performance.now();
      self.postMessage({
        type: "multi_region_detect_result",
        reqId: msg.reqId || 0,
        candidates,
        dualPolarity: meanLuma < 122,
        elapsedMs: Math.round(t2 - t0),
        inferMs: Math.round(t2 - t1),
      });
    } else if (msg.cmd === "search_multi") {
      if (!session || !protoIndex) {
        await loadEngine(msg.backendPref || activeBackendPref || "wasm");
      }
      const t0 = performance.now();
      const wasm = await ensureWasmCore();
      const meta = await ensureProtoMeta();
      const rgba = new Uint8Array(msg.rgbaBuffer);
      const boxes = (msg.boxes || []).slice(0, 36);
      const styleStr = msg.style || "";
      const styleFilter = resolveStyleFilter(meta, styleStr);
      const topk = Math.max(1, Math.min(24, Number(msg.topk) || 12));
      const orderMode = msg.orderMode || "vertical-rl";
      const enableTta = msg.enableTta !== false;
      const enableGrayTta = msg.enableGrayTta !== false;
      const grayWeight = Number.isFinite(msg.grayTtaWeight) ? Number(msg.grayTtaWeight) : 0.35;
      const ttaOrigWeight = Number.isFinite(msg.ttaOrigWeight) ? Number(msg.ttaOrigWeight) : 0.45;
      const enableScriptConsistency = msg.enableScriptConsistency !== false;
      const total = boxes.length;
      const items = [];
      const BATCH = 4;
      const PLANE = 3 * 224 * 224;
      let preprocessMs = 0, inferMs = 0, searchMs = 0;
      let ttaCount = 0;

      for (let start = 0; start < total; start += BATCH) {
        const chunk = boxes.slice(start, start + BATCH);
        const B = chunk.length;
        const tp0 = performance.now();
        const preps = [];
        const altIndices = [];
        const grayTensors = [];
        for (let bi = 0; bi < B; bi++) {
          const b = chunk[bi];
          const prep = wasm.prepare_char_box_from_rgba(rgba, msg.width, msg.height, b.x0, b.y0, b.x1, b.y1);
          const tensor = prep.tensor;
          const tensorAlt = prep.tensor_alt;
          const ar = prep.aspect_ratio;
          const hasAlt = enableTta && tensorAlt && tensorAlt.length === PLANE;
          if (hasAlt) {
            altIndices.push(bi);
            ttaCount += 1;
          }
          const tGray = enableGrayTta ? makeAutoInvGrayTensorFromRgba(rgba, msg.width, msg.height, b, false, 0) : null;
          if (tGray) grayTensors.push({ bi, tensor: tGray });
          preps.push({
            tensor,
            tensorAlt: hasAlt ? tensorAlt : null,
            ar,
            hasAlt,
            preview: prep.preview,
            croppedPng: prep.cropped_png,
          });
          prep.free();
        }
        const totalViews = B + altIndices.length + grayTensors.length;
        const batchTensor = new Float32Array(totalViews * PLANE);
        for (let bi = 0; bi < B; bi++) {
          batchTensor.set(preps[bi].tensor, bi * PLANE);
        }
        for (let ai = 0; ai < altIndices.length; ai++) {
          batchTensor.set(preps[altIndices[ai]].tensorAlt, (B + ai) * PLANE);
        }
        const grayBaseOffset = B + altIndices.length;
        for (let gi = 0; gi < grayTensors.length; gi++) {
          batchTensor.set(grayTensors[gi].tensor, (grayBaseOffset + gi) * PLANE);
        }
        const tp1 = performance.now();
        preprocessMs += tp1 - tp0;

        const inputTensor = new ort.Tensor("float32", batchTensor, [totalViews, 3, 224, 224]);
        const outputs = await session.run({ image: inputTensor });
        const embAll = new Float32Array(outputs.embedding.data);
        inputTensor.dispose();
        outputs.embedding.dispose();
        const ti1 = performance.now();
        inferMs += ti1 - tp1;

        const altPosByBi = new Map();
        altIndices.forEach((bi, ai) => altPosByBi.set(bi, B + ai));
        const grayPosByBi = new Map();
        grayTensors.forEach((g, gi) => grayPosByBi.set(g.bi, grayBaseOffset + gi));

        // 无书体限定时扩大初筛池（scanK=160，额外耗时 ~0ms），便于同碑书体家族共识与 LSE 共识发掘排在稍后的真原型
        const scanK = !styleStr && enableScriptConsistency ? 160 : Math.max(topk * 4, 80);

        for (let bi = 0; bi < B; bi++) {
          const rawOrig = embAll.subarray(bi * 512, (bi + 1) * 512);
          let bwEmb;
          if (altPosByBi.has(bi)) {
            const altSlot = altPosByBi.get(bi);
            const rawAlt = embAll.subarray(altSlot * 512, (altSlot + 1) * 512);
            bwEmb = blendNormalizedEmbs(rawOrig, rawAlt, ttaOrigWeight);
          } else {
            bwEmb = l2NormalizeInPlace(new Float32Array(rawOrig));
          }
          let emb = bwEmb;
          if (grayPosByBi.has(bi) && grayWeight > 0) {
            const gSlot = grayPosByBi.get(bi);
            const rawGray = embAll.subarray(gSlot * 512, (gSlot + 1) * 512);
            emb = blendNormalizedEmbs(bwEmb, rawGray, 1.0 - grayWeight);
          }
          const searchEmb = applyAbtt(emb, meta);
          const rawHits = JSON.parse(protoIndex.search_mask(searchEmb, styleFilter.mask, scanK));
          const baseHits = rawHits.map((h) => ({
            id: h.id,
            char: meta.chars[h.id],
            style: meta.styles[meta.style_ids[h.id]],
            dynasty: meta.dynasties[meta.dynasty_ids[h.id]] || "unknown",
            score: h.score,
            isCommunity: false,
            imageUrl: `./glyph/${h.id}?v=${activeProtoVer}`,
          }));
          const commHits = searchCommunityVecs(searchEmb, styleFilter.allowedSet, topk);
          const merged = baseHits.concat(commHits).sort((a, b) => b.score - a.score);
          const hits = rerankByCharConsensus(merged, topk);
          const vecPacked = quantizeFloat512ToB64(emb);
          items.push({
            idx: start + bi,
            box: chunk[bi],
            preview: preps[bi].preview,
            croppedPng: preps[bi].croppedPng,
            ttaUsed: preps[bi].hasAlt,
            aspectRatio: +(preps[bi].ar || 1).toFixed(2),
            embedding_b64: vecPacked.embedding_b64,
            inv_norm: vecPacked.inv_norm,
            hits,
            _viterbiPool: merged.slice(0, scanK),
          });
        }
        searchMs += performance.now() - ti1;
        self.postMessage({ type: "multi_progress", done: Math.min(total, start + B), total });
      }

      // 同碑书体家族共识（仅在用户选择「全部书体（不限）」时自动生效）
      let dominantFamily = null;
      if (!styleStr && enableScriptConsistency) {
        dominantFamily = applyWholeSteleScriptConsistency(items, topk, {
          consistencyBoost: msg.consistencyBoost,
          sealCoreExtra: msg.sealCoreExtra,
        });
      }

      // 古汉语三元+四分域二阶 Viterbi 语境重排（自动断列/断行 + 领域感知 + 低置信防护，< 2ms）
      const lm = await ensureClassicalBigram();
      const lmRes = viterbiRerankMulti(items, lm, {
        orderMode,
        styleStr,
        enableBreakGuard: msg.enableBreakGuard !== false,
        enableConfGuard: msg.enableConfGuard !== false,
      });
      items.forEach((it, idx) => {
        delete it._viterbiPool;
        it.lmBestChar = lmRes[idx].lmBestChar;
        it.lmBestHitId = lmRes[idx].lmBestHitId;
        it.lmRescued = lmRes[idx].lmRescued;
      });

      self.postMessage({
        type: "multi_search_result",
        items,
        lmReady: Boolean(lm),
        dominantFamily,
        ttaCount,
        communityCount: communityVecs.length,
        backend: activeBackend,
        timings: {
          preprocessMs: Math.round(preprocessMs),
          inferMs: Math.round(inferMs),
          searchMs: Math.round(searchMs),
          totalMs: Math.round(performance.now() - t0),
        },
      });
    } else if (msg.cmd === "search_multi_one") {
      // 单框快速重识（仅对用户微调过的那 1 个字框推理，~50ms，无需整批重算！）
      if (!session || !protoIndex) {
        await loadEngine(msg.backendPref || activeBackendPref || "wasm");
      }
      const t0 = performance.now();
      const wasm = await ensureWasmCore();
      const meta = await ensureProtoMeta();
      const rgba = new Uint8Array(msg.rgbaBuffer);
      const b = msg.box;
      const styleStr = msg.style || "";
      const styleFilter = resolveStyleFilter(meta, styleStr);
      const topk = Math.max(1, Math.min(24, Number(msg.topk) || 12));
      const enableTta = msg.enableTta !== false;
      const enableGrayTta = msg.enableGrayTta !== false;
      const grayWeight = Number.isFinite(msg.grayTtaWeight) ? Number(msg.grayTtaWeight) : 0.35;
      const ttaOrigWeight = Number.isFinite(msg.ttaOrigWeight) ? Number(msg.ttaOrigWeight) : 0.45;
      const PLANE = 3 * 224 * 224;

      const prep = wasm.prepare_char_box_from_rgba(rgba, msg.width, msg.height, b.x0, b.y0, b.x1, b.y1);
      const tensor = prep.tensor;
      const tensorAlt = prep.tensor_alt;
      const ar = prep.aspect_ratio;
      const preview = prep.preview;
      const croppedPng = prep.cropped_png;
      prep.free();

      const hasAlt = enableTta && tensorAlt && tensorAlt.length === PLANE;
      const tGray = enableGrayTta ? makeAutoInvGrayTensorFromRgba(rgba, msg.width, msg.height, b, false, 0) : null;
      const views = [tensor];
      const altIdx = hasAlt ? views.push(tensorAlt) - 1 : -1;
      const grayIdx = tGray ? views.push(tGray) - 1 : -1;
      const batchBuf = new Float32Array(views.length * PLANE);
      for (let vi = 0; vi < views.length; vi++) batchBuf.set(views[vi], vi * PLANE);

      const inputTensor = new ort.Tensor("float32", batchBuf, [views.length, 3, 224, 224]);
      const outputs = await session.run({ image: inputTensor });
      const embAll = new Float32Array(outputs.embedding.data);
      inputTensor.dispose();
      outputs.embedding.dispose();

      let bwEmb = l2NormalizeInPlace(new Float32Array(embAll.subarray(0, 512)));
      if (altIdx >= 0) {
        bwEmb = blendNormalizedEmbs(bwEmb, embAll.subarray(altIdx * 512, (altIdx + 1) * 512), ttaOrigWeight);
      }
      let emb = bwEmb;
      if (grayIdx >= 0 && grayWeight > 0) {
        emb = blendNormalizedEmbs(bwEmb, embAll.subarray(grayIdx * 512, (grayIdx + 1) * 512), 1.0 - grayWeight);
      }
      const searchEmb = applyAbtt(emb, meta);
      const scanK = Math.max(topk * 4, 120);
      const rawHits = JSON.parse(protoIndex.search_mask(searchEmb, styleFilter.mask, scanK));
      const baseHits = rawHits.map((h) => ({
        id: h.id,
        char: meta.chars[h.id],
        style: meta.styles[meta.style_ids[h.id]],
        dynasty: meta.dynasties[meta.dynasty_ids[h.id]] || "unknown",
        score: h.score,
        isCommunity: false,
        imageUrl: `./glyph/${h.id}?v=${activeProtoVer}`,
      }));
      const commHits = searchCommunityVecs(searchEmb, styleFilter.allowedSet, topk);
      const merged = baseHits.concat(commHits).sort((a, b2) => b2.score - a.score);
      const hits = rerankByCharConsensus(merged, topk);
      const vecPacked = quantizeFloat512ToB64(emb);
      self.postMessage({
        type: "multi_one_result",
        reqId: msg.reqId || 0,
        boxIdx: msg.boxIdx,
        item: {
          idx: msg.boxIdx,
          box: b,
          preview,
          croppedPng,
          ttaUsed: hasAlt || Boolean(tGray),
          aspectRatio: +(ar || 1).toFixed(2),
          embedding_b64: vecPacked.embedding_b64,
          inv_norm: vecPacked.inv_norm,
          hits,
        },
        elapsedMs: Math.round(performance.now() - t0),
      });
    } else if (msg.cmd === "rerank_multi") {
      // 用户切换阅读顺序或删框后，用已有视觉候选 + 坐标 1ms 重跑断列感知 Viterbi 语境纠偏
      const lm = await ensureClassicalBigram();
      const items = (msg.items || []).map((it) => ({
        box: it.box || null,
        hits: it.hits || [],
      }));
      const lmRes = viterbiRerankMulti(items, lm, {
        orderMode: msg.orderMode || "vertical-rl",
        styleStr: msg.style || "",
      });
      self.postMessage({ type: "multi_reranked", reqId: msg.reqId || 0, lmRes });
    } else if (msg.cmd === "clear_cache") {
      if (session) { try { await session.release(); } catch (_) {} session = null; }
      if (detSession) { try { await detSession.release(); } catch (_) {} detSession = null; detBackend = null; }
      if (protoIndex) { try { protoIndex.free(); } catch (_) {} protoIndex = null; }
      lastCacheInfo = null;
      await clearModelCache();
      self.postMessage({ type: "cache_cleared" });
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      message: err?.message || String(err),
    });
  }
};
