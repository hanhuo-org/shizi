/**
 * 汉火字形 (xing.hanhuo.org) —— OPFS 完整单文件零合并缓存 + 多线程分片下载 + 单事务并发 IndexedDB 回退引擎
 */
const DB_NAME = "hanzi_xing_cache_v1";
const DB_VERSION = 1;
const STORE_CHUNKS = "chunks";
const STORE_META = "meta";
const OPFS_DIR_NAME = "hanzi_xing_models_v1";

async function getOpfsModelsDir(create = false) {
  try {
    if (typeof navigator === "undefined" || !navigator.storage || typeof navigator.storage.getDirectory !== "function") {
      return null;
    }
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(OPFS_DIR_NAME, { create });
  } catch (_) {
    return null;
  }
}

function opfsModelFileName(modelKey, version) {
  return `${modelKey}.${version}.bin`;
}

async function opfsCheckModelFile(modelKey, version, expectedSize) {
  try {
    const dir = await getOpfsModelsDir(false);
    if (!dir) return false;
    const fh = await dir.getFileHandle(opfsModelFileName(modelKey, version), { create: false });
    const file = await fh.getFile();
    return file.size === expectedSize;
  } catch (_) {
    return false;
  }
}

async function opfsHasStaleModelFile(modelKey, currentVersion) {
  try {
    const dir = await getOpfsModelsDir(false);
    if (!dir) return false;
    const prefix = `${modelKey}.`;
    const curName = opfsModelFileName(modelKey, currentVersion);
    for await (const [name] of dir.entries()) {
      if (name.startsWith(prefix) && name !== curName) {
        return true;
      }
    }
    return false;
  } catch (_) {
    return false;
  }
}

async function opfsReadCompleteModel(modelKey, version, expectedSize) {
  try {
    const dir = await getOpfsModelsDir(false);
    if (!dir) return null;
    const fh = await dir.getFileHandle(opfsModelFileName(modelKey, version), { create: false });
    if (typeof fh.createSyncAccessHandle === "function") {
      const access = await fh.createSyncAccessHandle();
      try {
        const size = access.getSize();
        if (size !== expectedSize) return null;
        const out = new Uint8Array(size);
        let offset = 0;
        while (offset < size) {
          const n = access.read(out.subarray(offset), { at: offset });
          if (n <= 0) return null;
          offset += n;
        }
        return out;
      } finally {
        try { access.close(); } catch (_) {}
      }
    }
    const file = await fh.getFile();
    if (file.size !== expectedSize) return null;
    return new Uint8Array(await file.arrayBuffer());
  } catch (_) {
    return null;
  }
}

async function opfsWriteCompleteModel(modelKey, version, bytes) {
  try {
    const dir = await getOpfsModelsDir(true);
    if (!dir) return false;
    const targetName = opfsModelFileName(modelKey, version);
    let wrote = false;
    const fh = await dir.getFileHandle(targetName, { create: true });
    if (typeof fh.createSyncAccessHandle === "function") {
      const access = await fh.createSyncAccessHandle();
      try {
        access.truncate(0);
        let offset = 0;
        while (offset < bytes.byteLength) {
          const n = access.write(bytes.subarray(offset), { at: offset });
          if (n <= 0) break;
          offset += n;
        }
        access.flush();
        wrote = access.getSize() === bytes.byteLength;
      } finally {
        try { access.close(); } catch (_) {}
      }
    } else if (typeof fh.createWritable === "function") {
      const w = await fh.createWritable();
      try {
        await w.write(bytes);
        await w.close();
        const f = await fh.getFile();
        wrote = f.size === bytes.byteLength;
      } catch (_) {
        try { await w.abort(); } catch (__) {}
        wrote = false;
      }
    }
    if (!wrote) {
      try { await dir.removeEntry(targetName); } catch (_) {}
      return false;
    }
    try {
      const prefix = `${modelKey}.`;
      for await (const [name] of dir.entries()) {
        if (name.startsWith(prefix) && name !== targetName) {
          try { await dir.removeEntry(name); } catch (_) {}
        }
      }
    } catch (_) {}
    return true;
  } catch (_) {
    return false;
  }
}

async function opfsClearAllModels() {
  try {
    if (typeof navigator === "undefined" || !navigator.storage || typeof navigator.storage.getDirectory !== "function") {
      return;
    }
    const root = await navigator.storage.getDirectory();
    await root.removeEntry(OPFS_DIR_NAME, { recursive: true });
  } catch (_) {}
}

function openCacheDB() {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_CHUNKS)) {
          const store = db.createObjectStore(STORE_CHUNKS, { keyPath: "key" });
          store.createIndex("by_model", "modelKey", { unique: false });
        }
        if (!db.objectStoreNames.contains(STORE_META)) {
          db.createObjectStore(STORE_META, { keyPath: "modelKey" });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch (_) {
      resolve(null);
    }
  });
}

function idbPut(db, storeName, value) {
  if (!db) return Promise.resolve(false);
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(storeName, "readwrite");
      tx.objectStore(storeName).put(value);
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => resolve(false);
    } catch (_) {
      resolve(false);
    }
  });
}

function idbListKeysForModel(db, modelKey) {
  if (!db) return Promise.resolve([]);
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_CHUNKS, "readonly");
      const req = tx.objectStore(STORE_CHUNKS).index("by_model").getAllKeys(modelKey);
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => resolve([]);
    } catch (_) {
      resolve([]);
    }
  });
}

function idbGetMany(db, storeName, keys) {
  if (!db || !keys || keys.length === 0) return Promise.resolve(new Map());
  return new Promise((resolve) => {
    const out = new Map();
    try {
      const tx = db.transaction(storeName, "readonly");
      const store = tx.objectStore(storeName);
      for (const k of keys) {
        const req = store.get(k);
        req.onsuccess = () => { if (req.result) out.set(k, req.result); };
      }
      tx.oncomplete = () => resolve(out);
      tx.onerror = () => resolve(out);
      tx.onabort = () => resolve(out);
    } catch (_) {
      resolve(out);
    }
  });
}

function idbDeleteKeys(db, keys) {
  if (!db || !keys || keys.length === 0) return Promise.resolve();
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_CHUNKS, "readwrite");
      const store = tx.objectStore(STORE_CHUNKS);
      for (const k of keys) store.delete(k);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    } catch (_) {
      resolve();
    }
  });
}

export async function saveCommunityVectors(items) {
  const db = await openCacheDB();
  if (!db) return false;
  try {
    await idbPut(db, STORE_META, {
      modelKey: "__community_vectors__",
      items: Array.isArray(items) ? items : [],
      updatedAt: Date.now(),
    });
    return true;
  } finally {
    db.close();
  }
}

export async function loadCommunityVectors() {
  const db = await openCacheDB();
  if (!db) return [];
  try {
    return await new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_META, "readonly");
        const req = tx.objectStore(STORE_META).get("__community_vectors__");
        req.onsuccess = () => resolve(req.result?.items || []);
        req.onerror = () => resolve([]);
      } catch (_) {
        resolve([]);
      }
    });
  } finally {
    db.close();
  }
}

export async function saveStudioSession(session) {
  const db = await openCacheDB();
  if (!db) return false;
  try {
    await idbPut(db, STORE_META, {
      modelKey: "__studio_session_v1__",
      session: session || null,
      updatedAt: Date.now(),
    });
    return true;
  } finally {
    db.close();
  }
}

export async function loadStudioSession() {
  const db = await openCacheDB();
  if (!db) return null;
  try {
    return await new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_META, "readonly");
        const req = tx.objectStore(STORE_META).get("__studio_session_v1__");
        req.onsuccess = () => resolve(req.result?.session || null);
        req.onerror = () => resolve(null);
      } catch (_) {
        resolve(null);
      }
    });
  } finally {
    db.close();
  }
}

export async function clearStudioSession() {
  const db = await openCacheDB();
  if (!db) return false;
  try {
    return await new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_META, "readwrite");
        tx.objectStore(STORE_META).delete("__studio_session_v1__");
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
      } catch (_) {
        resolve(false);
      }
    });
  } finally {
    db.close();
  }
}

export async function clearModelCache() {
  await opfsClearAllModels();
  const db = await openCacheDB();
  if (!db) return true;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction([STORE_CHUNKS, STORE_META], "readwrite");
      tx.objectStore(STORE_CHUNKS).clear();
      tx.objectStore(STORE_META).clear();
      tx.oncomplete = () => { db.close(); resolve(true); };
      tx.onerror = () => { db.close(); resolve(false); };
    } catch (_) {
      try { db.close(); } catch (__) {}
      resolve(false);
    }
  });
}

async function gunzipUint8(gzBytes) {
  if (gzBytes.length < 2 || gzBytes[0] !== 0x1f || gzBytes[1] !== 0x8b) return gzBytes;
  if (typeof DecompressionStream === "undefined") {
    throw new Error("当前浏览器不支持原生 DecompressionStream('gzip')，请升级浏览器");
  }
  const ds = new DecompressionStream("gzip");
  const buf = await new Response(new Blob([gzBytes]).stream().pipeThrough(ds)).arrayBuffer();
  return new Uint8Array(buf);
}

async function sha256Hex(u8) {
  if (typeof crypto === "undefined" || !crypto.subtle) return null;
  const hashBuf = await crypto.subtle.digest("SHA-256", u8);
  const arr = new Uint8Array(hashBuf);
  let hex = "";
  for (let i = 0; i < arr.length; i++) hex += arr[i].toString(16).padStart(2, "0");
  return hex;
}

async function fetchChunkWithProgress(url, onDelta, retries = 3) {
  let lastErr = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    let reported = 0;
    try {
      const resp = await fetch(url, { cache: "force-cache" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status} (${url})`);
      if (!resp.body || typeof resp.body.getReader !== "function") {
        const u8 = new Uint8Array(await resp.arrayBuffer());
        onDelta(u8.byteLength);
        return u8;
      }
      const reader = resp.body.getReader();
      const parts = [];
      let total = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value && value.byteLength > 0) {
          parts.push(value);
          total += value.byteLength;
          reported += value.byteLength;
          onDelta(value.byteLength);
        }
      }
      const merged = new Uint8Array(total);
      let off = 0;
      for (const p of parts) { merged.set(p, off); off += p.byteLength; }
      return merged;
    } catch (err) {
      lastErr = err;
      if (reported > 0) onDelta(-reported);
      if (attempt < retries) await new Promise((r) => setTimeout(r, 400 * Math.pow(2, attempt)));
    }
  }
  throw lastErr || new Error(`分片下载失败: ${url}`);
}

export async function fetchManifest(manifestUrl = "./models_chunks/manifest.json") {
  const resp = await fetch(manifestUrl, { cache: "no-cache" });
  if (!resp.ok) throw new Error(`无法加载分片清单 ${manifestUrl} (HTTP ${resp.status})`);
  return await resp.json();
}

export async function inspectModelCache(manifest) {
  if (!manifest || !manifest.models) {
    return {
      totalGzBytes: 0, cachedGzBytes: 0, missingGzBytes: 0,
      totalChunks: 0, cachedChunks: 0, allCached: false, opfsModels: 0,
      hasUpdate: false, staleModels: [], buildId: "",
    };
  }
  const db = await openCacheDB();
  let totalGzBytes = 0, cachedGzBytes = 0, totalChunks = 0, cachedChunks = 0, opfsModels = 0;
  const staleModels = [];
  try {
    for (const [modelKey, mInfo] of Object.entries(manifest.models)) {
      if (!mInfo || !Array.isArray(mInfo.chunks)) continue;
      const modelGz = mInfo.chunks.reduce((a, c) => a + (c.gzSize || 0), 0);
      totalGzBytes += modelGz;
      totalChunks += mInfo.chunks.length;
      if (await opfsCheckModelFile(modelKey, mInfo.version, mInfo.totalSize)) {
        cachedGzBytes += modelGz;
        cachedChunks += mInfo.chunks.length;
        opfsModels += 1;
        continue;
      }
      const existingKeys = new Set(await idbListKeysForModel(db, modelKey));
      let modelCached = 0;
      for (const ch of mInfo.chunks) {
        if (existingKeys.has(`${modelKey}:${mInfo.version}:${ch.index}`)) {
          cachedGzBytes += ch.gzSize || 0;
          cachedChunks += 1;
          modelCached += 1;
        }
      }
      if (modelCached < mInfo.chunks.length) {
        const hasStaleOpfs = await opfsHasStaleModelFile(modelKey, mInfo.version);
        const hasStaleIdb = [...existingKeys].some((k) => !k.startsWith(`${modelKey}:${mInfo.version}:`));
        if (hasStaleOpfs || hasStaleIdb) {
          staleModels.push(modelKey);
        }
      }
    }
    // 同步检查可选模型（如 char_det）是否存在旧版本需要刷新
    for (const [modelKey, mInfo] of Object.entries(manifest.optionalModels || {})) {
      if (!mInfo || !mInfo.version) continue;
      if (!(await opfsCheckModelFile(modelKey, mInfo.version, mInfo.totalSize))) {
        const hasStaleOpfs = await opfsHasStaleModelFile(modelKey, mInfo.version);
        if (hasStaleOpfs) staleModels.push(modelKey);
      }
    }
  } finally {
    if (db) db.close();
  }
  const missingGzBytes = Math.max(0, totalGzBytes - cachedGzBytes);
  return {
    totalGzBytes,
    cachedGzBytes,
    missingGzBytes,
    totalChunks,
    cachedChunks,
    allCached: totalChunks > 0 && cachedChunks === totalChunks,
    opfsModels,
    hasUpdate: staleModels.length > 0 && missingGzBytes > 0,
    staleModels,
    buildId: manifest.buildId || "",
  };
}

export async function loadChunkedModel(
  modelKey,
  modelInfo,
  baseUrl = "./models_chunks/",
  { concurrency = 5, retries = 3, onProgress = null } = {}
) {
  const version = modelInfo.version;
  const chunks = modelInfo.chunks;
  const totalGzBytes = modelInfo.totalGzSize || chunks.reduce((a, c) => a + c.gzSize, 0);
  const t0 = performance.now();

  const opfsBytes = await opfsReadCompleteModel(modelKey, version, modelInfo.totalSize);
  if (opfsBytes) {
    if (onProgress) {
      onProgress({ modelKey, phase: "cache", loadedBytes: totalGzBytes, totalBytes: totalGzBytes, cachedChunks: chunks.length, downloadedChunks: 0, totalChunks: chunks.length, speedMBps: 0 });
      onProgress({ modelKey, phase: "done", loadedBytes: totalGzBytes, totalBytes: totalGzBytes, cachedChunks: chunks.length, downloadedChunks: 0, totalChunks: chunks.length, speedMBps: 0 });
    }
    return { bytes: opfsBytes, fromCacheOnly: true, fromOpfs: true, cachedChunks: chunks.length, downloadedChunks: 0, totalChunks: chunks.length, elapsedMs: performance.now() - t0 };
  }

  const db = await openCacheDB();
  const chunkKey = (idx) => `${modelKey}:${version}:${idx}`;
  const existingKeys = new Set(await idbListKeysForModel(db, modelKey));
  const staleKeys = [...existingKeys].filter((k) => !k.startsWith(`${modelKey}:${version}:`));
  if (staleKeys.length > 0) {
    await idbDeleteKeys(db, staleKeys);
    for (const k of staleKeys) existingKeys.delete(k);
  }

  const assembled = new Uint8Array(modelInfo.totalSize);
  const missing = [];
  let cachedChunks = 0, networkChunks = 0, loadedGzBytes = 0, downloadedNetBytes = 0;

  const emitProgress = (phase) => {
    if (!onProgress) return;
    const elapsedSec = Math.max(0.05, (performance.now() - t0) / 1000);
    const speedMBps = downloadedNetBytes > 0 ? downloadedNetBytes / (1024 * 1024) / elapsedSec : 0;
    onProgress({
      modelKey,
      phase,
      loadedBytes: Math.min(totalGzBytes, Math.max(0, loadedGzBytes)),
      totalBytes: totalGzBytes,
      cachedChunks,
      downloadedChunks: networkChunks,
      totalChunks: chunks.length,
      speedMBps,
    });
  };

  const keysToRead = chunks.map((ch) => chunkKey(ch.index)).filter((k) => existingKeys.has(k));
  const idbMap = keysToRead.length > 0 ? await idbGetMany(db, STORE_CHUNKS, keysToRead) : new Map();
  for (const ch of chunks) {
    const rec = idbMap.get(chunkKey(ch.index));
    if (rec && rec.data && rec.data.byteLength === ch.rawSize) {
      assembled.set(rec.data, ch.offset);
      cachedChunks++;
      loadedGzBytes += ch.gzSize;
      continue;
    }
    missing.push(ch);
  }
  if (cachedChunks > 0) emitProgress("cache");

  if (missing.length > 0) {
    let cursor = 0;
    const nWorkers = Math.min(concurrency, missing.length);
    const runWorker = async () => {
      while (cursor < missing.length) {
        const ch = missing[cursor++];
        const url = new URL(ch.file, new URL(baseUrl, self.location.href)).href;
        const gzBytes = await fetchChunkWithProgress(url, (delta) => {
          loadedGzBytes += delta;
          downloadedNetBytes += delta;
          emitProgress("download");
        }, retries);
        const raw = await gunzipUint8(gzBytes);
        if (raw.byteLength !== ch.rawSize) throw new Error(`分片 ${ch.file} 解压长度异常`);
        if (ch.rawSha256) {
          const gotHex = await sha256Hex(raw);
          if (gotHex && gotHex !== ch.rawSha256) throw new Error(`分片 ${ch.file} SHA-256 校验失败`);
        }
        assembled.set(raw, ch.offset);
        networkChunks++;
        await idbPut(db, STORE_CHUNKS, {
          key: chunkKey(ch.index),
          modelKey,
          version,
          index: ch.index,
          rawSize: ch.rawSize,
          data: raw,
        });
        emitProgress("download");
      }
    };
    await Promise.all(Array.from({ length: nWorkers }, () => runWorker()));
  }

  const savedToOpfs = await opfsWriteCompleteModel(modelKey, version, assembled);
  if (savedToOpfs) {
    await idbDeleteKeys(db, chunks.map((ch) => chunkKey(ch.index)));
  }
  await idbPut(db, STORE_META, { modelKey, version, totalSize: modelInfo.totalSize, storage: savedToOpfs ? "opfs" : "idb_chunks", updatedAt: Date.now() });
  if (db) db.close();

  emitProgress("done");
  return {
    bytes: assembled,
    fromCacheOnly: missing.length === 0,
    fromOpfs: false,
    migratedToOpfs: savedToOpfs,
    cachedChunks,
    downloadedChunks: networkChunks,
    totalChunks: chunks.length,
    elapsedMs: performance.now() - t0,
  };
}
