/**
 * 汉火字形 (xing.hanhuo.org) PWA Service Worker
 * - 离线秒开 App Shell（HTML / JS / Rust-WASM / 200KB 元数据 / ORT 引擎 / 图标）
 * - 保留 COOP / COEP / CORP 跨域隔离头
 * - 不缓存 /models_chunks/*.gzbin（已由 OPFS 整包零合并管理）
 * - 对浏览过的 /glyph/* 代表原图做轻缓存（上限自动滚存），离线也能看查过的原图
 */
const SW_CACHE_NAME = "hanzi-xing-pwa-20261123j";

function ensureIsolationHeaders(resp) {
  if (!resp || resp.type === "opaque") return resp;
  const h = new Headers(resp.headers);
  if (!h.has("Cross-Origin-Opener-Policy")) h.set("Cross-Origin-Opener-Policy", "same-origin");
  if (!h.has("Cross-Origin-Embedder-Policy")) h.set("Cross-Origin-Embedder-Policy", "require-corp");
  if (!h.has("Cross-Origin-Resource-Policy")) h.set("Cross-Origin-Resource-Policy", "same-origin");
  return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers: h });
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SW_CACHE_NAME)
      .then((cache) => cache.addAll(["./", "./manifest.webmanifest"]))
      .catch(() => {})
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("hanzi-xing-pwa-") && k !== SW_CACHE_NAME).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  const p = url.pathname;
  if (
    p === "/sw.js" ||
    p.startsWith("/api/") ||
    p === "/admin" ||
    p === "/admin.html" ||
    (p.endsWith(".gzbin") && p.startsWith("/models_chunks/"))
  ) {
    return;
  }

  // 模型分片清单 manifest.json 与字形元数据 proto_meta.json.gz：Network-First（确保两者版本原子对齐，离线平滑回退本地缓存）
  if (p.endsWith("/manifest.json") || p.endsWith("/proto_meta.json.gz")) {
    event.respondWith(
      (async () => {
        const cache = await caches.open(SW_CACHE_NAME);
        const cacheKey = new Request(new URL(p, self.location.origin).href);
        try {
          const netResp = await fetch(req, { cache: "no-cache" });
          if (netResp && netResp.ok) {
            event.waitUntil(cache.put(cacheKey, netResp.clone()).catch(() => {}));
            return ensureIsolationHeaders(netResp);
          }
        } catch (_) {}
        const cached = (await cache.match(req)) || (await cache.match(cacheKey));
        return cached ? ensureIsolationHeaders(cached) : fetch(req);
      })()
    );
    return;
  }

  // 代表字形原图、整页碑帖高清图 & 众包上传原图：无论 <img> 加载还是新标签页直接打开 (req.mode === "navigate")，
  // 均按完整 URL 精确缓存，绝不落入首页导航分支避免错图或污染 "./" 缓存！
  if (p.startsWith("/glyph/") || p.startsWith("/stele-page/") || p.startsWith("/upload/") || p.startsWith("/uploads/")) {
    event.respondWith(
      (async () => {
        const cache = await caches.open(SW_CACHE_NAME);
        const imgKey = new Request(url.href);
        const cached = await cache.match(imgKey);
        if (cached) return ensureIsolationHeaders(cached);
        const netResp = await fetch(req);
        if (netResp && netResp.ok && (netResp.headers.get("content-type") || "").startsWith("image/")) {
          event.waitUntil(cache.put(imgKey, netResp.clone()).catch(() => {}));
        }
        return ensureIsolationHeaders(netResp);
      })()
    );
    return;
  }

  // 高清看图页 viewer.html：Network-First（保留精确路径，绝不与首页 "./" 混淆）
  if (p === "/viewer.html") {
    event.respondWith(
      (async () => {
        const cache = await caches.open(SW_CACHE_NAME);
        const key = new Request(new URL("/viewer.html", self.location.origin).href);
        try {
          const netResp = await fetch(req, { cache: "no-cache" });
          if (netResp && netResp.ok && (netResp.headers.get("content-type") || "").includes("text/html")) {
            event.waitUntil(cache.put(key, netResp.clone()).catch(() => {}));
            return ensureIsolationHeaders(netResp);
          }
        } catch (_) {}
        const cached = await cache.match(key);
        return cached ? ensureIsolationHeaders(cached) : fetch(req);
      })()
    );
    return;
  }

  // 页面导航（仅限首页 HTML，绝不劫持图片或其它子路径）：Stale-While-Revalidate 本机 0ms 瞬开 + 后台更新
  if (p === "/" || p === "/index.html") {
    event.respondWith(
      (async () => {
        const cache = await caches.open(SW_CACHE_NAME);
        const cacheKey = new Request(new URL("./", self.location.href).href);
        const cached = await cache.match(cacheKey);
        const netPromise = fetch(req).then((r) => {
          if (r && r.ok && (r.headers.get("content-type") || "").includes("text/html")) {
            cache.put(cacheKey, r.clone()).catch(() => {});
          }
          return r;
        }).catch(() => null);
        if (cached && (cached.headers.get("content-type") || "").includes("text/html")) {
          event.waitUntil(netPromise);
          return ensureIsolationHeaders(cached);
        }
        const netResp = await netPromise;
        return netResp ? ensureIsolationHeaders(netResp) : fetch(req);
      })()
    );
    return;
  }

  // 静态资源 & 查过的原图：Cache-First 本机瞬读
  event.respondWith(
    (async () => {
      const cache = await caches.open(SW_CACHE_NAME);
      const cached = await cache.match(req);
      if (cached) return ensureIsolationHeaders(cached);
      const netResp = await fetch(req);
      if (netResp && netResp.ok) {
        event.waitUntil(cache.put(req, netResp.clone()).catch(() => {}));
      }
      return ensureIsolationHeaders(netResp);
    })()
  );
});
