/**
 * 汉火字形 (xing.hanhuo.org) Cloudflare Worker 全栈入口
 * 1. /glyph/{id}          : O(1) 查 glyph_toc.bin + R2 Range 零损直出 16.7 万基础代表原图
 * 2. /upload/uploads/...  : 从 R2 直出用户提交并压缩的框选 WebP 原图（严格限定 uploads/ 前缀）
 * 3. /api/*               : 基于 Cloudflare D1 (XING_DB) 的评价打分、社区增量向量同步、帮看广场与站长审核后台
 * 4. 静态资源三层强缓存 + COOP/COEP/CORP 跨域隔离头 + ETag/Last-Modified 304 协商
 */
const _computedEtagCache = new Map();
const BUILD_LAST_MODIFIED = "Sat, 21 Nov 2026 07:42:00 GMT";
const BUILD_LAST_MODIFIED_MS = Date.parse(BUILD_LAST_MODIFIED);
const MAX_UPLOAD_BYTES = 520 * 1024; // 单张框选 WebP 严格限 520 KB

const MIME_BY_ID = ["image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"];

let _tocBytes = null;
let _tocPromise = null;
let _stelePageTocBytes = null;
let _stelePageTocPromise = null;
let _steleContextData = null;
let _steleContextPromise = null;
let _schemaReady = false;

function jsonResp(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...extraHeaders,
    },
  });
}

function timingSafeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string") return false;
  const enc = new TextEncoder();
  const ba = enc.encode(a);
  const bb = enc.encode(b);
  if (ba.length !== bb.length) return false;
  let diff = 0;
  for (let i = 0; i < ba.length; i++) diff |= ba[i] ^ bb[i];
  return diff === 0;
}

function verifyAdmin(request, env) {
  const expected = env.ADMIN_TOKEN;
  if (!expected) return { ok: false, status: 503, msg: "尚未在 Cloudflare Workers 配置 ADMIN_TOKEN Secret" };
  const auth = request.headers.get("Authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7).trim() : "";
  if (!token || !timingSafeEqual(token, expected)) {
    return { ok: false, status: 401, msg: "管理员口令无效或已失效" };
  }
  return { ok: true };
}

function safeJsonParse(s, fallback) {
  try { return JSON.parse(s); } catch (_) { return fallback; }
}

function decodeDataUrlImage(dataUrl) {
  if (typeof dataUrl !== "string") return null;
  const m = dataUrl.match(/^data:(image\/(?:webp|jpeg|png));base64,([A-Za-z0-9+/=]+)$/);
  if (!m) return null;
  const contentType = m[1];
  const ext = contentType === "image/webp" ? "webp" : contentType === "image/jpeg" ? "jpg" : "png";
  const binStr = atob(m[2]);
  if (binStr.length === 0 || binStr.length > MAX_UPLOAD_BYTES) return null;
  const u8 = new Uint8Array(binStr.length);
  for (let i = 0; i < binStr.length; i++) u8[i] = binStr.charCodeAt(i);
  return { bytes: u8, contentType, ext };
}

async function saveUserImageToR2(env, folder, dataUrl) {
  if (!dataUrl || !env.GLYPH_BUCKET) return "";
  const decoded = decodeDataUrlImage(dataUrl);
  if (!decoded) throw new Error("图片格式无效或体积超过 500 KB");
  const day = new Date().toISOString().slice(0, 10).replace(/-/g, "");
  const uid = crypto.randomUUID().replace(/-/g, "").slice(0, 16);
  const key = `uploads/${folder}/${day}-${uid}.${decoded.ext}`;
  await env.GLYPH_BUCKET.put(key, decoded.bytes, {
    httpMetadata: { contentType: decoded.contentType },
  });
  return key;
}

async function deleteR2KeySafe(env, key) {
  if (!key || !env.GLYPH_BUCKET || !key.startsWith("uploads/")) return;
  try { await env.GLYPH_BUCKET.delete(key); } catch (_) {}
}

function validateEmbeddingB64(b64) {
  if (typeof b64 !== "string" || !b64) return "";
  // 512 bytes INT8 -> base64 长度固定为 684
  if (b64.length > 800 || !/^[A-Za-z0-9+/=]+$/.test(b64)) return "";
  try {
    const raw = atob(b64);
    if (raw.length !== 512) return "";
    return b64;
  } catch (_) {
    return "";
  }
}

async function ensureSchema(db) {
  if (_schemaReady) return;
  await db.batch([
    db.prepare(`CREATE TABLE IF NOT EXISTS glyph_feedback (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
      nickname TEXT NOT NULL,
      rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
      comment TEXT NOT NULL DEFAULT '',
      correct_char TEXT NOT NULL DEFAULT '',
      style TEXT NOT NULL DEFAULT 'unknown',
      dynasty TEXT NOT NULL DEFAULT 'unknown',
      top1_char TEXT NOT NULL DEFAULT '',
      top1_score REAL NOT NULL DEFAULT 0,
      image_key TEXT NOT NULL DEFAULT '',
      embedding_b64 TEXT NOT NULL DEFAULT '',
      inv_norm REAL NOT NULL DEFAULT 0,
      is_public INTEGER NOT NULL DEFAULT 0 CHECK (is_public IN (0, 1)),
      vec_approved INTEGER NOT NULL DEFAULT 0 CHECK (vec_approved IN (0, 1)),
      status TEXT NOT NULL DEFAULT 'pending',
      admin_note TEXT NOT NULL DEFAULT ''
    )`),
    db.prepare(`CREATE TABLE IF NOT EXISTS help_posts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
      nickname TEXT NOT NULL,
      title_note TEXT NOT NULL DEFAULT '',
      style_hint TEXT NOT NULL DEFAULT 'unknown',
      image_key TEXT NOT NULL DEFAULT '',
      embedding_b64 TEXT NOT NULL DEFAULT '',
      inv_norm REAL NOT NULL DEFAULT 0,
      top_candidates_json TEXT NOT NULL DEFAULT '[]',
      is_public INTEGER NOT NULL DEFAULT 0 CHECK (is_public IN (0, 1)),
      resolved_char TEXT NOT NULL DEFAULT '',
      resolved_style TEXT NOT NULL DEFAULT 'unknown',
      vec_approved INTEGER NOT NULL DEFAULT 0 CHECK (vec_approved IN (0, 1)),
      status TEXT NOT NULL DEFAULT 'pending',
      admin_note TEXT NOT NULL DEFAULT ''
    )`),
    db.prepare(`CREATE TABLE IF NOT EXISTS help_comments (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      post_id INTEGER NOT NULL,
      created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
      nickname TEXT NOT NULL,
      suggested_char TEXT NOT NULL DEFAULT '',
      comment TEXT NOT NULL DEFAULT '',
      is_public INTEGER NOT NULL DEFAULT 0 CHECK (is_public IN (0, 1)),
      is_accepted INTEGER NOT NULL DEFAULT 0 CHECK (is_accepted IN (0, 1)),
      status TEXT NOT NULL DEFAULT 'pending'
    )`),
    db.prepare(`CREATE INDEX IF NOT EXISTS idx_fb_pub ON glyph_feedback(is_public, id)`),
    db.prepare(`CREATE INDEX IF NOT EXISTS idx_fb_vec ON glyph_feedback(vec_approved, id)`),
    db.prepare(`CREATE INDEX IF NOT EXISTS idx_help_pub ON help_posts(is_public, id)`),
    db.prepare(`CREATE INDEX IF NOT EXISTS idx_help_vec ON help_posts(vec_approved, id)`),
    db.prepare(`CREATE INDEX IF NOT EXISTS idx_hc_post ON help_comments(post_id, is_public, id)`),
  ]);
  _schemaReady = true;
}

async function handleApi(request, env, url) {
  const db = env.XING_DB;
  if (!db) return jsonResp({ ok: false, error: "尚未绑定 D1 数据库 (XING_DB)" }, 503);
  await ensureSchema(db);

  const path = url.pathname;
  const method = request.method.toUpperCase();

  // 1) 提交识别打分、纠错、框选 WebP 图片与 512 维向量（默认待审核）
  if (path === "/api/feedback" && method === "POST") {
    let body;
    try { body = await request.json(); } catch (_) { return jsonResp({ ok: false, error: "请求格式错误" }, 400); }
    const nickname = String(body.nickname || "热心书友").trim().slice(0, 32) || "热心书友";
    const rating = Math.max(1, Math.min(5, Number.parseInt(body.rating, 10) || 5));
    const comment = String(body.comment || "").trim().slice(0, 500);
    const correctChar = String(body.correct_char || "").trim().slice(0, 8);
    const style = String(body.style || "unknown").trim().slice(0, 24) || "unknown";
    const dynasty = String(body.dynasty || "unknown").trim().slice(0, 24) || "unknown";
    const top1Char = String(body.top1_char || "").trim().slice(0, 8);
    const top1Score = Number.isFinite(Number(body.top1_score)) ? Number(body.top1_score) : 0;
    const embB64 = validateEmbeddingB64(body.embedding_b64 || "");
    const invNorm = Number.isFinite(Number(body.inv_norm)) ? Number(body.inv_norm) : 0;

    if (!comment && !correctChar) {
      return jsonResp({ ok: false, error: "请填写评论内容或对应的正确汉字" }, 400);
    }

    let imageKey = "";
    if (body.image_data_url) {
      try { imageKey = await saveUserImageToR2(env, "fb", body.image_data_url); }
      catch (err) { return jsonResp({ ok: false, error: err.message }, 400); }
    }

    const res = await db
      .prepare(
        `INSERT INTO glyph_feedback
          (nickname, rating, comment, correct_char, style, dynasty, top1_char, top1_score, image_key, embedding_b64, inv_norm, is_public, vec_approved, status)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 0, 0, 'pending')`
      )
      .bind(nickname, rating, comment, correctChar, style, dynasty, top1Char, top1Score, imageKey, embB64, invNorm)
      .run();

    return jsonResp({ ok: true, id: res.meta?.last_row_id || 0, image_key: imageKey, has_vector: Boolean(embB64) }, 201);
  }

  // 2) 公开精选评价墙（仅返回站长审核 is_public=1）
  if (path === "/api/feedback/public" && method === "GET") {
    const limit = Math.max(1, Math.min(60, Number.parseInt(url.searchParams.get("limit") || "24", 10) || 24));
    const { results } = await db
      .prepare(
        `SELECT id, created_at, nickname, rating, comment, correct_char, style, dynasty, top1_char, top1_score, image_key, vec_approved
         FROM glyph_feedback WHERE is_public = 1 ORDER BY id DESC LIMIT ?1`
      )
      .bind(limit)
      .all();
    return jsonResp(
      { ok: true, total: results.length, items: results },
      200,
      { "Cache-Control": "public, max-age=15, stale-while-revalidate=60" }
    );
  }

  // 3) 帮看广场：提交疑难字求助帖（强制带框选 WebP 原图 + 可选 512 维向量，默认待审核）
  if (path === "/api/help" && method === "POST") {
    let body;
    try { body = await request.json(); } catch (_) { return jsonResp({ ok: false, error: "请求格式错误" }, 400); }
    const nickname = String(body.nickname || "求字书友").trim().slice(0, 32) || "求字书友";
    const titleNote = String(body.title_note || "").trim().slice(0, 300);
    const styleHint = String(body.style_hint || "unknown").trim().slice(0, 24) || "unknown";
    const embB64 = validateEmbeddingB64(body.embedding_b64 || "");
    const invNorm = Number.isFinite(Number(body.inv_norm)) ? Number(body.inv_norm) : 0;
    const topCands = Array.isArray(body.top_candidates) ? body.top_candidates.slice(0, 5) : [];

    if (!body.image_data_url) {
      return jsonResp({ ok: false, error: "帮看求助必须附带框选后的字形图片" }, 400);
    }
    let imageKey = "";
    try { imageKey = await saveUserImageToR2(env, "help", body.image_data_url); }
    catch (err) { return jsonResp({ ok: false, error: err.message }, 400); }

    const res = await db
      .prepare(
        `INSERT INTO help_posts
          (nickname, title_note, style_hint, image_key, embedding_b64, inv_norm, top_candidates_json, is_public, status)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, 'pending')`
      )
      .bind(nickname, titleNote, styleHint, imageKey, embB64, invNorm, JSON.stringify(topCands))
      .run();

    return jsonResp({ ok: true, id: res.meta?.last_row_id || 0, image_key: imageKey }, 201);
  }

  // 4) 帮看广场公开列表（仅 is_public=1，附带每个帖子下已审核公开的帮看评论）
  if (path === "/api/help" && method === "GET") {
    const limit = Math.max(1, Math.min(50, Number.parseInt(url.searchParams.get("limit") || "30", 10) || 30));
    const { results: posts } = await db
      .prepare(
        `SELECT id, created_at, nickname, title_note, style_hint, image_key, top_candidates_json, resolved_char, resolved_style, vec_approved, status
         FROM help_posts WHERE is_public = 1 ORDER BY id DESC LIMIT ?1`
      )
      .bind(limit)
      .all();

    let commentsByPost = new Map();
    if (posts.length > 0) {
      const ids = posts.map((p) => p.id);
      const placeholders = ids.map((_, idx) => `?${idx + 1}`).join(",");
      const { results: cRows } = await db
        .prepare(
          `SELECT id, post_id, created_at, nickname, suggested_char, comment, is_accepted
           FROM help_comments WHERE is_public = 1 AND post_id IN (${placeholders})
           ORDER BY is_accepted DESC, id ASC`
        )
        .bind(...ids)
        .all();
      for (const c of cRows) {
        if (!commentsByPost.has(c.post_id)) commentsByPost.set(c.post_id, []);
        commentsByPost.get(c.post_id).push(c);
      }
    }

    const items = posts.map((p) => ({
      ...p,
      top_candidates: safeJsonParse(p.top_candidates_json || "[]", []),
      comments: commentsByPost.get(p.id) || [],
    }));
    return jsonResp(
      { ok: true, total: items.length, items },
      200,
      { "Cache-Control": "public, max-age=15, stale-while-revalidate=60" }
    );
  }

  // 5) 帮看帖发表认字评论（需站长审核后公开）
  const helpCommentMatch = path.match(/^\/api\/help\/(\d+)\/comments$/);
  if (helpCommentMatch && method === "POST") {
    const postId = Number.parseInt(helpCommentMatch[1], 10);
    const exists = await db.prepare("SELECT id FROM help_posts WHERE id = ?1 AND is_public = 1").bind(postId).first();
    if (!exists) return jsonResp({ ok: false, error: "帮看帖不存在或未公开" }, 404);

    let body;
    try { body = await request.json(); } catch (_) { return jsonResp({ ok: false, error: "请求格式错误" }, 400); }
    const nickname = String(body.nickname || "热心书友").trim().slice(0, 32) || "热心书友";
    const suggestedChar = String(body.suggested_char || "").trim().slice(0, 8);
    const comment = String(body.comment || "").trim().slice(0, 500);
    if (!suggestedChar && !comment) {
      return jsonResp({ ok: false, error: "请填写您认为的汉字或考证说明" }, 400);
    }

    const res = await db
      .prepare(
        `INSERT INTO help_comments (post_id, nickname, suggested_char, comment, is_public, is_accepted, status)
         VALUES (?1, ?2, ?3, ?4, 0, 0, 'pending')`
      )
      .bind(postId, nickname, suggestedChar, comment)
      .run();
    return jsonResp({ ok: true, id: res.meta?.last_row_id || 0 }, 201);
  }

  // 5.5) 轻量社区字库版本探针（仅 ~70B JSON，用于前端毫秒级检测是否有新入库字形并亮起红角标）
  if (path === "/api/community/version" && method === "GET") {
    const [fbRow, hpRow] = await Promise.all([
      db
        .prepare(
          `SELECT COUNT(*) AS cnt, COALESCE(MAX(id), 0) AS max_id
           FROM glyph_feedback
           WHERE vec_approved = 1 AND correct_char != '' AND embedding_b64 != '' AND inv_norm > 0`
        )
        .first(),
      db
        .prepare(
          `SELECT COUNT(*) AS cnt, COALESCE(MAX(id), 0) AS max_id
           FROM help_posts
           WHERE vec_approved = 1 AND resolved_char != '' AND embedding_b64 != '' AND inv_norm > 0`
        )
        .first(),
    ]);
    const fbCnt = Number(fbRow?.cnt || 0);
    const hpCnt = Number(hpRow?.cnt || 0);
    const fbMax = Number(fbRow?.max_id || 0);
    const hpMax = Number(hpRow?.max_id || 0);
    const count = fbCnt + hpCnt;
    return jsonResp(
      { ok: true, count, fb_max_id: fbMax, help_max_id: hpMax, sig: `${fbMax}_${hpMax}_${count}` },
      200,
      { "Cache-Control": "no-store" }
    );
  }

  // 6) 前端联网增量更新接口：拉取全部已审核采纳（vec_approved=1）的社区字形向量
  if (path === "/api/community/vectors" && method === "GET") {
    const [fbRes, helpRes] = await Promise.all([
      db
        .prepare(
          `SELECT id, correct_char AS char, style, dynasty, image_key, embedding_b64, inv_norm, created_at
           FROM glyph_feedback
           WHERE vec_approved = 1 AND correct_char != '' AND embedding_b64 != '' AND inv_norm > 0
           ORDER BY id ASC`
        )
        .all(),
      db
        .prepare(
          `SELECT id, resolved_char AS char, resolved_style AS style, '帮看定案' AS dynasty, image_key, embedding_b64, inv_norm, created_at
           FROM help_posts
           WHERE vec_approved = 1 AND resolved_char != '' AND embedding_b64 != '' AND inv_norm > 0
           ORDER BY id ASC`
        )
        .all(),
    ]);

    const items = [];
    for (const r of fbRes.results || []) {
      items.push({
        cid: `fb_${r.id}`,
        char: r.char,
        style: r.style || "unknown",
        dynasty: r.dynasty || "众包收录",
        image_url: r.image_key ? `./upload/${r.image_key}` : "",
        embedding_b64: r.embedding_b64,
        inv_norm: r.inv_norm,
        created_at: r.created_at,
      });
    }
    for (const r of helpRes.results || []) {
      items.push({
        cid: `help_${r.id}`,
        char: r.char,
        style: r.style || "unknown",
        dynasty: "帮看定案",
        image_url: r.image_key ? `./upload/${r.image_key}` : "",
        embedding_b64: r.embedding_b64,
        inv_norm: r.inv_norm,
        created_at: r.created_at,
      });
    }
    return jsonResp(
      { ok: true, count: items.length, items },
      200,
      { "Cache-Control": "no-cache" }
    );
  }

  // ===================== 站长管理后台 API（Bearer ADMIN_TOKEN） =====================
  if (path.startsWith("/api/admin/")) {
    const auth = verifyAdmin(request, env);
    if (!auth.ok) return jsonResp({ ok: false, error: auth.msg }, auth.status);

    if (path === "/api/admin/overview" && method === "GET") {
      const [fb, hp, hc] = await Promise.all([
        db.prepare("SELECT * FROM glyph_feedback ORDER BY id DESC LIMIT 200").all(),
        db.prepare("SELECT * FROM help_posts ORDER BY id DESC LIMIT 200").all(),
        db.prepare(
          `SELECT c.*, p.image_key AS post_image_key, p.title_note AS post_title_note, p.resolved_char AS post_resolved_char
           FROM help_comments c LEFT JOIN help_posts p ON p.id = c.post_id
           ORDER BY c.id DESC LIMIT 300`
        ).all(),
      ]);
      const helpPosts = (hp.results || []).map((p) => ({
        ...p,
        top_candidates: safeJsonParse(p.top_candidates_json || "[]", []),
        has_vector: Boolean(p.embedding_b64),
      }));
      const feedbacks = (fb.results || []).map((f) => ({
        ...f,
        has_vector: Boolean(f.embedding_b64),
      }));
      return jsonResp({
        ok: true,
        feedbacks,
        help_posts: helpPosts,
        help_comments: hc.results || [],
      });
    }

    const fbIdMatch = path.match(/^\/api\/admin\/feedback\/(\d+)$/);
    if (fbIdMatch) {
      const id = Number.parseInt(fbIdMatch[1], 10);
      if (method === "PATCH") {
        let body;
        try { body = await request.json(); } catch (_) { return jsonResp({ ok: false, error: "JSON 格式错误" }, 400); }
        const sets = [], vals = [];
        const add = (col, val) => { sets.push(`${col} = ?${vals.length + 1}`); vals.push(val); };
        if (body.is_public !== undefined) add("is_public", body.is_public ? 1 : 0);
        if (body.vec_approved !== undefined) add("vec_approved", body.vec_approved ? 1 : 0);
        if (body.correct_char !== undefined) add("correct_char", String(body.correct_char).trim().slice(0, 8));
        if (body.style !== undefined) add("style", String(body.style).trim().slice(0, 24));
        if (body.dynasty !== undefined) add("dynasty", String(body.dynasty).trim().slice(0, 24));
        if (body.status !== undefined) add("status", String(body.status).trim().slice(0, 20));
        if (body.admin_note !== undefined) add("admin_note", String(body.admin_note).trim().slice(0, 300));
        if (sets.length === 0) return jsonResp({ ok: false, error: "无更新字段" }, 400);
        vals.push(id);
        await db.prepare(`UPDATE glyph_feedback SET ${sets.join(", ")} WHERE id = ?${vals.length}`).bind(...vals).run();
        const updated = await db.prepare("SELECT * FROM glyph_feedback WHERE id = ?1").bind(id).first();
        return jsonResp({ ok: true, item: updated });
      }
      if (method === "DELETE") {
        const old = await db.prepare("SELECT image_key FROM glyph_feedback WHERE id = ?1").bind(id).first();
        await db.prepare("DELETE FROM glyph_feedback WHERE id = ?1").bind(id).run();
        if (old?.image_key) await deleteR2KeySafe(env, old.image_key);
        return jsonResp({ ok: true, deleted: id });
      }
    }

    const hpIdMatch = path.match(/^\/api\/admin\/help\/(\d+)$/);
    if (hpIdMatch) {
      const id = Number.parseInt(hpIdMatch[1], 10);
      if (method === "PATCH") {
        let body;
        try { body = await request.json(); } catch (_) { return jsonResp({ ok: false, error: "JSON 格式错误" }, 400); }
        const sets = [], vals = [];
        const add = (col, val) => { sets.push(`${col} = ?${vals.length + 1}`); vals.push(val); };
        if (body.is_public !== undefined) add("is_public", body.is_public ? 1 : 0);
        if (body.vec_approved !== undefined) add("vec_approved", body.vec_approved ? 1 : 0);
        if (body.resolved_char !== undefined) add("resolved_char", String(body.resolved_char).trim().slice(0, 8));
        if (body.resolved_style !== undefined) add("resolved_style", String(body.resolved_style).trim().slice(0, 24));
        if (body.status !== undefined) add("status", String(body.status).trim().slice(0, 20));
        if (body.admin_note !== undefined) add("admin_note", String(body.admin_note).trim().slice(0, 300));
        if (sets.length === 0) return jsonResp({ ok: false, error: "无更新字段" }, 400);
        vals.push(id);
        await db.prepare(`UPDATE help_posts SET ${sets.join(", ")} WHERE id = ?${vals.length}`).bind(...vals).run();
        const updated = await db.prepare("SELECT * FROM help_posts WHERE id = ?1").bind(id).first();
        return jsonResp({ ok: true, item: updated });
      }
      if (method === "DELETE") {
        const old = await db.prepare("SELECT image_key FROM help_posts WHERE id = ?1").bind(id).first();
        await db.batch([
          db.prepare("DELETE FROM help_comments WHERE post_id = ?1").bind(id),
          db.prepare("DELETE FROM help_posts WHERE id = ?1").bind(id),
        ]);
        if (old?.image_key) await deleteR2KeySafe(env, old.image_key);
        return jsonResp({ ok: true, deleted: id });
      }
    }

    const hcIdMatch = path.match(/^\/api\/admin\/help-comments\/(\d+)$/);
    if (hcIdMatch) {
      const id = Number.parseInt(hcIdMatch[1], 10);
      if (method === "PATCH") {
        let body;
        try { body = await request.json(); } catch (_) { return jsonResp({ ok: false, error: "JSON 格式错误" }, 400); }
        const sets = [], vals = [];
        const add = (col, val) => { sets.push(`${col} = ?${vals.length + 1}`); vals.push(val); };
        if (body.is_public !== undefined) add("is_public", body.is_public ? 1 : 0);
        if (body.is_accepted !== undefined) add("is_accepted", body.is_accepted ? 1 : 0);
        if (body.status !== undefined) add("status", String(body.status).trim().slice(0, 20));
        if (sets.length === 0) return jsonResp({ ok: false, error: "无更新字段" }, 400);
        vals.push(id);
        await db.prepare(`UPDATE help_comments SET ${sets.join(", ")} WHERE id = ?${vals.length}`).bind(...vals).run();
        // 若设为最佳答案且含建议字，同步回填到帮看帖定案字
        if (body.is_accepted) {
          const c = await db.prepare("SELECT post_id, suggested_char FROM help_comments WHERE id = ?1").bind(id).first();
          if (c && c.suggested_char) {
            await db
              .prepare("UPDATE help_posts SET resolved_char = ?1, status = 'resolved' WHERE id = ?2")
              .bind(c.suggested_char, c.post_id)
              .run();
          }
        }
        return jsonResp({ ok: true, id });
      }
      if (method === "DELETE") {
        await db.prepare("DELETE FROM help_comments WHERE id = ?1").bind(id).run();
        return jsonResp({ ok: true, deleted: id });
      }
    }

    if (path === "/api/admin/export" && method === "GET") {
      const [fb, hp] = await Promise.all([
        db.prepare("SELECT * FROM glyph_feedback ORDER BY id ASC").all(),
        db.prepare("SELECT * FROM help_posts ORDER BY id ASC").all(),
      ]);
      return new Response(JSON.stringify({ exported_at: new Date().toISOString(), feedbacks: fb.results, help_posts: hp.results }, null, 2), {
        status: 200,
        headers: {
          "Content-Type": "application/json; charset=utf-8",
          "Content-Disposition": 'attachment; filename="hanzi_xing_community_export.json"',
          "Cache-Control": "no-store",
        },
      });
    }
  }

  return jsonResp({ ok: false, error: "Not Found" }, 404);
}

async function loadGlyphToc(env, origin) {
  if (_tocBytes) return _tocBytes;
  if (_tocPromise) return _tocPromise;
  _tocPromise = (async () => {
    try {
      const resp = await env.ASSETS.fetch(new Request(`${origin}/assets/glyph_toc.bin`));
      if (resp.ok) {
        _tocBytes = new Uint8Array(await resp.arrayBuffer());
        return _tocBytes;
      }
    } catch (_) {}
    if (env.GLYPH_BUCKET) {
      const obj = await env.GLYPH_BUCKET.get("packs_v2/glyph_toc.bin");
      if (obj) {
        _tocBytes = new Uint8Array(await obj.arrayBuffer());
        return _tocBytes;
      }
    }
    _tocPromise = null;
    return null;
  })();
  return _tocPromise;
}

async function handleGlyphImage(idStr, env, origin) {
  const id = Number.parseInt(idStr, 10);
  if (!Number.isInteger(id) || id < 0) return new Response("Invalid glyph id", { status: 400, headers: { "Cache-Control": "no-store" } });
  if (!env.GLYPH_BUCKET) return new Response("R2 bucket not bound", { status: 503, headers: { "Cache-Control": "no-store" } });
  const toc = await loadGlyphToc(env, origin);
  if (!toc || (id + 1) * 12 > toc.byteLength) return new Response("Glyph not found", { status: 404, headers: { "Cache-Control": "no-store" } });
  const dv = new DataView(toc.buffer, toc.byteOffset + id * 12, 12);
  const packId = dv.getUint16(0, true);
  const mimeId = dv.getUint8(2);
  const offset = dv.getUint32(4, true);
  const length = dv.getUint32(8, true);
  if (packId === 0xffff || length === 0) return new Response("Glyph missing", { status: 404, headers: { "Cache-Control": "no-store" } });
  const packName = `packs_v2/pack_${String(packId).padStart(3, "0")}.bin`;
  const obj = await env.GLYPH_BUCKET.get(packName, { range: { offset, length } });
  if (!obj) return new Response("Pack not found", { status: 404, headers: { "Cache-Control": "no-store" } });
  const headers = new Headers();
  headers.set("Content-Type", MIME_BY_ID[mimeId] || "image/png");
  headers.set("Content-Length", String(length));
  headers.set("Cache-Control", "public, max-age=31536000, immutable");
  headers.set("CDN-Cache-Control", "public, max-age=31536000, immutable");
  headers.set("Cross-Origin-Resource-Policy", "same-origin");
  headers.set("ETag", `"gv2-${packId}-${offset}-${length}"`);
  return new Response(obj.body, { status: 200, headers });
}

async function loadStelePageToc(env, origin) {
  if (_stelePageTocBytes) return _stelePageTocBytes;
  if (_stelePageTocPromise) return _stelePageTocPromise;
  _stelePageTocPromise = (async () => {
    try {
      const resp = await env.ASSETS.fetch(new Request(`${origin}/assets/stele_page_toc.bin`));
      if (resp.ok) {
        _stelePageTocBytes = new Uint8Array(await resp.arrayBuffer());
        return _stelePageTocBytes;
      }
    } catch (_) {}
    if (env.GLYPH_BUCKET) {
      const obj = await env.GLYPH_BUCKET.get("stele_pages/stele_page_toc.bin");
      if (obj) {
        _stelePageTocBytes = new Uint8Array(await obj.arrayBuffer());
        return _stelePageTocBytes;
      }
    }
    _stelePageTocPromise = null;
    return null;
  })();
  return _stelePageTocPromise;
}

async function loadSteleContext(env, origin) {
  if (_steleContextData) return _steleContextData;
  if (_steleContextPromise) return _steleContextPromise;
  _steleContextPromise = (async () => {
    try {
      let gzBuf = null;
      try {
        const resp = await env.ASSETS.fetch(new Request(`${origin}/assets/stele_context.json.gz`));
        if (resp.ok) gzBuf = await resp.arrayBuffer();
      } catch (_) {}
      if (!gzBuf && env.GLYPH_BUCKET) {
        const obj = await env.GLYPH_BUCKET.get("stele_pages/stele_context.json.gz");
        if (obj) gzBuf = await obj.arrayBuffer();
      }
      if (!gzBuf) {
        _steleContextPromise = null;
        return null;
      }
      const u8 = new Uint8Array(gzBuf);
      let text;
      if (u8.length >= 2 && u8[0] === 0x1f && u8[1] === 0x8b) {
        const ds = new DecompressionStream("gzip");
        text = await new Response(new Blob([u8]).stream().pipeThrough(ds)).text();
      } else {
        text = new TextDecoder().decode(u8);
      }
      _steleContextData = JSON.parse(text);
      return _steleContextData;
    } catch (_) {
      _steleContextPromise = null;
      return null;
    }
  })();
  return _steleContextPromise;
}

async function handleGlyphContext(idStr, env, origin) {
  const id = Number.parseInt(idStr, 10);
  if (!Number.isInteger(id) || id < 0) return jsonResp({ ok: false, error: "Invalid glyph id" }, 400);
  const ctx = await loadSteleContext(env, origin);
  if (!ctx || id < ctx.baseId || id >= ctx.baseId + ctx.count) {
    return jsonResp({ ok: true, id, hasPage: false }, 200, { "Cache-Control": "public, max-age=86400" });
  }
  const idx = id - ctx.baseId;
  const pageId = ctx.pageIds[idx];
  const titleId = ctx.titleIds[idx];
  const title = titleId >= 0 ? (ctx.titles[titleId] || "") : "";
  const rawChar = ctx.rawChars[idx] || "";
  const excerpt = ctx.ctxs[idx] || "";
  if (pageId < 0) {
    return jsonResp({ ok: true, id, hasPage: false, title, rawChar, ctx: excerpt }, 200, { "Cache-Control": "public, max-age=86400" });
  }
  const bx = ctx.boxes;
  const box = [bx[idx * 4] / 10000, bx[idx * 4 + 1] / 10000, bx[idx * 4 + 2] / 10000, bx[idx * 4 + 3] / 10000];
  const dims = ctx.pageDims[pageId] || [0, 0];
  return jsonResp({
    ok: true,
    id,
    hasPage: true,
    title,
    rawChar,
    ctx: excerpt,
    pageId,
    pageUrl: `/stele-page/${pageId}`,
    pageDims: dims,
    box,
  }, 200, { "Cache-Control": "public, max-age=86400" });
}

async function handleStelePageImage(idStr, env, origin) {
  const id = Number.parseInt(idStr, 10);
  if (!Number.isInteger(id) || id < 0) return new Response("Invalid page id", { status: 400, headers: { "Cache-Control": "no-store" } });
  if (!env.GLYPH_BUCKET) return new Response("R2 bucket not bound", { status: 503, headers: { "Cache-Control": "no-store" } });
  const toc = await loadStelePageToc(env, origin);
  if (!toc || (id + 1) * 12 > toc.byteLength) return new Response("Stele page not found", { status: 404, headers: { "Cache-Control": "no-store" } });
  const dv = new DataView(toc.buffer, toc.byteOffset + id * 12, 12);
  const packId = dv.getUint16(0, true);
  const mimeId = dv.getUint8(2);
  const offset = dv.getUint32(4, true);
  const length = dv.getUint32(8, true);
  if (packId === 0xffff || length === 0) return new Response("Stele page missing", { status: 404, headers: { "Cache-Control": "no-store" } });
  const packName = `stele_pages/pack_${String(packId).padStart(3, "0")}.bin`;
  const obj = await env.GLYPH_BUCKET.get(packName, { range: { offset, length } });
  if (!obj) return new Response("Stele pack not found", { status: 404, headers: { "Cache-Control": "no-store" } });
  const headers = new Headers();
  headers.set("Content-Type", MIME_BY_ID[mimeId] || "image/jpeg");
  headers.set("Content-Length", String(length));
  headers.set("Cache-Control", "public, max-age=31536000, immutable");
  headers.set("CDN-Cache-Control", "public, max-age=31536000, immutable");
  headers.set("Cross-Origin-Resource-Policy", "same-origin");
  headers.set("ETag", `"sp1-${packId}-${offset}-${length}"`);
  return new Response(obj.body, { status: 200, headers });
}

async function handleUploadedImage(key, env) {
  if (!key.startsWith("uploads/") || key.includes("..")) {
    return new Response("Forbidden", { status: 403, headers: { "Cache-Control": "no-store" } });
  }
  if (!env.GLYPH_BUCKET) return new Response("R2 not bound", { status: 503, headers: { "Cache-Control": "no-store" } });
  const obj = await env.GLYPH_BUCKET.get(key);
  if (!obj) return new Response("Not Found", { status: 404, headers: { "Cache-Control": "no-store" } });
  const headers = new Headers();
  obj.writeHttpMetadata(headers);
  if (!headers.has("Content-Type")) headers.set("Content-Type", key.endsWith(".webp") ? "image/webp" : "image/jpeg");
  headers.set("Cache-Control", "public, max-age=31536000, immutable");
  headers.set("CDN-Cache-Control", "public, max-age=31536000, immutable");
  headers.set("Cross-Origin-Resource-Policy", "same-origin");
  if (obj.httpEtag) headers.set("ETag", obj.httpEtag);
  return new Response(obj.body, { status: 200, headers });
}

async function computeWeakEtag(buf) {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  const arr = new Uint8Array(digest);
  let hex = "";
  for (let i = 0; i < 16; i++) hex += arr[i].toString(16).padStart(2, "0");
  return `W/"${hex}"`;
}

function ifNoneMatchMatches(inmHeader, etag) {
  if (!inmHeader || !etag) return false;
  const clean = (s) => s.trim().replace(/^W\//i, "");
  const target = clean(etag);
  return inmHeader.split(",").some((part) => {
    const t = part.trim();
    return t === "*" || clean(t) === target;
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const p = url.pathname;

    if (p.startsWith("/api/glyph-context/")) return handleGlyphContext(p.slice(19), env, url.origin);
    if (p.startsWith("/api/")) return handleApi(request, env, url);
    if (p.startsWith("/glyph/")) return handleGlyphImage(p.slice(7), env, url.origin);
    if (p.startsWith("/stele-page/")) return handleStelePageImage(p.slice(12), env, url.origin);
    if (p.startsWith("/upload/")) return handleUploadedImage(p.slice(8), env);

    const assetReq = url.searchParams.has("v")
      ? new Request(new URL(p, url.origin).toString(), { method: request.method, headers: request.headers })
      : request;
    const resp = await env.ASSETS.fetch(assetReq);

    const headers = new Headers(resp.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    headers.set("Cross-Origin-Resource-Policy", "same-origin");

    const isVersioned = url.searchParams.has("v");
    const isHtmlOrManifest = p === "/" || p.endsWith(".html") || p.endsWith("manifest.json");
    let bodyOut = resp.body;

    if (resp.status === 200) {
      let etag = headers.get("etag");
      if (etag && isHtmlOrManifest && !/^W\//i.test(etag)) {
        etag = `W/${etag}`;
        headers.set("ETag", etag);
      }
      if (!etag && (isHtmlOrManifest || p.endsWith(".mjs") || p.endsWith(".json"))) {
        const cachedEtag = _computedEtagCache.get(p);
        if (cachedEtag) {
          etag = cachedEtag;
        } else if (request.method === "GET") {
          const ab = await resp.arrayBuffer();
          if (ab.byteLength > 0) {
            etag = await computeWeakEtag(ab);
            _computedEtagCache.set(p, etag);
          }
          bodyOut = ab;
        }
        if (etag) headers.set("ETag", etag);
      }

      if (p === "/sw.js") {
        headers.set("Content-Type", "application/javascript; charset=utf-8");
        headers.set("Cache-Control", "no-cache");
        headers.set("Service-Worker-Allowed", "/");
      } else if (p.endsWith(".webmanifest")) {
        headers.set("Content-Type", "application/manifest+json; charset=utf-8");
        headers.set("Cache-Control", "public, max-age=86400, stale-while-revalidate=604800");
        headers.set("CDN-Cache-Control", "public, max-age=86400, stale-while-revalidate=604800");
      } else if (p.endsWith(".gzbin") || p.endsWith(".wasm") || p.startsWith("/vendor/")) {
        if (p.endsWith(".wasm")) headers.set("Content-Type", "application/wasm");
        else if (p.endsWith(".gzbin")) headers.set("Content-Type", "application/octet-stream");
        headers.set("Cache-Control", "public, max-age=31536000, immutable");
        headers.set("CDN-Cache-Control", "public, max-age=31536000, immutable");
      } else if (isVersioned && !isHtmlOrManifest) {
        if (p.endsWith(".gz")) headers.set("Content-Type", "application/octet-stream");
        headers.set("Cache-Control", "public, max-age=31536000, immutable");
        headers.set("CDN-Cache-Control", "public, max-age=31536000, immutable");
      } else if (p.endsWith(".gz") || p.endsWith(".svg") || p.endsWith(".png") || p.endsWith(".ico")) {
        if (p.endsWith(".gz")) headers.set("Content-Type", "application/octet-stream");
        headers.set("Cache-Control", "public, max-age=86400, stale-while-revalidate=604800");
        headers.set("CDN-Cache-Control", "public, max-age=86400, stale-while-revalidate=604800");
      } else if (p === "/admin" || p === "/admin.html") {
        headers.set("Cache-Control", "no-store");
      } else if (p === "/" || p.endsWith(".html")) {
        headers.set("Last-Modified", BUILD_LAST_MODIFIED);
        headers.set("Cache-Control", "public, max-age=300, stale-while-revalidate=86400");
        headers.set("CDN-Cache-Control", "public, max-age=300, stale-while-revalidate=86400");
      } else if (isHtmlOrManifest || p.endsWith(".mjs") || p.endsWith(".json")) {
        headers.set("Cache-Control", "public, max-age=0, must-revalidate");
        headers.set("CDN-Cache-Control", "public, max-age=60, stale-while-revalidate=3600");
      }

      if (request.method === "GET" || request.method === "HEAD") {
        const inm = request.headers.get("if-none-match");
        const ims = request.headers.get("if-modified-since");
        const etagHit = Boolean(etag && inm && ifNoneMatchMatches(inm, etag));
        const imsMs = ims ? Date.parse(ims) : NaN;
        const lmHit = Boolean((p === "/" || p.endsWith(".html")) && !inm && Number.isFinite(imsMs) && imsMs >= BUILD_LAST_MODIFIED_MS);
        if (etagHit || lmHit) {
          headers.delete("content-length");
          return new Response(null, { status: 304, headers });
        }
      }
    } else if (resp.status !== 304) {
      headers.set("Cache-Control", "no-store");
    }

    return new Response(request.method === "HEAD" ? null : bodyOut, {
      status: resp.status,
      statusText: resp.statusText,
      headers,
    });
  },
};
