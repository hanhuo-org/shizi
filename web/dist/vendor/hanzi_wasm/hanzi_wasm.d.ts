/* tslint:disable */
/* eslint-disable */

export class PreparedInput {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly aspect_ratio: number;
    readonly cropped_png: string;
    readonly preview: string;
    readonly tensor: Float32Array;
    readonly tensor_alt: Float32Array;
    readonly warning: string;
}

/**
 * 常驻内存的 INT8 原型库检索器（避免每次检索跨 JS/WASM 边界拷贝 86 MB！）
 */
export class ProtoIndex {
    free(): void;
    [Symbol.dispose](): void;
    count(): number;
    /**
     * 从 HXV1 二进制（OPFS 整包直读）+ style_ids 初始化常驻索引
     */
    constructor(vec_bin: Uint8Array | number[], style_ids: Uint8Array | number[]);
    /**
     * 极速 INT8 暴力余弦 Top-K 检索（filter_style < 0 表示不限书体）
     */
    search(query_f32: Float32Array | number[], filter_style: number, topk: number): string;
    /**
     * 按书体位掩码（style_mask == 0 表示不限书体）执行 INT8 暴力余弦 Top-K 检索，支持书体大类（如篆/古文字/楷行草）精确过滤
     */
    search_mask(query_f32: Float32Array | number[], style_mask: number, topk: number): string;
}

/**
 * 多字模式区域透视拉正：按 4 角点把整碑/目标区域拉正裁切为高清彩色 PNG（最大边 1400px）
 */
export function crop_quad_to_png(rgba: Uint8Array | number[], w: number, h: number, quad_flat: Float32Array | number[]): string;

/**
 * 自动检测识别框 + 返回预处理 256×256 预览图（上传照片即刻本地完成，0 网络耗时）
 */
export function detect_from_rgba(rgba: Uint8Array | number[], w: number, h: number): string;

/**
 * 多字模式专用：按紧致矩形框 [x0, y0, x1, y1] 裁切单字并保留全部偏旁部件，输出 224 张量 + 黑白预览 + 彩色裁剪图
 */
export function prepare_char_box_from_rgba(rgba: Uint8Array | number[], w: number, h: number, x0: number, y0: number, x1: number, y1: number): PreparedInput;

/**
 * 按用户微调后的四角坐标做透视拉正 + 字形二值化归一化，一次返回 (3×224×224 f32 张量 + 黑白预览 + 透视拉正彩色裁剪图 + 警告)
 */
export function prepare_input_from_rgba(rgba: Uint8Array | number[], w: number, h: number, quad_flat: Float32Array | number[]): PreparedInput;

/**
 * 仅执行透视拉正与黑白预览生成（不跑模型，用于用户点「确定裁剪」或发「帮看」时实时更新）
 */
export function preview_crop_from_rgba(rgba: Uint8Array | number[], w: number, h: number, quad_flat: Float32Array | number[]): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_preparedinput_free: (a: number, b: number) => void;
    readonly __wbg_protoindex_free: (a: number, b: number) => void;
    readonly crop_quad_to_png: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly detect_from_rgba: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly prepare_char_box_from_rgba: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly prepare_input_from_rgba: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly preparedinput_aspect_ratio: (a: number) => number;
    readonly preparedinput_cropped_png: (a: number) => [number, number];
    readonly preparedinput_preview: (a: number) => [number, number];
    readonly preparedinput_tensor: (a: number) => [number, number];
    readonly preparedinput_tensor_alt: (a: number) => [number, number];
    readonly preparedinput_warning: (a: number) => [number, number];
    readonly preview_crop_from_rgba: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly protoindex_count: (a: number) => number;
    readonly protoindex_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly protoindex_search: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly protoindex_search_mask: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
