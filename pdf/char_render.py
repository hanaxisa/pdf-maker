# -*- coding: utf-8 -*-
"""彩色视频字符画渲染。

主体墨量用饱和度（背景是中性灰，饱和度接近 0，能干净留空），
再用暗度层补很暗的部分（黑眼睛、阴影）。全片只算一次归一化参数，
逐帧不漂移；不做语义识别，也不逐帧自适应。

屏幕网格 168x63（字形 9.52pt、行高 11.43pt，正好铺满 960x720）。
"""
import cv2
import numpy as np

import ascii

RAMP = " .:-=+*#%@"
COLS, ROWS = 168, 63
PAGE_W, PAGE_H = 960, 720
GAMMA = 0.85
DARK_LO, DARK_HI = 0.50, 0.85   # 暗度强调只吃很暗的像素
MIN_SPAN = 0.05


def _page(bgr):
    """等比缩放到 960x720 并补黑边，返回 (页面, 内容掩码)。"""
    sh, sw = bgr.shape[:2]
    scale = min(PAGE_W / sw, PAGE_H / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    page = np.zeros((PAGE_H, PAGE_W, 3), np.uint8)
    mask = np.zeros((PAGE_H, PAGE_W), np.float32)
    x, y = (PAGE_W - nw) // 2, (PAGE_H - nh) // 2
    page[y:y + nh, x:x + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    mask[y:y + nh, x:x + nw] = 1.0
    return page, mask


def _channels(page, mask):
    hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    f = page.astype(np.float32) / 255.0
    luma = 0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]
    return sat * mask, (1.0 - luma) * mask


def _iter(video, fps, count):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dt = 1.0 / fps
    next_t = 0.0
    idx = produced = 0
    try:
        while produced < count:
            ok, frame = cap.read()
            if not ok:
                break
            if idx / src_fps + 1e-9 >= next_t:
                yield _channels(*_page(frame))
                produced += 1
                next_t += dt
            idx += 1
    finally:
        cap.release()


def film_params(video, fps, count, sample=24):
    """全片固定的饱和度归一化范围（背景≈0，主体≈上限）。"""
    step = max(1, count // sample)
    vals = []
    for i, (sat, _) in enumerate(_iter(video, fps, count)):
        if i % step == 0:
            s = sat[sat > 0]
            if s.size:
                vals.append(s.ravel())
        if i + 1 >= count:
            break
    flat = np.concatenate(vals) if vals else np.array([0.0, 1.0])
    lo = float(np.percentile(flat, 60))
    hi = float(np.percentile(flat, 99.5))
    if hi - lo < MIN_SPAN:
        lo, hi = float(flat.min()), max(float(flat.max()), lo + MIN_SPAN)
    print(f"[chars] 全片参数：sat {lo:.3f}~{hi:.3f} gamma={GAMMA}")
    return lo, hi


def char_frames(video, fps, count):
    """逐帧生成 63 行 x 168 字符。"""
    lo, hi = film_params(video, fps, count)
    old = (ascii.COLS, ascii.ROWS)
    ascii.set_grid_size(COLS, ROWS)
    out = []
    try:
        for sat, dark in _iter(video, fps, count):
            s = cv2.resize(sat, (COLS, ROWS), interpolation=cv2.INTER_AREA)
            d = cv2.resize(dark, (COLS, ROWS), interpolation=cv2.INTER_AREA)
            base = np.clip((s - lo) / (hi - lo), 0, 1) ** GAMMA
            acc = np.clip((d - DARK_LO) / (DARK_HI - DARK_LO), 0, 1)
            v = np.maximum(base, acc)
            lv = np.clip(np.round(v * (len(RAMP) - 1)), 0, len(RAMP) - 1).astype(np.int32)
            out.append(["".join(RAMP[i] for i in row) for row in lv.tolist()])
            if len(out) >= count:
                break
    finally:
        ascii.set_grid_size(*old)
    print(f"[chars] 字符画 {len(out)} 帧 / {COLS}x{ROWS}")
    return out
