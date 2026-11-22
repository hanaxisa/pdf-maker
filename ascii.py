# -*- coding: utf-8 -*-
"""Bad Apple ASCII 版：灰度 -> 字符密度，逐帧一页 PDF"""
import argparse
import os

import cv2
import numpy as np
from reportlab.pdfgen import canvas

from v2 import detect_content_box

W, H = 960, 720                    # 页面尺寸，4:3
COLS, ROWS = 120, 45               # 字符网格（120x45，13.3pt 时正好铺满 960x720）
FS = W / (COLS * 0.6)              # 等宽字体积：正好铺满宽度
LEAD = H / ROWS                    # 行距：正好铺满高度

# 各档网格分辨率（cols x rows），越大细节越多但字越小
GRID_SIZE_BY_MODEL = {
    "tiny": (120, 45),
    "small": (120, 45),
    "base_plus": (144, 54),
    "large": (168, 63),
}


def grid_size_for_model(model_name):
    """从模型名（支持 'sam2_hiera_large' 或 'large'）映射到网格尺寸。"""
    name = str(model_name).lower().replace("sam2_hiera_", "").replace("sam2.1_hiera_", "")
    return GRID_SIZE_BY_MODEL.get(name, (120, 45))


def set_grid_size(cols, rows):
    """根据档位调整网格分辨率，FS/LEAD 会自动重算。"""
    global COLS, ROWS, FS, LEAD
    COLS, ROWS = int(cols), int(rows)
    FS = W / (COLS * 0.6)
    LEAD = H / ROWS

# 字符表：video=教程六档表，ten=10级密度表（色差更细腻）
BINS = [25, 50, 100, 150, 200]
RAMP_ON_DARK = ["-", ":", "?", "/", "b", "#"]    # 白字黑底：暗->疏，亮->密
RAMP_ON_LIGHT = ["#", "b", "/", "?", ":", "-"]   # 黑字白底：暗->密，亮->疏
TEN_ON_DARK = list(" .:-=+*#")                   # 白字黑底：暗->空，亮->密（去掉 @ % 防刺眼）
TEN_ON_LIGHT = list(" .:-=+*#")[::-1]            # 黑字白底：暗->密，亮->空

# 暗场配色：(背景 RGB, 字符 RGB)
THEMES = {
    "mono":   ((0.0, 0.0, 0.0),    (0.80, 0.80, 0.80)),
    "green":  ((0.016, 0.086, 0.039), (0.43, 1.0, 0.55)),
    "amber":  ((0.10, 0.06, 0.0),  (1.0, 0.70, 0.26)),
    "blue":   ((0.02, 0.06, 0.13), (0.50, 0.77, 1.0)),
    "paper":  ((1.0, 1.0, 1.0),    (0.0, 0.0, 0.0)),
}


def enhance(small, contrast, autolevel, bg_threshold=0):
    """局部自动色阶 + 对比度拉伸，把中间灰推向两极。

    bg_threshold > 0 时进入 Bad Apple 干净模式：
      - 低于阈值的像素直接清零（去除降采样/压缩噪声，防频闪）
      - 以阈值为中心而非 128 做对比度拉伸，让前景边缘更陡、轮廓更干净
      - 建议配合 autolevel=False 使用，避免逐帧色阶漂移引起闪烁
    """
    g = small.astype(np.float32)
    if autolevel:
        lo, hi = np.percentile(g, (2, 98))
        if hi - lo > 5:
            g = (g - lo) * 255.0 / (hi - lo)
    if bg_threshold > 0:
        g[g < bg_threshold] = 0.0
        g = np.clip((g - bg_threshold) * contrast, 0, 255)
    else:
        g = np.clip((g - 128.0) * contrast + 128.0, 0, 255)
    return g.astype(np.uint8)


def fit_grid(gray, cell_aspect=2.0, mode="cover", bg_threshold=20):
    """把任意尺寸的灰度画面缩放成 COLS x ROWS 的网格。

    字符单元在屏幕上是 8pt x 16pt（1:2），120x45 覆盖 4:3 区域。
    mode:
      letterbox 等比缩放 + 补黑边，不变形、不裁切
      cover     等比放大到铺满，裁掉超出的两边，不变形（默认，最大化利用网格）
      stretch   直接拉伸到整格，完全铺满但画面会变形
    bg_threshold: 低于此值的像素视为背景清零，去除压缩/降采样噪声，减少频闪
    """
    h, w = gray.shape[:2]
    canvas = np.zeros((ROWS, COLS), dtype=np.uint8)
    if h <= 0 or w <= 0:
        return canvas
    target = COLS / (ROWS * cell_aspect)

    if mode == "stretch":
        out = cv2.resize(gray, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        if bg_threshold > 0:
            out[out < bg_threshold] = 0
        return out

    ratio = w / h
    if mode == "cover":
        if ratio > target:      # 画面偏宽：保留中间，裁掉左右
            keep = max(1, int(round(h * target)))
            left = (w - keep) // 2
            gray = gray[:, left:left + keep]
        else:                   # 画面偏高：保留中间，裁掉上下
            keep = max(1, int(round(w / target)))
            top = (h - keep) // 2
            gray = gray[top:top + keep, :]
        out = cv2.resize(gray, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        if bg_threshold > 0:
            out[out < bg_threshold] = 0
        return out

    if ratio >= target:
        new_w = COLS
        new_h = max(1, min(ROWS, int(round(COLS / (ratio * cell_aspect)))))
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        top = (ROWS - new_h) // 2
        canvas[top:top + new_h, :] = resized
    else:
        new_h = ROWS
        new_w = max(1, min(COLS, int(round(ROWS * ratio * cell_aspect))))
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        left = (COLS - new_w) // 2
        canvas[:, left:left + new_w] = resized
    if bg_threshold > 0:
        canvas[canvas < bg_threshold] = 0
    return canvas


def fit_mode_for(width, height, cell_aspect=2.0, max_loss=0.30):
    """根据源画面比例选择裁切方式，避免竖屏视频被 cover 裁掉头部/脚部。

    返回 "cover" 或 "letterbox"：宽高比与网格差得太远（裁切损失超过 30%）时
    用 letterbox 保留完整画面，左右/上下补黑边。
    """
    if width <= 0 or height <= 0:
        return "cover"
    target = COLS / (ROWS * cell_aspect)
    ratio = width / height
    loss = 1.0 - (target / ratio if ratio > target else ratio / target)
    return "letterbox" if loss > max_loss else "cover"


def fit_grid_color(bgr, cell_aspect=2.0, mode="cover"):
    """与 fit_grid 完全相同的裁切几何，输出 COLS x ROWS 的每格平均颜色 (BGR)。

    色度感知渲染需要和灰度网格逐格对齐，所以这里复用同一套 cover 裁切规则，
    只是把单通道灰度换成三通道颜色。
    """
    h, w = bgr.shape[:2]
    canvas = np.zeros((ROWS, COLS, 3), dtype=np.uint8)
    if h <= 0 or w <= 0:
        return canvas
    target = COLS / (ROWS * cell_aspect)

    if mode == "stretch":
        return cv2.resize(bgr, (COLS, ROWS), interpolation=cv2.INTER_AREA)

    ratio = w / h
    if mode == "cover":
        if ratio > target:
            keep = max(1, int(round(h * target)))
            left = (w - keep) // 2
            bgr = bgr[:, left:left + keep]
        else:
            keep = max(1, int(round(w / target)))
            top = (h - keep) // 2
            bgr = bgr[top:top + keep, :]
        return cv2.resize(bgr, (COLS, ROWS), interpolation=cv2.INTER_AREA)

    if ratio >= target:
        new_w = COLS
        new_h = max(1, min(ROWS, int(round(COLS / (ratio * cell_aspect)))))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        top = (ROWS - new_h) // 2
        canvas[top:top + new_h, :] = resized
    else:
        new_h = ROWS
        new_w = max(1, min(COLS, int(round(ROWS * ratio * cell_aspect))))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        left = (COLS - new_w) // 2
        canvas[:, left:left + new_w] = resized
    return canvas


def extract_text(video, duration, fps, box, contrast, autolevel, theme, ramp_key, blur, bgmode, smooth, bg_threshold=35):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dt = 1.0 / fps
    next_t = 0.0
    total = int(duration * fps) if duration > 0 else 10 ** 9
    grids, idx = [], 0
    while len(grids) < total:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / src_fps
        if t + 1e-9 >= next_t:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if box:
                gray = gray[box[1]:box[3], box[0]:box[2]]
            small = cv2.resize(gray, (COLS, ROWS), interpolation=cv2.INTER_AREA)
            if blur > 0:
                small = cv2.GaussianBlur(small, (0, 0), blur)
            grids.append(small)
            next_t += dt
        idx += 1
    cap.release()

    n = len(grids)
    if theme == "paper":
        dark_flags = [False] * n
    elif bgmode == "stable":
        dark_flags = [True] * n          # 恒定黑底：不再按帧翻转
    else:
        means = np.array([g.mean() for g in grids], dtype=np.float32)
        k = max(1, int(smooth))
        if k > 1 and n > k:
            means = np.convolve(means, np.ones(k, np.float32) / k, mode="same")
        dark_flags, state = [], True
        for m in means:                   # 滞回：只有明显偏亮才换，且回差 45
            if m > 150:
                state = False
            elif m < 105:
                state = True
            dark_flags.append(state)

    bg, fg = THEMES[theme]
    frames = []
    for g, dark in zip(grids, dark_flags):
        v = enhance(g, contrast, autolevel, bg_threshold=bg_threshold)
        if ramp_key == "ten":
            ramp = TEN_ON_DARK if dark else TEN_ON_LIGHT
            lvl = (v.astype("int32") * len(ramp) // 256).clip(0, len(ramp) - 1)
        else:
            ramp = RAMP_ON_DARK if dark else RAMP_ON_LIGHT
            lvl = np.digitize(v, BINS)
        lines = ["".join(ramp[i] for i in row) for row in lvl.tolist()]
        frames.append((lines, dark))
    return frames, bg, fg


def build_pdf(frames, out, bg, fg, credits=None, credit_pages=0):
    c = canvas.Canvas(out, pagesize=(W, H), pageCompression=1)
    for lines, dark in frames:
        if dark:
            c.setFillColorRGB(*bg)
            c.rect(0, 0, W, H, fill=1, stroke=0)
            c.setFillColorRGB(*fg)
        else:
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, 0, W, H, fill=1, stroke=0)
            c.setFillColorRGB(0, 0, 0)
        t = c.beginText(0, H - LEAD + (LEAD - FS) / 2 + FS * 0.2)
        t.setFont("Courier", FS)
        t.setLeading(LEAD)
        for ln in lines:
            t.textLine(ln)
        c.drawText(t)
        c.showPage()
    if credits and credit_pages > 0:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        pdfmetrics.registerFont(TTFont("CN", r"C:\Windows\Fonts\simhei.ttf"))
        size = 22
        for _ in range(credit_pages):
            c.setFillColorRGB(*bg)
            c.rect(0, 0, W, H, fill=1, stroke=0)
            c.setFillColorRGB(*fg)
            c.setFont("CN", size)
            y = H / 2 + (len(credits) - 1) * size * 0.9
            for ln in credits:
                c.drawString((W - c.stringWidth(ln, "CN", size)) / 2, y, ln)
                y -= size * 1.8
            c.showPage()
    c.save()


def main():
    p = argparse.ArgumentParser(description="Bad Apple ASCII 版生成器")
    p.add_argument("video")
    p.add_argument("-o", "--out", default=os.path.join("out", "badapple_ascii.pdf"))
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--theme", choices=list(THEMES), default="mono")
    p.add_argument("--ramp", choices=["ten", "video"], default="ten", help="字符表：ten=10级，video=教程六档")
    p.add_argument("--contrast", type=float, default=5.0, help="对比度拉伸（Bad Apple 建议高对比）")
    p.add_argument("--blur", type=float, default=0.0, help="网格尺度高斯模糊（0=关闭，防边缘光晕）")
    p.add_argument("--bgmode", choices=["stable", "switch"], default="stable",
                   help="stable=恒定黑底；switch=跟随画面明暗翻转底色")
    p.add_argument("--smooth", type=int, default=18, help="翻转判定的时间平滑窗口（帧）")
    p.add_argument("--inset", type=int, default=10, help="裁切框四周内缩像素，去掉片源边缘暗边")
    p.add_argument("--credits", default="", help="片尾字幕，多行用 | 分隔")
    p.add_argument("--credit-sec", type=float, default=3.0, help="片尾停留秒数")
    p.add_argument("--bg-threshold", type=int, default=35, help="背景清零阈值，去噪声防频闪")
    p.add_argument("--autolevel", action="store_true", help="开启逐帧自动色阶（默认关闭以防频闪）")
    a = p.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    box = detect_content_box(a.video)
    if box and a.inset > 0:
        box = (box[0] + a.inset, box[1] + a.inset, box[2] - a.inset, box[3] - a.inset)
    print(f"画面区域 {box}  网格 {COLS}x{ROWS}  字体 {FS:.1f}pt  主题 {a.theme}  "
          f"字符表 {a.ramp}  对比度 {a.contrast}  模糊 {a.blur}  底色 {a.bgmode}")
    frames, bg, fg = extract_text(a.video, a.duration, a.fps, box, a.contrast,
                                  a.autolevel, a.theme, a.ramp, a.blur,
                                  a.bgmode, a.smooth, bg_threshold=a.bg_threshold)
    flips = sum(1 for i in range(1, len(frames)) if frames[i][1] != frames[i - 1][1])
    credits = [s for s in a.credits.split("|") if s] if a.credits else []
    credit_pages = int(a.credit_sec * a.fps) if credits else 0
    print(f"共 {len(frames)+credit_pages} 帧，底色翻转 {flips} 次，片尾 {credit_pages} 帧")
    build_pdf(frames, a.out, bg, fg, credits, credit_pages)
    print(f"完成 {a.out} {os.path.getsize(a.out)/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
