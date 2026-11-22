# -*- coding: utf-8 -*-
"""PDF 模板：将模型产出的轮廓/灰度数据渲染为 PDF。

支持三种渲染模式：
  ascii   —— 前景灰度映射为 ASCII 字符密度（与现有 Bad Apple 视频风格一致）
  contour —— 提取前景轮廓并用矢量线条绘制
  filled  —— 按前景灰度填充色块

设计参考 retina_project 的 build_report.py 思路：
  数据（.npz）与渲染（PDF）解耦，模板只负责把结构化数据画到页面上。
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas

# 复用 ascii.py 的字符密度表与主题
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ascii import THEMES, TEN_ON_DARK, TEN_ON_LIGHT, enhance

PAGE_W, PAGE_H = 960, 720


@dataclass
class PDFConfig:
    """PDF 渲染配置。"""
    mode: str = "ascii"              # ascii | contour | filled
    theme: str = "mono"              # mono | green | amber | blue | paper
    cols: int = 128                  # ASCII 模式列数
    contrast: float = 5.0            # 对比度拉伸（Bad Apple 用高对比让轮廓干净）
    autolevel: bool = False          # 自动色阶（关闭防逐帧漂移频闪）
    blur: float = 0.0                # 高斯模糊（关闭防边缘光晕）
    bg_threshold: int = 35           # 背景清零阈值（去噪声防频闪）
    fps: int = 12                    # 播放帧率
    credits: Optional[list] = None   # 片尾字幕


class ContourPDFBuilder:
    """从分割结果构建 PDF。"""

    def __init__(self, config: Optional[PDFConfig] = None):
        self.cfg = config or PDFConfig()

    def build(self, segment_result, out_path: str):
        """主入口：segment_result -> PDF。"""
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        bg, fg = THEMES[self.cfg.theme]

        if self.cfg.mode == "ascii":
            self._build_ascii(segment_result, out_path, bg, fg)
        elif self.cfg.mode == "contour":
            self._build_contour(segment_result, out_path, bg, fg)
        elif self.cfg.mode == "filled":
            self._build_filled(segment_result, out_path, bg, fg)
        else:
            raise ValueError(f"未知模式: {self.cfg.mode}")

        print(f"[pdf] 完成 {out_path}  "
              f"{os.path.getsize(out_path) / 1024 / 1024:.2f} MB")

    # ---------------------------------------------------------- ascii 模式
    def _build_ascii(self, result, out_path: str, bg, fg):
        """前景灰度 -> ASCII 字符密度，逐帧一页。"""
        c = canvas.Canvas(out_path, pagesize=(PAGE_W, PAGE_H), pageCompression=1)
        cols = self.cfg.cols
        rows = max(1, int(cols * result.height / result.width))
        fs = PAGE_W / (cols * 0.6)
        lead = PAGE_H / rows

        for f in result.frames:
            # 只取前景灰度，背景置 0
            gray = f.filled_gray
            small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
            if self.cfg.blur > 0:
                small = cv2.GaussianBlur(small, (0, 0), self.cfg.blur)
            v = enhance(small, self.cfg.contrast, self.cfg.autolevel,
                        bg_threshold=self.cfg.bg_threshold)

            # 前景灰度 -> ASCII 密度表，背景（灰度 0）用空格
            dark = True  # 恒定黑底
            ramp = TEN_ON_DARK if dark else TEN_ON_LIGHT
            lvl = (v.astype("int32") * len(ramp) // 256).clip(0, len(ramp) - 1)
            # 背景像素（灰度接近 0）显示为空格
            bg_mask = v < 5
            lvl[bg_mask] = 0  # 空格对应 ramp[0] = ' '
            lines = ["".join(ramp[i] for i in row) for row in lvl.tolist()]

            c.setFillColorRGB(*bg)
            c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
            c.setFillColorRGB(*fg)
            t = c.beginText(0, PAGE_H - lead + (lead - fs) / 2 + fs * 0.2)
            t.setFont("Courier", fs)
            t.setLeading(lead)
            for ln in lines:
                t.textLine(ln)
            c.drawText(t)
            c.showPage()

        if self.cfg.credits:
            self._draw_credits(c, bg, fg)
        c.save()

    # ---------------------------------------------------------- contour 模式
    def _build_contour(self, result, out_path: str, bg, fg):
        """提取前景轮廓并用矢量线条绘制。"""
        c = canvas.Canvas(out_path, pagesize=(PAGE_W, PAGE_H), pageCompression=1)

        for f in result.frames:
            c.setFillColorRGB(*bg)
            c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
            c.setStrokeColorRGB(*fg)
            c.setLineWidth(0.5)

            mask = f.mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            scale_x = PAGE_W / result.width
            scale_y = PAGE_H / result.height

            for cnt in contours:
                if len(cnt) < 2:
                    continue
                # 简化轮廓
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) < 2:
                    continue
                p = approx.reshape(-1, 2).astype(np.float64)
                p[:, 0] *= scale_x
                # PDF 坐标原点在左下，OpenCV 在左上，需要翻转 Y
                p[:, 1] = PAGE_H - p[:, 1] * scale_y
                path = c.beginPath()
                path.moveTo(p[0, 0], p[0, 1])
                for pt in p[1:]:
                    path.lineTo(pt[0], pt[1])
                path.close()
                c.drawPath(path, fill=0, stroke=1)
            c.showPage()

        if self.cfg.credits:
            self._draw_credits(c, bg, fg)
        c.save()

    # ---------------------------------------------------------- filled 模式
    def _build_filled(self, result, out_path: str, bg, fg):
        """按前景灰度填充色块（矢量矩形网格）。"""
        c = canvas.Canvas(out_path, pagesize=(PAGE_W, PAGE_H), pageCompression=1)
        cols = self.cfg.cols
        rows = max(1, int(cols * result.height / result.width))
        cell_w = PAGE_W / cols
        cell_h = PAGE_H / rows

        for f in result.frames:
            c.setFillColorRGB(*bg)
            c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

            gray = f.filled_gray
            small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
            if self.cfg.blur > 0:
                small = cv2.GaussianBlur(small, (0, 0), self.cfg.blur)
            v = enhance(small, self.cfg.contrast, self.cfg.autolevel,
                        bg_threshold=self.cfg.bg_threshold)

            for y in range(rows):
                for x in range(cols):
                    val = v[y, x]
                    if val < 5:
                        continue
                    # 灰度 -> 前景色透明度
                    alpha = val / 255.0
                    c.setFillColor(Color(fg[0], fg[1], fg[2], alpha=alpha))
                    px = x * cell_w
                    py = PAGE_H - (y + 1) * cell_h
                    c.rect(px, py, cell_w, cell_h, fill=1, stroke=0)
            c.showPage()

        if self.cfg.credits:
            self._draw_credits(c, bg, fg)
        c.save()

    # ---------------------------------------------------------- 片尾
    def _draw_credits(self, c: canvas.Canvas, bg, fg):
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        try:
            pdfmetrics.registerFont(TTFont("CN", r"C:\Windows\Fonts\simhei.ttf"))
            font = "CN"
        except Exception:
            font = "Courier"
        size = 22
        c.setFillColorRGB(*bg)
        c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        c.setFillColorRGB(*fg)
        c.setFont(font, size)
        y = PAGE_H / 2 + (len(self.cfg.credits) - 1) * size * 0.9
        for ln in self.cfg.credits:
            c.drawString((PAGE_W - c.stringWidth(ln, font, size)) / 2, y, ln)
            y -= size * 1.8
        c.showPage()


# ---------------------------------------------------------------- CLI
def main():
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from contour.sam_segmenter import SegmentResult

    p = argparse.ArgumentParser(description="分割结果 -> PDF")
    p.add_argument("npz", help="分割结果 .npz")
    p.add_argument("-o", "--out", default="out/badapple_contour.pdf")
    p.add_argument("--mode", choices=["ascii", "contour", "filled"], default="ascii")
    p.add_argument("--theme", choices=list(THEMES), default="mono")
    p.add_argument("--cols", type=int, default=128)
    p.add_argument("--credits", default="", help="片尾字幕，多行用 | 分隔")
    a = p.parse_args()

    result = SegmentResult.load(a.npz)
    credits = [s for s in a.credits.split("|") if s]
    cfg = PDFConfig(mode=a.mode, theme=a.theme, cols=a.cols, credits=credits)
    builder = ContourPDFBuilder(cfg)
    builder.build(result, a.out)


if __name__ == "__main__":
    main()
