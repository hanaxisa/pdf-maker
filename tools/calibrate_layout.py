"""生成标定 PDF：用已知图案测出 Acrobat 真实渲染的字符格子尺寸。

打开 output/_calibration.pdf，按 Play（或让它在播放中截图），然后截图发回来。
图案含义：
  最外圈 1 格白框  -> 128/160 列 x 60 行 的真实边界
  正中 20x20 格方块 -> 如果渲染后是正方形，说明字符格比例正确（1:2）
  方框内十字线    -> 判断左右/上下是否被拉伸
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import ascii
from contour.sam_segmenter import FrameData, SegmentResult
from pdf.player_pdf import build_ascii_player

ROWS, COLS = ascii.ROWS, ascii.COLS


def pattern(shift: int):
    grid = np.zeros((ROWS, COLS), dtype=np.uint8)
    grid[0, :] = grid[-1, :] = 255          # 上下边框
    grid[:, 0] = grid[:, -1] = 255          # 左右边框
    top = ROWS // 2 - 10
    left = COLS // 2 - 10 + shift
    grid[top:top + 20, left:left + 20] = 255  # 中央方块（20 格宽 x 20 行高）
    grid[ROWS // 2, :] = 160                # 中间横线
    grid[:, COLS // 2] = 160                # 中间竖线
    return grid


def main():
    frames = [FrameData(gray=np.zeros((100, 100), np.uint8),
                        mask=np.zeros((100, 100), bool),
                        filled_gray=np.zeros((100, 100), np.uint8),
                        grid=pattern(i % 2 * 6)) for i in range(8)]
    result = SegmentResult(frames=frames, fps=4, model_name="calibration",
                           width=100, height=100)
    out = ROOT / "output" / "_calibration.pdf"
    build_ascii_player(result, None, str(out))
    print(f"标定文件：{out}")


if __name__ == "__main__":
    main()
