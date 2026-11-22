# -*- coding: utf-8 -*-
"""基于 SAM2 的视频前景分割与灰度提取。

功能：
  1. 加载预训练 SAM2 大模型（hiera_tiny / small / base_plus / large）
  2. 逐帧自动生成掩码，筛选前景（高灰度）区域
  3. 提取前景像素的原始灰度值（模型判定哪些像素属于前景，保留其灰度）
  4. 输出结构化数据供 PDF 模块渲染

依赖：
  - torch / torchvision（retina_project 环境已有）
  - sam2（pip install sam2）

若 sam2 不可用，自动降级为传统 CV 分割（阈值 + findContours），
保证链路可跑通，便于在无 GPU 环境下快速验证。
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class FrameData:
    """单帧分割结果。"""
    gray: np.ndarray          # 原始灰度帧 (H, W) uint8
    mask: np.ndarray          # 前景掩码 (H, W) bool，True=前景
    filled_gray: np.ndarray   # 仅保留前景的灰度图，背景置 0 (H, W) uint8
    contours: list = field(default_factory=list)  # 轮廓点列表（可选）
    grid: Optional[np.ndarray] = None  # 从原分辨率采样的 48x128 字符网格（可选）
    color_grid: Optional[np.ndarray] = None  # 与 grid 逐格对齐的 BGR 颜色（可选）


@dataclass
class SegmentResult:
    """整段视频的分割结果。"""
    frames: List[FrameData]
    fps: int
    model_name: str
    width: int
    height: int

    def save(self, path: str):
        """保存为 .npz，便于 PDF 模块加载。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        grays = np.stack([f.gray for f in self.frames])
        masks = np.stack([f.mask for f in self.frames])
        filled = np.stack([f.filled_gray for f in self.frames])
        payload = dict(
            grays=grays,
            masks=masks.astype(np.uint8),
            filled=filled,
            fps=self.fps,
            model_name=self.model_name,
            width=self.width,
            height=self.height,
        )
        grids = [f.grid for f in self.frames]
        if all(g is not None for g in grids):
            payload["grids"] = np.stack(grids)
        colors = [f.color_grid for f in self.frames]
        if all(c is not None for c in colors):
            payload["color_grids"] = np.stack(colors)
        np.savez_compressed(path, **payload)
        print(f"[segment] 已保存 {path}  ({len(self.frames)} 帧, "
              f"{os.path.getsize(path) / 1024 / 1024:.2f} MB)")

    @staticmethod
    def load(path: str) -> "SegmentResult":
        data = np.load(path)
        grays = data["grays"]
        masks = data["masks"].astype(bool)
        filled = data["filled"]
        grids = data["grids"] if "grids" in data.files else None
        colors = data["color_grids"] if "color_grids" in data.files else None
        frames = [
            FrameData(gray=g, mask=m, filled_gray=f,
                      grid=None if grids is None else grids[i],
                      color_grid=None if colors is None else colors[i])
            for i, (g, m, f) in enumerate(zip(grays, masks, filled))
        ]
        return SegmentResult(
            frames=frames,
            fps=int(data["fps"]),
            model_name=str(data["model_name"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


# 项目根目录（contour/ 的上一级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"


def _bright_box(gray, thresh: int = 25):
    """找出画面真实内容的包围盒（去掉四周黑边）。"""
    height, width = gray.shape[:2]
    cols = np.where(gray.max(axis=0) > thresh)[0]
    rows = np.where(gray.max(axis=1) > thresh)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, width, height
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def _scale_box(box, sx: float, sy: float, width: int, height: int):
    """把小分辨率下的包围盒换算到原分辨率。"""
    x0, y0, x1, y1 = box
    return (max(0, int(x0 * sx)), max(0, int(y0 * sy)),
            min(width, int(x1 * sx)), min(height, int(y1 * sy)))


class SAMSegmenter:
    """SAM2 视频前景分割器。"""

    def __init__(self, model_size: str = "tiny", device: str = "cpu",
                 checkpoint_dir: Optional[str] = None,
                 points_per_side: int = 8,
                 keyframe_interval: int = 0):
        """
        Args:
            model_size: tiny | small | base_plus | large
            device: cpu | cuda
            checkpoint_dir: 权重目录，默认 <项目根>/checkpoints
            points_per_side: SAM2 每边采样点数，越小越快（4/8/16/32）
            keyframe_interval: 关键帧间隔，>0 时每 N 帧用 SAM2，其余用传统 CV
        """
        self.model_size = model_size
        self.device = device
        self._ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else _CHECKPOINT_DIR
        self._points_per_side = points_per_side
        self._keyframe_interval = keyframe_interval
        self._predictor = None
        self._mask_generator = None
        self._use_fallback = False
        self._foreground_bright: Optional[bool] = None
        self._model_name = f"sam2_hiera_{model_size}"

    def _resolve_checkpoint(self, ckpt_name: str) -> str:
        """解析权重路径：优先本地 checkpoints/ 目录，否则尝试从 hf-mirror 下载。"""
        local = self._ckpt_dir / ckpt_name
        if local.exists():
            return str(local)

        # 尝试从国内镜像下载
        print(f"[segment] 本地未找到 {ckpt_name}，尝试从 hf-mirror 下载 ...")
        try:
            import os
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from huggingface_hub import hf_hub_download
            repo = f"facebook/sam2-hiera-{self.model_size.replace('_', '-')}"
            path = hf_hub_download(repo_id=repo, filename=ckpt_name,
                                   local_dir=str(self._ckpt_dir))
            return path
        except Exception as e:
            print(f"[segment] 下载失败: {e}")
            raise

    def _load_sam2(self):
        """加载 SAM2 模型。"""
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            print("[segment] 未安装 sam2，降级为传统 CV 分割")
            self._use_fallback = True
            self._model_name = "traditional_cv"
            return

        cfg_map = {
            "tiny": "sam2_hiera_t.yaml",
            "small": "sam2_hiera_s.yaml",
            "base_plus": "sam2_hiera_b+.yaml",
            "large": "sam2_hiera_l.yaml",
        }
        ckpt_map = {
            "tiny": "sam2_hiera_tiny.pt",
            "small": "sam2_hiera_small.pt",
            "base_plus": "sam2_hiera_base_plus.pt",
            "large": "sam2_hiera_large.pt",
        }
        cfg = cfg_map[self.model_size]
        ckpt_name = ckpt_map[self.model_size]

        try:
            ckpt = self._resolve_checkpoint(ckpt_name)
        except Exception:
            print("[segment] 无法获取权重，降级为传统 CV 分割")
            self._use_fallback = True
            self._model_name = "traditional_cv"
            return

        print(f"[segment] 加载 SAM2: {cfg}  权重: {ckpt}")
        model = build_sam2(cfg, ckpt, device=self.device)
        self._mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=self._points_per_side,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            crop_n_layers=0,
            min_mask_region_area=100,
        )
        self._model_name = f"sam2_hiera_{self.model_size}"
        print(f"[segment] SAM2 加载完成 ({self.device}, points_per_side={self._points_per_side})")

    def _pick_polarity(self, gray: np.ndarray) -> bool:
        """整片统一前景极性：True=亮前景，False=暗前景。

        逐帧根据均值切换会把白底场景整帧反色，这里只在遇到第一张有内容的
        画面时确定一次，之后全部沿用同一极性。
        """
        if self._foreground_bright is None and float(gray.std()) > 10:
            self._foreground_bright = float(gray.mean()) < 128
        if self._foreground_bright is None:
            return float(gray.mean()) < 128
        return self._foreground_bright

    def _segment_frame_sam2(self, gray: np.ndarray) -> np.ndarray:
        """用 SAM2 分割单帧，返回前景掩码。

        前景极性由 _pick_polarity 全片统一决定，避免亮背景帧整帧反色。
        """
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        masks = self._mask_generator.generate(rgb)

        if not masks:
            return self._segment_frame_traditional(gray)

        bright_foreground = self._pick_polarity(gray)
        global_mean = gray.mean()
        foreground = np.zeros_like(gray, dtype=bool)
        total = gray.size

        for m in masks:
            seg = m["segmentation"]
            if not seg.any():
                continue
            # 跳过几乎覆盖全帧的掩码（通常是背景）
            if seg.sum() > total * 0.95:
                continue
            mean_in_mask = float(gray[seg].mean())
            if bright_foreground:
                # 暗背景 -> 前景是亮区
                if mean_in_mask > global_mean:
                    foreground |= seg
            else:
                # 亮背景 -> 前景是暗区
                if mean_in_mask < global_mean:
                    foreground |= seg

        if not foreground.any():
            return self._segment_frame_traditional(gray)
        return foreground

    def _segment_frame_traditional(self, gray: np.ndarray) -> np.ndarray:
        """传统 CV 分割：Otsu 阈值 + 形态学去噪，极性全片统一。"""
        bright_foreground = self._pick_polarity(gray)
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Otsu 自动阈值
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 形态学开运算去小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return (binary > 0) if bright_foreground else (binary == 0)

    def process_video(
        self,
        video_path: str,
        out_path: Optional[str] = None,
        fps: int = 12,
        duration: float = 0,
        max_dim: int = 256,
    ) -> SegmentResult:
        """处理整段视频。

        Args:
            video_path: 视频文件路径
            out_path: 输出 .npz 路径，None 则不保存
            fps: 采样帧率
            duration: 处理时长（秒），0=全片
            max_dim: 帧最大边长（缩放后送入模型，减小计算量）
        """
        if self._mask_generator is None and not self._use_fallback:
            self._load_sam2()
        self._foreground_bright = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算缩放比例
        scale = min(max_dim / src_w, max_dim / src_h, 1.0)
        out_w = max(1, int(src_w * scale))
        out_h = max(1, int(src_h * scale))

        dt = 1.0 / fps
        next_t = 0.0
        total = int(duration * fps) if duration > 0 else 10**9

        frames: List[FrameData] = []
        idx = 0
        frame_count = 0
        use_sam2 = not self._use_fallback
        kf_interval = max(0, self._keyframe_interval)
        try:
            import ascii as ascii_mod
        except ImportError:
            ascii_mod = None
        grid_box = None

        mode_desc = "全帧 SAM2" if (use_sam2 and kf_interval == 0) else \
                    (f"关键帧 SAM2 (每{kf_interval}帧)" if use_sam2 else "传统 CV")
        print(f"[segment] 视频 {src_w}x{src_h} @ {src_fps:.1f}fps, "
              f"缩放至 {out_w}x{out_h}, 采样 {fps}fps, "
              f"模型={self._model_name}, 模式={mode_desc}")

        while len(frames) < total:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / src_fps
            if t + 1e-9 >= next_t:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = gray_full
                if scale < 1.0:
                    gray = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)

                # 关键帧模式：SAM2 只跑关键帧，其余帧用传统 CV
                if use_sam2 and (kf_interval == 0 or frame_count % kf_interval == 0):
                    mask = self._segment_frame_sam2(gray)
                else:
                    mask = self._segment_frame_traditional(gray)

                filled = np.where(mask, gray, 0).astype(np.uint8)
                grid = None
                color_grid = None
                if ascii_mod is not None:
                    if grid_box is None:
                        grid_box = _scale_box(_bright_box(gray), src_w / out_w, src_h / out_h,
                                              src_w, src_h)
                    bx0, by0, bx1, by1 = grid_box
                    crop = gray_full[by0:by1, bx0:bx1]
                    # 按源画面比例决定 cover / letterbox：竖屏视频用 cover 会裁掉头部
                    grid_mode = ascii_mod.fit_mode_for(crop.shape[1], crop.shape[0])
                    grid = ascii_mod.fit_grid(crop if crop.size else gray_full,
                                              mode=grid_mode, bg_threshold=0)
                    # 同一裁切几何下再采一份颜色网格，供彩色模式的色度感知使用
                    color_crop = frame[by0:by1, bx0:bx1]
                    color_grid = ascii_mod.fit_grid_color(color_crop if color_crop.size else frame,
                                                          mode=grid_mode)
                frames.append(FrameData(gray=gray, mask=mask, filled_gray=filled,
                                        grid=grid, color_grid=color_grid))

                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"  已处理 {frame_count} 帧 ...")
                next_t += dt
            idx += 1

        cap.release()
        print(f"[segment] 完成，共 {len(frames)} 帧")

        result = SegmentResult(
            frames=frames,
            fps=fps,
            model_name=self._model_name,
            width=out_w,
            height=out_h,
        )
        if out_path:
            result.save(out_path)
        return result


# ---------------------------------------------------------------- CLI
def main():
    import argparse

    p = argparse.ArgumentParser(description="SAM2 视频前景分割")
    p.add_argument("video", help="输入视频路径")
    p.add_argument("-o", "--out", default="out/segmented.npz", help="输出 .npz")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--duration", type=float, default=0, help="秒，0=全片")
    p.add_argument("--model", choices=["tiny", "small", "base_plus", "large"],
                   default="tiny")
    p.add_argument("--max-dim", type=int, default=192)
    p.add_argument("--points-per-side", type=int, default=8,
                   help="SAM2 采样点数，越小越快（4/8/16）")
    p.add_argument("--keyframe-interval", type=int, default=10,
                   help="关键帧间隔，>0 时每 N 帧用 SAM2，其余用传统 CV；0=全帧 SAM2")
    a = p.parse_args()

    seg = SAMSegmenter(model_size=a.model,
                       points_per_side=a.points_per_side,
                       keyframe_interval=a.keyframe_interval)
    seg.process_video(a.video, a.out, fps=a.fps, duration=a.duration,
                      max_dim=a.max_dim)


if __name__ == "__main__":
    main()
