"""轮廓识别模块：基于预训练大模型（SAM2）的视频前景分割与灰度提取。"""
from .sam_segmenter import SAMSegmenter

__all__ = ["SAMSegmenter"]
