# -*- coding: utf-8 -*-
"""按 B 站 BV 号下载公开视频；默认只保存原视频，加 --split 才分离音视频。

用法：
    python download_bilibili.py BV1AbCdEfGhI
    python download_bilibili.py "https://www.bilibili.com/video/BV1AbCdEfGhI/"

默认只下载原视频（含音轨），转换时可直接读取：
    dist/input/待分离/BV号.mp4

加 --split 才会额外分离：
    dist/input/静视频/BV号.mp4    无音轨视频
    dist/input/纯音频/BV号.mp3    纯音频

仅下载无需登录即可公开访问的视频，不处理付费或受限内容。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from separate_media import AUDIO_DIR, SILENT_DIR, SOURCE_DIR, ensure_directories, separate_media

BV_PATTERN = re.compile(
    r"(?<![0-9A-Za-z])(BV[0-9A-Za-z]{10})(?![0-9A-Za-z])",
    re.IGNORECASE,
)


def extract_bvid(value: str) -> str:
    """从 BV 号或包含 BV 号的链接中提取标准 BV 前缀。"""
    match = BV_PATTERN.search(value.strip())
    if not match:
        raise ValueError(f"未找到有效 BV 号：{value}")
    return "BV" + match.group(1)[2:]


def download_original(video_id: str) -> Path:
    """下载并合并音视频，返回 dist/input/待分离/BV号.mp4。"""
    bvid = extract_bvid(video_id)
    ensure_directories()
    original_path = SOURCE_DIR / f"{bvid}.mp4"
    if original_path.is_file():
        print(f"复用已下载的原视频：{original_path}")
        return original_path

    try:
        import imageio_ffmpeg
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "缺少下载依赖，请先运行：python -m pip install -r requirements.txt"
        ) from exc

    url = f"https://www.bilibili.com/video/{bvid}"
    options = {
        "outtmpl": str(SOURCE_DIR / f"{bvid}.%(ext)s"),
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "postprocessors": [{
            "key": "FFmpegVideoRemuxer",
            "preferedformat": "mp4",
        }],
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "noplaylist": True,
    }

    print(f"正在下载 {bvid}：{url}")
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([url])
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(f"{bvid} 下载失败：{exc}") from exc

    if not original_path.is_file():
        raise RuntimeError(f"下载结束，但未找到预期文件：{original_path}")
    return original_path


def download_bilibili(video_id: str, split: bool = False,
                      overwrite: bool = False) -> tuple[Path, Path | None, Path | None]:
    """下载一个 BV 视频；split=True 时额外生成静视频和纯音频。"""
    original_path = download_original(video_id)
    if not split:
        return original_path, None, None
    silent_path, audio_path = separate_media(original_path, overwrite=overwrite)
    return original_path, silent_path, audio_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 BV 号下载 B 站公开视频；默认只保存原视频，加 --split 可额外分离",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python download_bilibili.py BV1AbCdEfGhI                  只下载原视频
  python download_bilibili.py "https://www.bilibili.com/video/BV1AbCdEfGhI/"
  python download_bilibili.py BV1AbCdEfGhI BV2AbCdEfGhI     批量下载原视频
  python download_bilibili.py BV1AbCdEfGhI --split          额外生成静视频和纯音频
  python download_bilibili.py BV1AbCdEfGhI --split --overwrite  重新分离已有文件
""",
    )
    parser.add_argument(
        "video_ids", nargs="*",
        help="一个或多个 BV 号，或包含 BV 号的视频链接；不填则提示输入",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="下载后额外分离出静视频和纯音频（默认不分离）",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="配合 --split 使用：覆盖已存在的静视频和音频；已下载的原视频始终复用",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.video_ids:
        try:
            raw = input("请输入 BV 号或视频链接（多个用空格分隔）：").strip()
        except KeyboardInterrupt:
            print("\n已取消下载。")
            return 130
        if not raw:
            print("未输入 BV 号或视频链接。", file=sys.stderr)
            return 1
        args.video_ids = raw.split()

    failed = 0
    try:
        for video_id in args.video_ids:
            print(f"\n=== {video_id} ===")
            try:
                original, silent, audio = download_bilibili(
                    video_id, split=args.split, overwrite=args.overwrite)
                print(f"原视频：{original}")
                if silent and audio:
                    print(f"静视频：{silent}")
                    print(f"纯音频：{audio}")
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                failed += 1
                print(f"处理失败：{exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n已取消下载。")
        return 130

    if failed:
        print(f"\n完成，但有 {failed} 个视频处理失败。", file=sys.stderr)
        return 1
    if args.split:
        print(f"\n全部完成。目录：\n  原视频：{SOURCE_DIR}\n  静视频：{SILENT_DIR}\n  纯音频：{AUDIO_DIR}")
    else:
        print(f"\n全部完成。原视频目录：{SOURCE_DIR}")
        print("如需静视频/纯音频，重新运行并加 --split，或直接使用 MediaSeparator。")
    return 0


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        for stream in (sys.stdout, sys.stderr):
            stream.reconfigure(encoding="utf-8", errors="replace")
    exit_code = main()
    if getattr(sys, "frozen", False) and len(sys.argv) == 1:
        try:
            input("\n按回车键退出...")
        except EOFError:
            pass
    raise SystemExit(exit_code)
