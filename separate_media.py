# -*- coding: utf-8 -*-
"""批量或单独分离视频画面与音频。

默认处理 dist/input/待分离/ 中的全部视频：
    python separate_media.py

只处理指定视频：
    python separate_media.py "dist/input/待分离/demo.mp4"

输出：
    dist/input/静视频/同名文件.原视频扩展名
    dist/input/纯音频/同名文件.mp3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent / "dist"
)
INPUT_DIR = ROOT / "input"
SOURCE_DIR = INPUT_DIR / "待分离"
SILENT_DIR = INPUT_DIR / "静视频"
AUDIO_DIR = INPUT_DIR / "纯音频"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}


def ensure_directories() -> None:
    for directory in (SOURCE_DIR, SILENT_DIR, AUDIO_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_video(raw_path: str) -> Path:
    """支持完整路径、相对路径及 待分离/ 内的文件名。"""
    cleaned = raw_path.strip().strip('"').strip("'")
    path = Path(cleaned).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            Path.cwd() / path,
            ROOT / path,
            SOURCE_DIR / path,
            INPUT_DIR / path,
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到视频：{cleaned}")


def run_ffmpeg(source: Path, output: Path, stream_args: list[str], overwrite: bool) -> None:
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner", "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", str(source),
        *stream_args,
        str(output),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "未知 FFmpeg 错误"
        raise RuntimeError(f"生成 {output.name} 失败：\n{detail}")


def separate_media(source: str | Path, overwrite: bool = False) -> tuple[Path, Path]:
    """分离一个视频，返回静视频和纯音频路径。"""
    ensure_directories()
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到视频：{source}")

    silent_path = SILENT_DIR / f"{source.stem}{source.suffix.lower() or '.mp4'}"
    audio_path = AUDIO_DIR / f"{source.stem}.mp3"
    jobs = [
        (silent_path, ["-map", "0:v:0", "-an", "-c:v", "copy"], "静视频"),
        (audio_path, ["-map", "0:a:0", "-vn", "-c:a", "libmp3lame", "-b:a", "192k"], "纯音频"),
    ]

    for output, stream_args, label in jobs:
        if output.exists() and not overwrite:
            print(f"跳过已存在的{label}：{output}")
            continue
        print(f"正在生成{label}：{output.name}")
        run_ffmpeg(source, output, stream_args, overwrite)
    return silent_path, audio_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="分离视频画面和音频，输出到 dist/input/静视频 与 dist/input/纯音频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python separate_media.py                         批量处理 dist/input/待分离/ 中的视频
  python separate_media.py demo.mp4                 处理 dist/input/待分离/demo.mp4
  python separate_media.py "D:\\videos\\demo.mkv"    处理指定路径
  python separate_media.py --overwrite              重新生成已存在的输出
""",
    )
    parser.add_argument("videos", nargs="*", help="视频路径或文件名；不填则批量处理 待分离/")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的静视频和音频")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_directories()
    if args.videos:
        items = args.videos
    else:
        items = [
            str(path) for path in sorted(SOURCE_DIR.iterdir())
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS
        ]
        if not items:
            print(f"没有待分离视频，请先把文件放入：{SOURCE_DIR}", file=sys.stderr)
            return 1

    failed = 0
    try:
        for item in items:
            print(f"\n=== {Path(item).name} ===")
            try:
                source = resolve_video(item)
                silent_path, audio_path = separate_media(source, args.overwrite)
                print(f"静视频：{silent_path}")
                print(f"纯音频：{audio_path}")
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                failed += 1
                print(f"处理失败：{exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n已取消分离。")
        return 130

    if failed:
        print(f"\n完成，但有 {failed} 个文件处理失败。", file=sys.stderr)
        return 1
    print("\n全部视频分离完成。")
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
