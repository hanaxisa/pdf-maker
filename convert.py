# -*- coding: utf-8 -*-
"""视频一键转 PDF 播放器。

最简单用法：
    1. 把视频放入 dist/input/ 文件夹
    2. 运行 python convert.py
    3. 按提示选择视频和处理范围，默认生成单页交互播放器

命令行用法：
    python convert.py dist/input/待分离/demo.mp4 --test
    python convert.py dist/input/待分离/demo.mp4 --collect       分析 + 采集数据，不生成 PDF
    python convert.py dist/input/待分离/demo.mp4 --audio dist/input/纯音频/bgm.mp3
    python convert.py dist/input/待分离/demo.mp4 --static --mode contour
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "dist" / "input"
SOURCE_DIR = INPUT_DIR / "待分离"
OUTPUT_DIR = ROOT / "output"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
MODE_NAMES = {"ascii": "ASCII 字符", "contour": "矢量轮廓", "filled": "灰度填充"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="视频一键转 PDF 播放器：SAM2 分割 + 单页交互播放",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python convert.py                           中文交互式转换
  python convert.py dist/input/demo.mp4 --test       转换前 10 秒为播放器
  python convert.py dist/input/demo.mp4              转换完整视频为播放器
  python convert.py dist/input/demo.mp4 --audio dist/input/bgm.mp3
  python convert.py dist/input/demo.mp4 --reuse-data 复用分割数据
  python convert.py dist/input/demo.mp4 --collect    只分析并采集数据，不生成 PDF
  python convert.py dist/input/demo.mp4 --static --mode contour  静态轮廓 PDF
""",
    )
    p.add_argument("video", nargs="?", help="视频路径；不填写时进入交互模式")
    p.add_argument("-o", "--out", help="输出 PDF 路径，默认 output/视频名_player.pdf")
    p.add_argument("--data", help="中间数据 .npz 路径，默认 output/视频名_data.npz")
    p.add_argument("--reuse-data", action="store_true",
                   help="复用已有 .npz，跳过分割，直接生成播放器")
    p.add_argument("--refit-grid", action="store_true",
                   help="从已存的灰度+mask 重新生成 ASCII 网格（cover 模式+应用 mask），"
                        "用于升级旧数据，无需重跑 SAM2")
    p.add_argument("--collect", action="store_true",
                   help="只分析并采集分割数据 .npz，不生成 PDF（之后可用 PDF生成工具.exe）")
    p.add_argument("--no-analyze", action="store_true",
                   help="跳过转换前的分析报告")
    p.add_argument("--static", action="store_true",
                   help="改为生成逐帧静态 PDF；默认生成单页交互播放器")
    p.add_argument("--audio", help="可选音频/含音轨视频路径，默认读取输入视频音轨")
    p.add_argument("--audio-track", type=int, default=0,
                   help="选择音轨序号（从 0 开始），多音轨视频/音频时生效，默认 0")
    p.add_argument("--color-mode", choices=["high", "low"], default="high",
                   help="彩色片渲染风格：high=高色差（默认），low=低色差（仅按亮度）")
    p.add_argument("--test", action="store_true", help="只处理前 10 秒")
    p.add_argument("--duration", type=float, default=0,
                   help="处理时长（秒），0=完整视频，默认 0")
    p.add_argument("--fps", type=int, default=12, help="每秒抽帧数，默认 12")
    p.add_argument("--mode", choices=list(MODE_NAMES), default="ascii",
                   help="静态 PDF 效果；播放器固定使用 ASCII，默认 ascii")
    p.add_argument("--theme", choices=["mono", "green", "amber", "blue", "paper"],
                   default="mono", help="静态 PDF 配色主题，默认 mono")
    p.add_argument("--cols", type=int, default=128,
                   help="静态 ASCII/填充模式列数，默认 128")
    p.add_argument("--model", choices=["tiny", "small", "base_plus", "large"],
                   default="tiny", help="SAM2 模型大小，默认 tiny")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="运行设备，默认 cpu")
    p.add_argument("--max-dim", type=int, default=192,
                   help="分割前画面最长边，默认 192")
    p.add_argument("--keyframe-interval", type=int, default=10,
                   help="每隔多少帧运行一次 SAM2；0=每帧都运行，默认 10")
    p.add_argument("--points-per-side", type=int, default=8,
                   help="SAM2 每边采样点数，默认 8")
    p.add_argument("--credits", default="",
                   help="可选片尾字幕，多行用 | 分隔")
    p.add_argument("--credit-sec", type=float, default=3.0,
                   help="播放器片尾停留秒数，默认 3")
    return p


def interactive_config(args: argparse.Namespace) -> argparse.Namespace:
    """无位置参数时，通过中文提示收集最必要的选项。"""
    print("\n=== 视频一键转 PDF 播放器 ===")
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    videos = sorted(
        p for p in SOURCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ) if SOURCE_DIR.is_dir() else []
    if not videos:
        videos = sorted(
            p for p in INPUT_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
    if videos:
        print("\ndist/input/待分离/ 中找到以下视频：")
        for i, path in enumerate(videos, 1):
            print(f"  {i}. {path.name}")
        raw = input("\n输入编号，或拖入其他视频文件 [默认 1]: ").strip()
        if not raw:
            raw = "1"
        if raw.isdigit() and 1 <= int(raw) <= len(videos):
            args.video = str(videos[int(raw) - 1])
        else:
            args.video = raw
    else:
        print(f"\n请先把视频放入：{INPUT_DIR}")
        args.video = input("也可以直接把视频文件拖到这里，然后回车：").strip()

    if not args.video:
        raise ValueError("未选择视频")

    print("\n处理范围：")
    print("  1. 前 10 秒（快速试片）")
    print("  2. 完整视频")
    scope = input("请选择 [默认 2]: ").strip()
    args.duration = 10 if scope == "1" else 0

    print("\n输出类型：")
    print("  1. 单页 PDF 播放器（推荐：ASCII 画面 + 播放控制 + 音频）")
    print("  2. 逐帧静态 PDF")
    print("  3. 只采集数据（.npz，之后用 PDF生成工具 合成）")
    output_type = input("请选择 [默认 1]: ").strip()
    args.static = output_type == "2"
    args.collect = output_type == "3"

    if args.static:
        print("\n静态 PDF 效果：")
        print("  1. ASCII 字符")
        print("  2. 矢量轮廓线")
        print("  3. 灰度色块填充")
        mode = input("请选择 [默认 1]: ").strip()
        args.mode = {"2": "contour", "3": "filled"}.get(mode, "ascii")
    elif not args.collect:
        args.mode = "ascii"
        audio = input("\n音频文件路径（可选，回车读取视频音轨）：").strip()
        if audio:
            args.audio = audio
        # 多条音轨时让用户选择
        try:
            from pdf.player_pdf import list_audio_tracks
            tracks = list_audio_tracks(args.audio or args.video)
            if len(tracks) > 1:
                print("\n检测到多条音轨：")
                for t in tracks:
                    lang = f" [{t['lang']}]" if t["lang"] else ""
                    default = "（默认）" if t["default"] else ""
                    print(f"  {t['index']}. {t['codec']} {t['rate']} Hz "
                          f"{t['channels']}{lang}{default}")
                raw = input("选择音轨编号 [默认 0]: ").strip()
                if raw.isdigit() and 0 <= int(raw) < len(tracks):
                    args.audio_track = int(raw)
        except Exception:
            pass
    return args


def resolve_file(raw_path: str, kind="文件") -> Path:
    """支持完整路径、相对路径、dist/input/ 内文件名和拖拽带来的引号。"""
    cleaned = raw_path.strip().strip('"').strip("'")
    path = Path(cleaned).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path.cwd() / path, ROOT / path, INPUT_DIR / path]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到{kind}：{cleaned}")


def output_paths(args: argparse.Namespace, video: Path) -> tuple[Path, Path]:
    duration = 10 if args.test else args.duration
    test_tag = f"_test{duration:g}s" if duration > 0 else ""
    data_path = Path(args.data) if args.data else (
        OUTPUT_DIR / f"{video.stem}{test_tag}_data.npz"
    )
    kind = f"{args.mode}_static" if args.static else "player"
    pdf_path = Path(args.out) if args.out else (
        OUTPUT_DIR / f"{video.stem}{test_tag}_{kind}.pdf"
    )
    if data_path.suffix.lower() != ".npz":
        data_path = data_path.with_suffix(".npz")
    if pdf_path.suffix.lower() != ".pdf":
        pdf_path = pdf_path.with_suffix(".pdf")
    return data_path, pdf_path


def print_analysis(video: Path, duration: float) -> None:
    """转换前快速分析，结论会直接影响后续分割与播放器效果。"""
    import analyze

    info = analyze.probe(video)
    print("\n=== 转换前分析 ===")
    print(f"  画面：{info['codec']} {info['width']}x{info['height']}  "
          f"{info['fps']:.2f} fps  共 {info['frames']} 帧 / {info['duration']:.1f} s")
    try:
        box = analyze.detect_content_box(str(video))
        if (box[2] - box[0]) < info["width"] or (box[3] - box[1]) < info["height"]:
            print(f"  黑边：有效画面 {box[2] - box[0]}x{box[3] - box[1]}，"
                  f"区域 {box}（分割时自动裁切）")
        else:
            print("  黑边：未检测到")
    except Exception as exc:
        print(f"  黑边：检测跳过（{exc}）")
    try:
        bright = analyze.sample_brightness(str(video), info["fps"], duration)
        count = max(bright["count"], 1)
        print(f"  亮度：平均 {bright['mean']:.1f}，"
              f"暗帧 {bright['dark'] / count:.0%}，亮帧 {bright['bright'] / count:.0%}")
    except Exception as exc:
        print(f"  亮度：采样跳过（{exc}）")
    if info["has_audio"]:
        volume = analyze.measure_volume(video)
        if volume:
            print(f"  音频：{info['audio_codec']} {info['audio_rate']} Hz，"
                  f"平均 {volume[0]:.1f} dB")
            if volume[0] < -50:
                print("  提示：音轨接近静音，播放器会自动改用静音。")
        else:
            print("  音频：有音轨，但未测到音量")
    else:
        print("  音频：无音轨，播放器将使用静音")


def run_conversion(args: argparse.Namespace) -> Path:
    video = resolve_file(args.video, "视频")
    duration = 10 if args.test else args.duration
    if duration < 0 or args.credit_sec < 0:
        raise ValueError("--duration 和 --credit-sec 不能小于 0")
    if args.fps <= 0 or args.cols <= 0 or args.max_dim <= 0:
        raise ValueError("--fps、--cols、--max-dim 必须大于 0")
    if args.keyframe_interval < 0 or args.points_per_side <= 0:
        raise ValueError("关键帧间隔不能小于 0，采样点数必须大于 0")
    if args.collect and args.reuse_data:
        raise ValueError("--collect 用于采集新数据，不能与 --reuse-data 同时使用")

    audio_source = resolve_file(args.audio, "音频") if (args.audio and not args.collect) else video
    data_path, pdf_path = output_paths(args, video)
    OUTPUT_DIR.mkdir(exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.collect:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n转换配置：")
    print(f"  视频：{video}")
    print(f"  范围：{'完整视频' if duration == 0 else f'前 {duration:g} 秒'}")
    if args.collect:
        print("  输出：仅采集分割数据（不生成 PDF）")
    elif args.static:
        print(f"  输出：逐帧静态 PDF / {MODE_NAMES[args.mode]} / {args.theme}")
    else:
        print("  输出：单页 ASCII 交互播放器（播放控制 + 内嵌音频）")
        print(f"  音频来源：{audio_source}")
        print(f"  音轨：第 {args.audio_track} 条")
        print(f"  色差版本：{'高色差' if args.color_mode == 'high' else '低色差'}")
    print(f"  设备：{args.device}，SAM2 关键帧间隔：{args.keyframe_interval}")
    print(f"  数据：{data_path}")
    if not args.collect:
        print(f"  PDF：{pdf_path}")

    if not args.no_analyze and not args.reuse_data:
        try:
            print_analysis(video, duration)
        except Exception as exc:
            print(f"[分析] 跳过：{exc}", file=sys.stderr)

    try:
        from contour.sam_segmenter import SAMSegmenter, SegmentResult
        from pdf.contour_pdf import ContourPDFBuilder, PDFConfig
        from pdf.player_pdf import build_ascii_player
    except ImportError as exc:
        raise RuntimeError(
            "缺少依赖，请先运行：python -m pip install -r requirements.txt"
        ) from exc

    if args.reuse_data:
        if not data_path.is_file():
            raise FileNotFoundError(f"找不到可复用的数据：{data_path}")
        print("\n[1/1] 正在复用分割数据并生成 PDF ...")
        result = SegmentResult.load(str(data_path))
        # 复用数据时从 model_name 推断网格尺寸，保证 PDF 布局与采集时一致
        import ascii as _ascii_mod
        _model = str(result.model_name) if getattr(result, "model_name", None) else args.model
        _gs = _ascii_mod.grid_size_for_model(_model)
        _ascii_mod.set_grid_size(*_gs)
        print(f"  网格尺寸：{_ascii_mod.COLS}x{_ascii_mod.ROWS}（模型 {_model}）")
    else:
        print("\n[1/2] 正在分割视频，首次运行可能需要下载模型 ...")
        # 根据模型设置网格分辨率：高档用更大的网格捕捉更多细节
        import ascii as _ascii_mod
        _gs = _ascii_mod.grid_size_for_model(args.model)
        _ascii_mod.set_grid_size(*_gs)
        print(f"  网格尺寸：{_ascii_mod.COLS}x{_ascii_mod.ROWS}（模型 {args.model}）")
        segmenter = SAMSegmenter(
            model_size=args.model,
            device=args.device,
            points_per_side=args.points_per_side,
            keyframe_interval=args.keyframe_interval,
        )
        result = segmenter.process_video(
            str(video),
            str(data_path),
            fps=args.fps,
            duration=duration,
            max_dim=args.max_dim,
        )
        if not args.collect:
            print("[2/2] 正在生成 PDF ...")

    if args.collect:
        print("\n数据采集完成！")
        print(f"分割数据：{data_path.resolve()}")
        print(f"下一步：用 PDF生成工具.exe 选择视频和该数据生成 PDF：{data_path}")
        return data_path

    if args.refit_grid:
        print("\n[refit] 从灰度重新生成 ASCII 网格（cover 模式 + 背景阈值）...")
        import ascii as _ascii
        _model = str(result.model_name) if getattr(result, "model_name", None) else args.model
        _gs = _ascii.grid_size_for_model(_model)
        _ascii.set_grid_size(*_gs)
        from pdf.player_pdf import _content_box
        _box = _content_box(result)
        _refit = 0
        for _f in result.frames:
            _src = _f.gray
            _crop = _src[_box[1]:_box[3], _box[0]:_box[2]]
            if _crop.size == 0:
                _crop = _src
            _f.grid = _ascii.fit_grid(_crop, mode="cover")
            _refit += 1
        print(f"[refit] 已用 {_ascii.COLS}x{_ascii.ROWS} 网格重新生成 {_refit} 帧")
        result.save(str(data_path))
        print(f"[refit] 已更新数据文件：{data_path.resolve()}")

    credits = [line for line in args.credits.split("|") if line]
    if args.static:
        config = PDFConfig(
            mode=args.mode,
            theme=args.theme,
            cols=args.cols,
            credits=credits,
        )
        ContourPDFBuilder(config).build(result, str(pdf_path))
    else:
        from pdf.player_pdf import build_char_player, tone_profile
        # 高色差：彩色/连续色调片走字符画渲染；低色差与黑底剪影走 ASCII 路径。
        profile = tone_profile(result)
        high = args.color_mode == "high"
        if high and profile["mode"] == "color" and video:
            build_char_player(
                result, str(video), str(pdf_path),
                credits=credits, credit_sec=args.credit_sec,
                audio_track=args.audio_track,
            )
        else:
            build_ascii_player(
                result, audio_source, str(pdf_path),
                credits=credits, credit_sec=args.credit_sec,
                audio_track=args.audio_track, color_analysis=high,
            )

    print("\n转换完成！")
    print(f"PDF 文件：{pdf_path.resolve()}")
    if not args.reuse_data:
        print(f"分割数据：{data_path.resolve()}")
        print("下次可用 --reuse-data 跳过模型分割，直接重新生成播放器。")
    if args.static:
        print("注意：本次按 --static 输出逐帧静态 PDF。")
    else:
        print("这是可交互的 PDF 播放器，请使用 Adobe Acrobat 打开，不要用浏览器阅读器。")
    return pdf_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if not args.video:
            args = interactive_config(args)
        run_conversion(args)
        return 0
    except KeyboardInterrupt:
        print("\n已取消转换。")
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\n转换失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
