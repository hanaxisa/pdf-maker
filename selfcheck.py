# -*- coding: utf-8 -*-
"""项目自检：检查环境、依赖、模型、目录、视频、音频和输出，并给出原因与处理方法。

用法：
    python selfcheck.py                       # 常规自检
    python selfcheck.py --fix                 # 自动创建缺失的工作目录
    python selfcheck.py --deep                # 额外检查 torch / CUDA
    python selfcheck.py --error "报错文本"    # 根据报错内容诊断原因
    python selfcheck.py --log error.txt       # 从文件读取报错内容
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "dist" / "input"
SOURCE_DIR = INPUT_DIR / "待分离"
SILENT_DIR = INPUT_DIR / "静视频"
AUDIO_DIR = INPUT_DIR / "纯音频"
OUTPUT_DIR = ROOT / "output"
CHECKPOINT_DIR = ROOT / "checkpoints"
DIST_DIR = ROOT / "dist"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}

REQUIRED = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "reportlab": "reportlab",
    "pypdf": "pypdf",
    "PIL": "Pillow",
    "imageio_ffmpeg": "imageio-ffmpeg",
}
OPTIONAL = {
    "yt_dlp": ("yt-dlp", "按 BV 号下载"),
    "torch": ("torch", "SAM2 大模型"),
    "sam2": ("sam2", "SAM2 大模型"),
    "huggingface_hub": ("huggingface_hub", "模型自动下载"),
    "skimage": ("scikit-image", "可选的图像增强"),
}

results: list[dict] = []


def add(level: str, name: str, detail: str = "", fix: str = "") -> None:
    results.append({"level": level, "name": name, "detail": detail, "fix": fix})


def run_ffmpeg(args: list[str]) -> str:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-hide_banner", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return (result.stderr or "") + (result.stdout or "")


def parse_hms(text: str) -> float:
    seconds = 0.0
    for part in text.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def probe_media(path: Path) -> dict:
    text = run_ffmpeg(["-i", str(path)])
    duration = re.search(r"Duration:\s*([\d:.]+)", text)
    return {
        "has_video": "Video:" in text,
        "has_audio": "Audio:" in text,
        "duration": parse_hms(duration.group(1)) if duration else 0.0,
        "text": text,
    }


def measure_volume(path: Path) -> float | None:
    text = run_ffmpeg(["-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", os.devnull])
    match = re.search(r"mean_volume:\s*(-?[\d.]+) dB", text)
    return float(match.group(1)) if match else None


def check_python() -> None:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 9):
        add("FAIL", f"Python {version}", "版本过低，项目需要 3.9 及以上",
            "安装 Python 3.10+，并用新 Python 重新执行 pip install")
    else:
        add("PASS", f"Python {version} ({'64 位' if sys.maxsize > 2**32 else '32 位'})")


def find_ready_python() -> list[str]:
    """在 Windows 上查找其它已装好本项目核心依赖的 Python。"""
    if os.name != "nt":
        return []
    try:
        out = subprocess.run(["py", "-0p"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=8)
    except Exception:
        return []
    interpreters = re.findall(r"([A-Za-z]:\\[^\r\n]*?python\.exe)", out.stdout or "")
    ready = []
    for exe in dict.fromkeys(interpreters):
        try:
            if Path(exe).resolve() == Path(sys.executable).resolve():
                continue
            probe = subprocess.run([exe, "-c", "import cv2, reportlab, imageio_ffmpeg"],
                                   capture_output=True, timeout=30)
            if probe.returncode == 0:
                ready.append(exe)
        except Exception:
            continue
    return ready


def check_dependencies() -> None:
    missing = []
    for module, package in REQUIRED.items():
        try:
            __import__(module)
        except Exception as exc:
            missing.append((package, str(exc)))
    if missing:
        names = "、".join(package for package, _ in missing)
        ready = find_ready_python()
        fix = f'"{sys.executable}" -m pip install -r requirements.txt'
        if ready:
            fix += f"，或把编辑器解释器切换为：{ready[0]}"
        add("FAIL", f"必需依赖缺失：{names}",
            "；".join(error for _, error in missing), fix)
    else:
        add("PASS", "必需依赖全部可用", " / ".join(REQUIRED))

    absent = [f"{package}（{purpose}）" for module, (package, purpose) in OPTIONAL.items()
              if importlib.util.find_spec(module) is None]
    if absent:
        add("WARN", f"可选依赖缺失：{'、'.join(absent)}",
            "缺少下载或大模型能力，相关功能会报 ModuleNotFoundError",
            "python -m pip install -r requirements.txt")
    else:
        add("PASS", "可选依赖全部可用", " / ".join(package for package, _ in OPTIONAL.values()))


def check_ffmpeg() -> None:
    try:
        import imageio_ffmpeg
        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        add("FAIL", "找不到 FFmpeg", str(exc),
            "python -m pip install --upgrade imageio-ffmpeg")
        return
    if not ffmpeg.is_file():
        add("FAIL", "FFmpeg 不可用", f"路径不存在：{ffmpeg}",
            "python -m pip install --force-reinstall imageio-ffmpeg")
        return
    try:
        out = subprocess.run([str(ffmpeg), "-version"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        if out.returncode == 0:
            add("PASS", "FFmpeg 可用", ffmpeg.name)
        else:
            add("FAIL", "FFmpeg 启动失败", out.stderr.strip()[:200],
                "python -m pip install --force-reinstall imageio-ffmpeg")
    except Exception as exc:
        add("FAIL", "FFmpeg 启动失败", str(exc), "重新安装 imageio-ffmpeg")


def check_weights() -> None:
    weights = sorted(CHECKPOINT_DIR.glob("*.pt")) if CHECKPOINT_DIR.is_dir() else []
    if not weights:
        add("WARN", "checkpoints/ 中没有 SAM2 权重",
            "首次运行 python convert.py 分割时会自动下载 sam2_hiera_tiny.pt（约 150 MB）",
            "保持网络连接直接转换；或手动下载权重放入 checkpoints/ 根目录（不要嵌套文件夹）")
        return
    detail = "、".join(f"{p.name} ({p.stat().st_size / 1024 / 1024:.0f} MB)" for p in weights)
    add("PASS", "SAM2 权重已就位", detail)


def check_gpu(deep: bool) -> None:
    if importlib.util.find_spec("torch") is None:
        return
    if not deep:
        add("INFO", "GPU 检查已跳过", "加 --deep 可检测 CUDA 是否可用")
        return
    try:
        import torch
        if torch.cuda.is_available():
            add("PASS", "CUDA 可用", f"{torch.cuda.get_device_name(0)}，可加 --device cuda")
        else:
            add("INFO", "CUDA 不可用，将使用 CPU",
                "CPU 分割较慢，建议保持 --keyframe-interval 10")
    except Exception as exc:
        add("WARN", "torch 导入失败", str(exc), "重新安装 torch：python -m pip install torch torchvision")


def check_directories(fix: bool) -> None:
    wanted = [SOURCE_DIR, SILENT_DIR, AUDIO_DIR, OUTPUT_DIR]
    missing = [p for p in wanted if not p.is_dir()]
    if not missing:
        add("PASS", "工作目录完整", "dist/input/{待分离,静视频,纯音频} 与 output/ 均存在")
        return
    detail = "、".join(str(p.relative_to(ROOT)) for p in missing)
    if fix:
        for path in missing:
            path.mkdir(parents=True, exist_ok=True)
        add("PASS", "已创建缺失目录", detail)
    else:
        add("WARN", f"缺少目录：{detail}", "脚本运行时会自动创建，但不是所有阶段都会",
            "python selfcheck.py --fix")


def check_videos() -> None:
    if importlib.util.find_spec("imageio_ffmpeg") is None:
        add("INFO", "跳过视频检查", "当前解释器缺少 imageio-ffmpeg，无法读取视频信息")
        return
    videos = sorted(p for p in SOURCE_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS) if SOURCE_DIR.is_dir() else []
    if not videos:
        add("WARN", "dist/input/待分离 中没有视频",
            "转换与分析都需要先准备视频",
            "把视频放入 dist/input/待分离/，或运行 python download_bilibili.py BV号")
        return

    for path in videos:
        try:
            info = probe_media(path)
        except Exception as exc:
            add("FAIL", f"{path.name} 无法读取", str(exc), "确认文件未损坏，或重新下载")
            continue
        if not info["has_video"]:
            add("FAIL", f"{path.name} 没有视频流", "文件可能只是音频或已损坏",
                "用 separate_media.py 重新分离，或更换片源")
            continue
        if info["duration"] <= 1:
            add("FAIL", f"{path.name} 时长异常", f"{info['duration']:.1f} s",
                "文件可能下载不完整，删除后重新下载")
            continue

        problems = []
        fixes = []
        if not info["has_audio"]:
            sibling = AUDIO_DIR / f"{path.stem}.mp3"
            if sibling.is_file():
                problems.append("视频无音轨，但存在同名 MP3")
                fixes.append(f'--audio "{sibling}"')
            else:
                problems.append("视频无音轨")
                fixes.append("运行 python separate_media.py 或单独导出音频后加 --audio")
        else:
            volume = measure_volume(path)
            if volume is not None and volume < -50:
                problems.append(f"音轨近似静音（mean {volume:.1f} dB）")
                sibling = AUDIO_DIR / f"{path.stem}.mp3"
                if sibling.is_file():
                    fixes.append(f'改用 --audio "{sibling}"；若该 MP3 也静音，请在剪辑软件里重新导出音频')
                else:
                    fixes.append("从剪辑工程单独导出等长 MP3/WAV，再用 --audio 指定")
            elif volume is not None:
                problems.append(f"音频正常（mean {volume:.1f} dB）")

        sibling = AUDIO_DIR / f"{path.stem}.mp3"
        if sibling.is_file() and info["has_audio"]:
            try:
                audio_info = probe_media(sibling)
                diff = abs(audio_info["duration"] - info["duration"])
                if diff > 1.0:
                    problems.append(f"同名 MP3 与视频时长相差 {diff:.1f} s")
                    fixes.append("重新用 separate_media.py 分离，避免音画不同步")
            except Exception:
                pass

        if any("静音" in p or "无音轨" in p or "时长相差" in p for p in problems):
            add("WARN", f"{path.name} 音频需要注意", "；".join(problems), "；".join(fixes))
        else:
            add("PASS", f"{path.name} 可用",
                f"{info['duration']:.1f} s，{'有音轨' if info['has_audio'] else '无音轨'}"
                + (f"，{problems[0]}" if problems else ""))


def check_outputs() -> None:
    if not OUTPUT_DIR.is_dir():
        return
    files = [p for p in OUTPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".npz"}]
    if not files:
        add("INFO", "output/ 中还没有转换结果", "运行 python convert.py 后会生成 PDF 和 npz")
        return
    problems = []
    for path in files:
        size = path.stat().st_size
        if size == 0:
            problems.append(f"{path.name} 是 0 字节")
            continue
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                pages = len(PdfReader(str(path)).pages)
                if pages == 0:
                    problems.append(f"{path.name} 页数为 0")
                elif pages == 1:
                    add("PASS", path.name, f"单页播放器 PDF，{size / 1024 / 1024:.1f} MB")
                else:
                    add("PASS", path.name, f"逐帧静态 PDF，共 {pages} 页，{size / 1024 / 1024:.1f} MB")
            except Exception as exc:
                problems.append(f"{path.name} 无法解析：{exc}")
        else:
            try:
                import numpy as np
                with np.load(path, allow_pickle=False) as data:
                    frames = data["masks"].shape[0] if "masks" in data.files else 0
                    missing = {"grays", "masks", "filled"} - set(data.files)
                if frames == 0:
                    problems.append(f"{path.name} 没有有效帧")
                elif missing:
                    problems.append(f"{path.name} 缺少字段：{'、'.join(sorted(missing))}")
                else:
                    add("PASS", path.name, f"分割数据 {frames} 帧，{size / 1024 / 1024:.1f} MB")
            except Exception as exc:
                problems.append(f"{path.name} 无法读取：{exc}")
    if problems:
        add("FAIL", f"output/ 中有 {len(problems)} 个异常文件", "；".join(problems),
            "删除 0 字节或损坏的文件后重新转换；npz 损坏就删掉让它重新分割")


def check_disk() -> None:
    try:
        usage = shutil.disk_usage(str(ROOT))
        free_gb = usage.free / 1024 ** 3
        if free_gb < 2:
            add("WARN", f"磁盘剩余空间偏少：{free_gb:.1f} GB",
                "转换会额外产生 npz、PDF 和临时视频", "清理磁盘或把项目移到空间更大的分区")
        else:
            add("PASS", f"磁盘剩余空间 {free_gb:.1f} GB")
    except Exception as exc:
        add("INFO", "磁盘空间检查失败", str(exc))


def check_exe() -> None:
    exes = [DIST_DIR / "BilibiliDownloader.exe", DIST_DIR / "MediaSeparator.exe"]
    found = [p.name for p in exes if p.is_file()]
    if found:
        add("INFO", "已打包 EXE", "、".join(found))
    else:
        add("INFO", "未发现打包好的 EXE", "只影响免 Python 使用，不影响脚本转换")


ERROR_RULES = [
    (("no module named", "modulenotfounderror"), "缺少 Python 依赖（报错里的模块名就是缺失项）",
     "在项目根目录执行：python -m pip install -r requirements.txt；缺单个包时 python -m pip install 包名"),
    (("cuda out of memory", "out of memory"), "显存或内存不足",
     "降低参数：--max-dim 96 --fps 6，或用 --device cpu；也可把视频切成两段分别处理"),
    (("memoryerror",), "内存不足", "降低 --max-dim 和 --fps，关闭其它占内存程序"),
    (("ffmpeg", "not recognized", "不是内部或外部命令"), "FFmpeg 不可用",
     "python -m pip install --force-reinstall imageio-ffmpeg"),
    (("无法打开视频", "failed to open", "could not open", "找不到视频", "no such file or directory"),
     "视频路径或文件本身有问题",
     "把视频放到 dist/input/待分离/ 并确认扩展名正确；先用 separate_media.py 重新分离成 mp4"),
    (("downloaderror", "sign in to confirm", "403", "412", "下载失败"),
     "B 站下载被拒绝：链接需要登录、视频受限，或 yt-dlp 版本过旧",
     "确认视频无需登录即可播放；升级：python -m pip install -U yt-dlp；换公开视频再试"),
    (("permissionerror", "access is denied", "拒绝访问", "being used by another process"), "文件被占用或没有写权限",
     "关闭正在打开该 PDF/视频的 Acrobat 或播放器；换到非系统盘目录重试"),
    (("no space left", "errno 28", "磁盘空间不足"), "磁盘写满",
     "清理空间；把项目移到剩余空间更大的分区"),
    (("keyerror", "npz"), "分割数据字段缺失或版本不匹配",
     "删除 output/ 下对应的 *_data.npz 后重新运行 convert.py"),
    (("ttfont", "simhei", "font"), "片尾字幕字体缺失",
     "Windows 自带 simhei.ttf；其它系统请修改 ascii.py 中的字体路径"),
    (("javascript", "acrobat", "多媒体", "空白"), "播放器 PDF 需要 Acrobat 的 JavaScript/多媒体支持",
     "用 Adobe Acrobat 打开，按提示启用 JavaScript 和多媒体；不要用浏览器内置 PDF 阅读器"),
    (("unicodedecodeerror", "gbk", "codec can't decode"), "控制台编码不一致",
     "PowerShell 中先执行：$env:PYTHONUTF8=1，再重新运行"),
    (("importerror", "dll load failed"), "依赖安装不完整或位数不匹配",
     "用 64 位 Python 重新执行 python -m pip install -r requirements.txt；必要时先 pip uninstall 再装"),
    (("eoferror",), "交互输入被中断（常见于双击运行后窗口关闭）",
     "在 PowerPoint/终端里手动运行，或使用命令行参数避免交互"),
    (("syntaxerror",), "代码或 Python 版本不匹配", "确认使用 Python 3.9+，并更新到与仓库一致的代码"),
]


def diagnose(text: str) -> None:
    content = text.lower()
    for keywords, cause, fix in ERROR_RULES:
        if any(keyword in content for keyword in keywords):
            add("FAIL", f"报错诊断：{cause}", "已根据报错关键字匹配到原因", fix)
            return
    add("INFO", "未能识别报错关键字", "请把完整报错（含 Traceback 最后几行）提供给维护者",
        "先运行 python selfcheck.py 排除环境问题，再重试")


def print_report() -> int:
    levels = {"PASS": 0, "INFO": 0, "WARN": 0, "FAIL": 0}
    for item in results:
        levels[item["level"]] += 1
    print("\n================ 自检结果 ================")
    for item in results:
        line = f"[{item['level']}] {item['name']}"
        if item["detail"]:
            line += f"：{item['detail']}"
        print(line)
        if item["fix"] and item["level"] in {"WARN", "FAIL"}:
            print(f"       处理：{item['fix']}")

    print("==========================================")
    print(f"通过 {levels['PASS']} 项，提示 {levels['INFO']} 项，"
          f"警告 {levels['WARN']} 项，失败 {levels['FAIL']} 项")
    if levels["FAIL"]:
        print("结论：存在需要处理的问题，按上面的「处理」建议修复后重新自检。")
        return 1
    if levels["WARN"]:
        print("结论：可以继续使用，但建议按提示处理警告项。")
        return 0
    print("结论：环境与素材均正常，可以开始转换。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BadApplePDF 项目自检：定位环境、素材和输出问题并给出处理方法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python selfcheck.py
  python selfcheck.py --fix
  python selfcheck.py --deep
  python selfcheck.py --error "ModuleNotFoundError: No module named 'sam2'"
  python selfcheck.py --log error.txt
""",
    )
    parser.add_argument("--fix", action="store_true", help="自动创建缺失的工作目录")
    parser.add_argument("--deep", action="store_true", help="额外导入 torch 检测 CUDA")
    parser.add_argument("--error", help="直接传入报错文本进行诊断")
    parser.add_argument("--log", help="从文本文件读取报错内容进行诊断")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print("BadApplePDF 自检")
    print(f"项目目录：{ROOT}")
    print(f"解释器  ：{sys.executable}")
    check_python()
    check_dependencies()
    check_ffmpeg()
    check_weights()
    check_gpu(args.deep)
    check_directories(args.fix)
    check_videos()
    check_outputs()
    check_disk()
    check_exe()

    error_text = args.error or ""
    if args.log:
        try:
            error_text += "\n" + Path(args.log).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            add("WARN", f"无法读取日志文件：{args.log}", str(exc))
    if error_text.strip():
        diagnose(error_text)

    return print_report()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
