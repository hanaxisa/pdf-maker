# -*- coding: utf-8 -*-
"""视频分析：一条命令查看元数据、画面亮度、黑边和音频，并给出转换参数建议。

用法：
    python analyze.py                                  # 自动分析 dist/input/待分离 中唯一的视频
    python analyze.py dist/input/demo.mp4
    python analyze.py dist/input/demo.mp4 --seconds 60 # 只采样前 60 秒画面
    python analyze.py --data output/demo_data.npz      # 额外分析 SAM2 分割数据
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    sys.exit(
        f"缺少依赖 {exc.name}，当前解释器：{sys.executable}\n"
        f"处理一：用当前解释器安装依赖：\n"
        f'  "{sys.executable}" -m pip install -r requirements.txt\n'
        f"处理二：如果本机已有一个装好依赖的 Python（本项目验证过 3.12），"
        f"把编辑器右下角的解释器切换过去再运行。"
    )

ROOT = Path(__file__).resolve().parent
FROZEN = getattr(sys, "frozen", False)
HERE = Path(sys.executable).resolve().parent if FROZEN else ROOT
if FROZEN:
    PROJECT_ROOT = HERE.parent if (HERE.parent / "requirements.txt").is_file() else HERE
    INPUT_DIR = HERE / "input"
else:
    PROJECT_ROOT = HERE
    INPUT_DIR = HERE / "dist" / "input"
SOURCE_DIR = INPUT_DIR / "待分离"
AUDIO_DIR = INPUT_DIR / "纯音频"
OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}


def detect_content_box(video: str, thresh: int = 25, sample_step: int = 24):
    """扫描采样帧，找出真实画面区域（去掉左右/上下纯黑边）。"""
    cap = cv2.VideoCapture(video)
    acc_c = np.zeros(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    acc_r = np.zeros(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % sample_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            acc_c = np.maximum(acc_c, gray.max(axis=0))
            acc_r = np.maximum(acc_r, gray.max(axis=1))
        i += 1
    cap.release()
    cx = np.where(acc_c > thresh)[0]
    ry = np.where(acc_r > thresh)[0]
    if len(cx) == 0 or len(ry) == 0:
        return 0, 0, int(acc_c.size), int(acc_r.size)
    return (int(cx.min()) + 2, int(ry.min()) + 2,
            int(cx.max()) + 1 - 2, int(ry.max()) + 1 - 2)

_FFMPEG = None


def ffmpeg_exe() -> str:
    global _FFMPEG
    if _FFMPEG is None:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG


def run_ffmpeg(args: list[str]) -> str:
    result = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return (result.stderr or "") + (result.stdout or "")


def pick_video(raw: str | None) -> Path:
    if raw:
        path = Path(raw.strip().strip('"').strip("'")).expanduser()
        if path.is_file():
            return path.resolve()
        for base in (Path.cwd(), PROJECT_ROOT, INPUT_DIR, SOURCE_DIR):
            candidate = base / path
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"找不到视频：{raw}")

    videos = sorted(
        p for p in SOURCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ) if SOURCE_DIR.is_dir() else []
    if not videos:
        raise FileNotFoundError(f"dist/input/待分离 中没有视频，请先放入视频或指定路径：{SOURCE_DIR}")
    if len(videos) > 1:
        if sys.stdin.isatty():
            print("待分离目录中有多个视频：")
            for i, path in enumerate(videos, 1):
                print(f"  {i}. {path.name}")
            raw = input("请选择视频编号 [默认 1]: ").strip()
            if not raw:
                return videos[0]
            if raw.isdigit() and 1 <= int(raw) <= len(videos):
                return videos[int(raw) - 1]
            raise ValueError(f"无效的编号：{raw}")
        names = "\n".join(f"  {p.name}" for p in videos)
        raise ValueError(f"待分离目录中有多个视频，请指定其中一个：\n{names}")
    return videos[0]


def parse_hms(text: str) -> float:
    parts = text.strip().split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def probe(path: Path) -> dict:
    info = {
        "codec": "?", "width": 0, "height": 0, "fps": 0.0, "frames": 0,
        "duration": 0.0, "audio_codec": "", "audio_rate": 0,
        "audio_channels": "", "has_audio": False,
    }
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["fps"] = cap.get(cv2.CAP_PROP_FPS) or 0.0
        info["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    text = run_ffmpeg(["-i", str(path)])
    match = re.search(r"Video:\s*(\w+).*?,\s*(\d{2,5})x(\d{2,5})", text)
    if match:
        info["codec"] = match.group(1)
        info["width"] = int(match.group(2))
        info["height"] = int(match.group(3))
    match = re.search(r"Video:.*?([\d.]+)\s+fps", text)
    if match and not info["fps"]:
        info["fps"] = float(match.group(1))
    match = re.search(r"Duration:\s*([\d:.]+)", text)
    if match:
        info["duration"] = parse_hms(match.group(1))
    match = re.search(r"Audio:\s*(\w+).*?(\d+)\s+Hz,\s*([^,]+)", text)
    if match:
        info["has_audio"] = True
        info["audio_codec"] = match.group(1)
        info["audio_rate"] = int(match.group(2))
        info["audio_channels"] = match.group(3).strip()
    if not info["duration"] and info["fps"] and info["frames"]:
        info["duration"] = info["frames"] / info["fps"]
    return info


def measure_volume(path: Path, seconds: float = 0) -> tuple[float, float] | None:
    args = ["-i", str(path), "-vn"]
    if seconds > 0:
        args += ["-t", str(seconds)]
    args += ["-af", "volumedetect", "-f", "null", os.devnull]
    text = run_ffmpeg(args)
    mean = re.search(r"mean_volume:\s*(-?[\d.]+) dB", text)
    peak = re.search(r"max_volume:\s*(-?[\d.]+) dB", text)
    if not mean:
        return None
    return float(mean.group(1)), float(peak.group(1)) if peak else -999.0


def sample_brightness(path: Path, fps: float, seconds: float = 0) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    step = max(1, int(round((fps or 30) / 1.0)))   # 每秒 1 帧
    limit = int(seconds * (fps or 30)) if seconds > 0 else 0
    means, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            means.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        i += 1
        if limit and i >= limit:
            break
    cap.release()
    if not means:
        raise RuntimeError("没有读到任何帧")
    data = np.array(means, dtype=np.float32)
    return {
        "count": len(data),
        "mean": float(data.mean()),
        "min": float(data.min()),
        "max": float(data.max()),
        "p10": float(np.percentile(data, 10)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "dark": int((data < 60).sum()),
        "bright": int((data > 180).sum()),
    }


def analyze_npz(path: Path) -> dict | None:
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        print(f"  [警告] 无法读取分割数据：{exc}")
        return None
    if "masks" not in data.files:
        print(f"  [警告] 分割数据缺少 masks 字段，实际字段：{', '.join(data.files)}")
        return None
    masks = np.asarray(data["masks"])
    grays = np.asarray(data["grays"]) if "grays" in data.files else None
    ratio = masks.reshape(len(masks), -1).mean(axis=1)
    result = {
        "frames": len(masks),
        "width": int(masks.shape[2]),
        "height": int(masks.shape[1]),
        "fps": int(data["fps"]) if "fps" in data.files else 0,
        "ratio_mean": float(ratio.mean()),
        "ratio_p10": float(np.percentile(ratio, 10)),
        "ratio_p90": float(np.percentile(ratio, 90)),
        "empty": int((ratio < 0.01).sum()),
        "full": int((ratio > 0.75).sum()),
    }
    if grays is not None:
        fg_gray = (grays * masks).sum(axis=(1, 2)) / np.maximum(masks.sum(axis=(1, 2)), 1)
        result["fg_gray"] = float(fg_gray.mean())
    return result


def human_size(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024 else f"{size / 1024:.0f} KB"


def recommend(info: dict, bright: dict, box: tuple[int, int, int, int] | None) -> dict:
    fps = 12
    theme = "mono"
    if bright["mean"] > 170:
        theme = "paper"
    dark_ratio = bright["dark"] / max(bright["count"], 1)
    bright_ratio = bright["bright"] / max(bright["count"], 1)
    # 画面以暗场为主（或亮帧很少）时，恒定黑底最稳定；明暗交替明显才跟随翻转
    bgmode = "stable" if (dark_ratio > 0.6 or bright_ratio < 0.1 or bright["p50"] < 110) else "switch"
    contrast = 2.0 if (bright["p90"] - bright["p10"]) < 80 else 1.7
    inset = 10 if box else 0
    return {"fps": fps, "theme": theme, "bgmode": bgmode,
            "contrast": contrast, "inset": inset, "dark_ratio": dark_ratio}


def quote(path: Path) -> str:
    return f'"{path}"'


# ---------------------------------------------------------------- 硬件扫描
MODEL_RANK = {"tiny": 0, "small": 1, "base_plus": 2, "large": 3}
MODEL_PEAK_MB = {"tiny": 900, "small": 1100, "base_plus": 1600, "large": 3200}
CKPT_NAMES = {
    "tiny": "sam2_hiera_tiny.pt",
    "small": "sam2_hiera_small.pt",
    "base_plus": "sam2_hiera_base_plus.pt",
    "large": "sam2_hiera_large.pt",
}
# 本机 CPU 实测：tiny@192 / pps8 ≈ 2.6 秒/关键帧
BASE_KEYFRAME_SEC = 2.6


def _cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:
            pass
    return platform.processor() or "未知 CPU"


def _memory_gb() -> tuple[float, float]:
    """返回 (总内存, 可用内存)，单位 GB。"""
    if os.name != "nt":
        return 0.0, 0.0

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys / 1e9, status.ullAvailPhys / 1e9
    except Exception:
        return 0.0, 0.0


def _gpu_info() -> dict:
    """通过 nvidia-smi 只读查询 N 卡；没有 nvidia-smi 视为无独显。"""
    info = {"name": "", "vram_total": 0.0, "vram_free": 0.0, "driver": "",
            "util": 0, "vram_used": 0.0}
    exe = shutil.which("nvidia-smi")
    if not exe:
        return info
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free,driver_version,"
                   "utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        line = (result.stdout or "").strip().splitlines()
        if not line:
            return info
        parts = [p.strip() for p in line[0].split(",")]
        if len(parts) >= 6:
            info["name"] = parts[0]
            info["vram_total"] = float(parts[1]) / 1024
            info["vram_free"] = float(parts[2]) / 1024
            info["driver"] = parts[3]
            info["util"] = int(float(parts[4]))
            info["vram_used"] = float(parts[5]) / 1024
    except Exception:
        pass
    return info


def _runtime_probe(py: str | None) -> dict:
    """在目标解释器里查 torch / CUDA / sam2，避免本进程导入 torch。"""
    result = {"python": py, "torch": "", "cuda": False, "sam2": False}
    if not py:
        return result
    code = (
        "import importlib.util as u\n"
        "try:\n"
        "    import torch\n"
        "    print(torch.__version__)\n"
        "    print('1' if torch.cuda.is_available() else '0')\n"
        "except Exception:\n"
        "    print('')\n"
        "    print('0')\n"
        "print('1' if u.find_spec('sam2') else '0')\n"
    )
    try:
        out = subprocess.run(
            [py, "-c", code], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120,
        )
        lines = (out.stdout or "").strip().splitlines()
        if len(lines) >= 3:
            result["torch"] = lines[-3].strip()
            result["cuda"] = lines[-2].strip() == "1"
            result["sam2"] = lines[-1].strip() == "1"
    except Exception:
        pass
    return result


def _host_python() -> str | None:
    """实际执行转换的 Python 解释器。"""
    if not FROZEN:
        return sys.executable
    try:
        import exe_launcher

        infos = exe_launcher.find_interpreters()
        if infos:
            return max(infos, key=exe_launcher.score)["exe"]
    except Exception:
        pass
    return shutil.which("python") or shutil.which("python3")


def scan_hardware() -> dict:
    """只读扫描硬件与运行环境；任何一步失败都降级为默认值，不抛异常。"""
    hw = {
        "cpu_name": _cpu_name(),
        "threads": os.cpu_count() or 0,
        "ram_total": 0.0,
        "ram_avail": 0.0,
        "gpu": _gpu_info(),
        "python": "",
        "torch": "",
        "cuda": False,
        "sam2": False,
        "disk_free": 0.0,
        "weights": set(),
    }
    hw["ram_total"], hw["ram_avail"] = _memory_gb()
    try:
        hw["disk_free"] = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    except Exception:
        pass
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    for name in CKPT_NAMES.values():
        if (ckpt_dir / name).is_file():
            hw["weights"].add(name)
    probe = _runtime_probe(_host_python())
    hw.update(probe)
    # 有 N 卡但 torch 是 CPU 版时不能硬上 CUDA
    hw["cuda"] = bool(probe["cuda"] and hw["gpu"]["name"])
    return hw


def _frame_bytes(info: dict, max_dim: int) -> float:
    """单帧在内存中的近似占用（灰阶 + 掩码 + 前景图 + 字符网格）。"""
    w, h = info["width"] or 16, info["height"] or 9
    scale = min(max_dim / max(w, h), 1.0)
    out_w, out_h = max(1, int(w * scale)), max(1, int(h * scale))
    return out_w * out_h * 3.1 + 120 * 45


def estimate(cfg: dict, hw: dict, info: dict) -> dict:
    """估算用时与内存峰值；结果偏保守，用区间展示。"""
    duration = info["duration"] or 0.0
    sample_frames = max(1, int(duration * cfg["fps"]))
    kf = cfg["keyframe_interval"] or 1
    keyframes = max(1, math.ceil(sample_frames / kf))

    model_f = {"tiny": 1.0, "small": 1.08, "base_plus": 1.15, "large": 1.30}[cfg["model"]]
    dim_f = (cfg["max_dim"] / 192) ** 0.3
    pps_f = (cfg["points_per_side"] / 8) ** 0.5
    dev_f = 0.12 if cfg["device"] == "cuda" else 1.0
    threads = hw["threads"] or 4
    core_f = max(0.75, min(1.6, 8 / threads)) if cfg["device"] == "cpu" else 1.0

    sam = keyframes * BASE_KEYFRAME_SEC * model_f * dim_f * pps_f * dev_f * core_f
    decode = sample_frames * 0.012 + duration * 0.02
    pdf = sample_frames * 0.03 + 8
    load = 8 if cfg["device"] == "cuda" else 16
    total = sam + decode + pdf + load

    frame_mb = _frame_bytes(info, cfg["max_dim"]) * sample_frames / 1e6
    peak_gb = (frame_mb + MODEL_PEAK_MB[cfg["model"]] + 300) / 1024
    ratio = peak_gb / hw["ram_total"] if hw["ram_total"] else 0.5
    return {
        "seconds": total,
        "eta_low": total * 0.78,
        "eta_high": total * 1.28,
        "keyframes": keyframes,
        "sample_frames": sample_frames,
        "peak_gb": peak_gb,
        "mem_ratio": ratio,
    }


def _risk(ratio: float, vram_ratio: float = 0.0) -> tuple[str, int, int]:
    """把内存占用比例换算成死机概率区间（保守估计）。"""
    worst = max(ratio, vram_ratio)
    if worst < 0.35:
        return "很低", 1, 3
    if worst < 0.50:
        return "低", 2, 5
    if worst < 0.65:
        return "中等", 5, 12
    if worst < 0.80:
        return "偏高", 12, 25
    if worst < 0.95:
        return "高", 25, 45
    return "很高", 45, 70


def format_seconds(seconds: float) -> str:
    seconds = max(seconds, 1)
    if seconds < 90:
        return f"{int(round(seconds))} 秒"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} 分钟"
    hours = int(minutes // 60)
    rest = int(minutes % 60)
    return f"{hours} 小时 {rest:02d} 分"


def _quality(cfg: dict) -> float:
    return (MODEL_RANK[cfg["model"]] * 100 + cfg["max_dim"] / 192 * 50
            + cfg["fps"] - cfg["keyframe_interval"] * 0.5)


def build_tiers(hw: dict, info: dict) -> list[dict]:
    """按硬件生成可用档位；连低档都达不到时返回空列表。"""
    if hw["ram_total"] < 2.0 or hw["disk_free"] < 1.0:
        return []

    # 低/中档保持在 CPU：稳定、不受 GPU 被游戏或渲染占用影响
    low = dict(key="low", name="低档", desc="最快、占用最低",
               model="tiny", max_dim=128, keyframe_interval=16,
               points_per_side=6, fps=10, device="cpu")
    mid = dict(key="mid", name="中档", desc="上次 BadApple 成品画质（推荐）",
               model="tiny", max_dim=192, keyframe_interval=10,
               points_per_side=8, fps=12, device="cpu")

    if hw["cuda"]:
        ladder = [
            dict(model="large", max_dim=512, keyframe_interval=4,
                 points_per_side=8, fps=15, device="cuda",
                 desc="硬件理论最高配置（已留 20% 余量）"),
            dict(model="base_plus", max_dim=448, keyframe_interval=5,
                 points_per_side=8, fps=14, device="cuda",
                 desc="显存偏紧，自动降一档"),
            dict(model="base_plus", max_dim=384, keyframe_interval=6,
                 points_per_side=8, fps=13, device="cuda",
                 desc="显存偏紧，再降一档"),
        ]
        if hw["gpu"]["vram_total"] and hw["gpu"]["vram_total"] < 5:
            ladder = ladder[1:]
    else:
        ladder = [
            dict(model="base_plus", max_dim=384, keyframe_interval=6,
                 points_per_side=8, fps=14, device="cpu",
                 desc="无独显，CPU 最高配置（已留 20% 余量）"),
            dict(model="small", max_dim=288, keyframe_interval=8,
                 points_per_side=8, fps=13, device="cpu",
                 desc="内存不足，自动降一档"),
            dict(model="tiny", max_dim=160, keyframe_interval=12,
                 points_per_side=6, fps=10, device="cpu",
                 desc="内存紧张，自动降到最稳的配置"),
        ]

    high = ladder[-1]
    for cand in ladder:
        est = estimate(cand, hw, info)
        vram_ratio = 0.0
        if cand["device"] == "cuda" and hw["gpu"]["vram_total"]:
            vram_ratio = (MODEL_PEAK_MB[cand["model"]] * 1.1 / 1024) / hw["gpu"]["vram_total"]
        if est["mem_ratio"] <= 0.82 and vram_ratio <= 0.85:
            high = cand
            break
    high["key"] = "high"
    high["name"] = "高档"

    tiers = [low]
    if _quality(high) > _quality(mid):
        tiers += [mid, high]
    else:
        high["desc"] = "本机最高可稳定运行的配置（低于上次成品画质）"
        tiers += [high]

    for tier in tiers:
        est = estimate(tier, hw, info)
        vram_ratio = 0.0
        if tier["device"] == "cuda" and hw["gpu"]["vram_total"]:
            vram_ratio = (MODEL_PEAK_MB[tier["model"]] * 1.1 / 1024) / hw["gpu"]["vram_total"]
        label, lo, hi = _risk(est["mem_ratio"], vram_ratio)
        tier.update(est, risk_label=label, risk_low=lo, risk_high=hi,
                    missing_ckpt=CKPT_NAMES[tier["model"]] not in hw["weights"])
        if tier["device"] == "cuda" and hw["gpu"]["util"] >= 50:
            # GPU 被游戏/渲染占用时实测会慢一个数量级，这里按 8 倍粗估
            tier["eta_low"] *= 8
            tier["eta_high"] *= 8
            tier["contended"] = True
    return tiers


def print_hardware(hw: dict) -> None:
    print("\n=== 硬件扫描 ===")
    print(f"  CPU      : {hw['cpu_name']}（{hw['threads']} 线程）")
    if hw["ram_total"]:
        print(f"  内存     : {hw['ram_total']:.1f} GB（当前可用 {hw['ram_avail']:.1f} GB）")
    gpu = hw["gpu"]
    if gpu["name"]:
        busy = f"，已用 {gpu['vram_used']:.1f} GB，负载 {gpu['util']}%" if gpu["util"] else ""
        print(f"  显卡     : {gpu['name']}，显存 {gpu['vram_total']:.1f} GB{busy}")
    else:
        print("  显卡     : 未检测到 NVIDIA 独显")
    if hw["torch"]:
        print(f"  PyTorch  : {hw['torch']}，CUDA {'可用' if hw['cuda'] else '不可用'}")
    else:
        print("  PyTorch  : 当前解释器未检测到（转换时会自动寻找可用的 Python）")
    if hw["weights"]:
        names = "、".join(sorted(hw["weights"]))
        print(f"  已备权重 : {names}")
    else:
        print("  已备权重 : 无（首次转换会下载模型）")
    print(f"  磁盘剩余 : {hw['disk_free']:.1f} GB")
    print("  扫描说明 : 只读查询系统信息与 nvidia-smi，不改驱动、不超频、不改参数；"
          "首次扫描可能慢 1–2 秒，杀毒软件可能弹窗询问，允许即可。")
    if gpu["name"] and not hw["cuda"]:
        print("  [注意] 检测到 N 卡，但 torch 不是 CUDA 版，高档只能按 CPU 运行；"
              "需要 GPU 加速请装 CUDA 版 torch。")
    if hw["cuda"] and gpu["util"] >= 50:
        print(f"  [注意] GPU 当前负载 {gpu['util']}%（可能有游戏/渲染在跑），"
              "高档会明显变慢，建议等 GPU 空闲再转。")
    if hw["threads"] and hw["threads"] < 4:
        print("  [注意] CPU 线程较少，预计用时会比同档位标称值更长。")


def print_tiers(tiers: list[dict], hw: dict) -> None:
    print("\n=== 画质档位（网格固定 120×45，档位只影响分割精度与帧率） ===")
    for i, tier in enumerate(tiers, 1):
        eta = f"{format_seconds(tier['eta_low'])} – {format_seconds(tier['eta_high'])}"
        print(f"  {i}. {tier['name']}  {tier['desc']}")
        print(f"     SAM2 {tier['model']} · 分割 {tier['max_dim']}px · "
              f"关键帧间隔 {tier['keyframe_interval']} · {tier['fps']}fps · {tier['device']}")
        mem = ""
        if hw["ram_total"]:
            mem = f" · 峰值内存约 {tier['peak_gb']:.1f}/{hw['ram_total']:.1f} GB"
        print(f"     预计用时 {eta} · 死机概率约 {tier['risk_low']}%–{tier['risk_high']}%"
              f"（{tier['risk_label']}）{mem}")
        if tier["missing_ckpt"]:
            print("     [提示] 该档模型权重未下载，首次运行会先下载，用时会更长。")
        if tier.get("contended"):
            print("     [提示] GPU 当前被其它程序占用，预计用时已按 8 倍放大粗估。")
    print("  死机概率按预计内存/显存占用比例估算，属于保守估计，不是实测值。")


def choose_tier(tiers: list[dict]) -> dict | None:
    default = 2 if len(tiers) >= 2 and tiers[0]["key"] == "low" and any(
        t["key"] == "mid" for t in tiers) else 1
    while True:
        raw = input(f"\n请选择档位 [默认 {default}]: ").strip()
        if not raw:
            index = default
        elif raw.isdigit():
            index = int(raw)
        else:
            print("  请输入编号。")
            continue
        if 1 <= index <= len(tiers):
            return tiers[index - 1]
        print(f"  编号需要在 1–{len(tiers)} 之间。")


def launch_conversion(video: Path, audio: Path | None, tier: dict, hw: dict) -> int:
    """按选定档位调用 convert.py 开始转换。"""
    script = PROJECT_ROOT / "convert.py"
    if not script.is_file():
        print(f"[失败] 找不到转换脚本：{script}")
        return 1
    py = hw.get("python")
    if not py:
        print("[失败] 未找到可用的 Python（需要 3.9 及以上），无法运行 SAM2 分割。")
        return 1

    args = [
        str(video), "--fps", str(tier["fps"]), "--model", tier["model"],
        "--device", tier["device"], "--max-dim", str(tier["max_dim"]),
        "--keyframe-interval", str(tier["keyframe_interval"]),
        "--points-per-side", str(tier["points_per_side"]), "--no-analyze",
    ]
    if audio and Path(audio).resolve() != video.resolve():
        args += ["--audio", str(audio)]

    print(f"\n开始转换（{tier['name']}），中途可按 Ctrl+C 中断 ...\n")
    sys.stdout.flush()
    try:
        return subprocess.call([py, str(script), *args], cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n已中断转换。")
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(
        description="分析视频元数据、画面亮度、黑边和音频，输出转换参数建议",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python analyze.py                                 交互选择低/中/高档并直接转换
  python analyze.py --tier high                     直接按高档转换
  python analyze.py --no-convert                    只看分析结果，不转换
  python analyze.py dist/input/待分离/demo.mp4
  python analyze.py dist/input/待分离/demo.mp4 --seconds 60
  python analyze.py --data output/demo_data.npz
""",
    )
    parser.add_argument("video", nargs="?", help="视频路径；不填则自动使用 待分离/ 中唯一的视频")
    parser.add_argument("--data", help="要额外分析的 SAM2 分割数据 .npz")
    parser.add_argument("--seconds", type=float, default=0, help="画面采样时长，0=全片")
    parser.add_argument("--audio-seconds", type=float, default=0, help="音量检测时长，0=全片")
    parser.add_argument("--tier", choices=["low", "mid", "high"],
                        help="直接指定档位并开始转换，跳过选择提示")
    parser.add_argument("--no-convert", action="store_true",
                        help="只分析并列出档位，不启动转换")
    args = parser.parse_args()

    try:
        video = pick_video(args.video)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"=== 视频分析：{video.name} ===")
    info = probe(video)
    print(f"文件      : {video}  ({human_size(video.stat().st_size)})")
    print(f"画面      : {info['codec']}  {info['width']}x{info['height']}  "
          f"{info['fps']:.2f} fps  共 {info['frames']} 帧")
    print(f"时长      : {info['duration']:.1f} s")

    hw = scan_hardware()
    print_hardware(hw)

    print("\n正在扫描黑边（会解码整段画面）...")
    box = None
    try:
        box = detect_content_box(str(video))
        bw, bh = box[2] - box[0], box[3] - box[1]
        coverage = bw * bh / max(info["width"] * info["height"], 1)
        print(f"有效画面  : {box}  裁切后 {bw}x{bh}，占全幅 {coverage * 100:.1f}%")
        if coverage < 0.5:
            print("  [注意] 黑边占比很大，转换时会自动裁切；必要时加大 --inset")
    except Exception as exc:
        print(f"  [警告] 黑边检测失败：{exc}")

    print("\n正在采样画面亮度（每秒 1 帧）...")
    try:
        bright = sample_brightness(video, info["fps"], args.seconds)
        print(f"平均亮度  : {bright['mean']:.1f}   p10 {bright['p10']:.1f}  "
              f"p50 {bright['p50']:.1f}  p90 {bright['p90']:.1f}")
        print(f"明暗分布  : 暗帧(<60) {bright['dark']}/{bright['count']} "
              f"({bright['dark'] / bright['count'] * 100:.1f}%)，"
              f"亮帧(>180) {bright['bright']}/{bright['count']} "
              f"({bright['bright'] / bright['count'] * 100:.1f}%)")
    except Exception as exc:
        print(f"  [警告] 亮度采样失败：{exc}")
        bright = {"mean": 128, "p10": 0, "p50": 128, "p90": 255,
                  "dark": 0, "bright": 0, "count": 1}

    recommendation = recommend(info, bright, box)

    print("\n=== 音频 ===")
    audio_path = None
    audio_note = ""
    if info["has_audio"]:
        audio_path = video
        audio_note = f"视频内音轨 {info['audio_codec']} / {info['audio_rate']} Hz / {info['audio_channels']}"
    else:
        sibling = AUDIO_DIR / f"{video.stem}.mp3"
        if sibling.is_file():
            audio_path = sibling
            audio_note = f"视频无音轨，使用同名 MP3：{sibling.name}"
        else:
            print("  [警告] 视频没有音轨，dist/input/纯音频 中也没有同名 MP3")
            print("         可用 separate_media.py 分离，或单独导出音频后用 --audio 指定")
    if audio_path:
        volume = measure_volume(audio_path, args.audio_seconds)
        if volume is None:
            print(f"  [警告] {audio_note}，但读不到音频数据")
        else:
            mean_db, peak_db = volume
            print(f"  {audio_note}")
            print(f"  音量：mean {mean_db:.1f} dB / max {peak_db:.1f} dB")
            if mean_db < -50:
                print("  [失败] 音轨近似静音。改用在剪辑软件里单独导出的 MP3：")
                print(f'         python convert.py {quote(video)} --audio "你的音频.mp3"')
            else:
                print("  结论：音频正常")

    print("\n=== 推荐参数 ===")
    print(f"  --fps {recommendation['fps']} --theme {recommendation['theme']} "
          f"--bgmode {recommendation['bgmode']} --contrast {recommendation['contrast']} "
          f"--inset {recommendation['inset']}")
    if recommendation["bgmode"] == "stable":
        print("  说明：画面以暗场为主，恒定黑底最稳定；若片中大面积白底画面较多可改 --bgmode switch")
    else:
        print("  说明：明暗交替明显，建议跟随画面翻转底色；若翻转太频繁可加大 --smooth")

    print("\n=== 生成命令 ===")
    print(f"python convert.py {quote(video)} --test")
    print(f"python convert.py {quote(video)}")
    if args.data or (OUTPUT_DIR / f"{video.stem}_data.npz").is_file():
        data_path = Path(args.data) if args.data else OUTPUT_DIR / f"{video.stem}_data.npz"
        print(f"\n=== 分割数据：{data_path} ===")
        result = analyze_npz(data_path)
        if result:
            print(f"  帧数 {result['frames']}  分辨率 {result['width']}x{result['height']}  "
                  f"fps {result['fps'] or '未知'}")
            print(f"  前景占比 mean {result['ratio_mean'] * 100:.1f}%  "
                  f"p10 {result['ratio_p10'] * 100:.1f}%  p90 {result['ratio_p90'] * 100:.1f}%")
            if "fg_gray" in result:
                print(f"  前景平均灰度 {result['fg_gray']:.1f}")
            if result["full"]:
                print(f"  [警告] {result['full']} 帧前景占比超过 75%，SAM2 可能把背景当成角色，"
                      f"建议降低 --max-dim 或调整 --points-per-side 后重新分割")
            if result["empty"]:
                print(f"  [警告] {result['empty']} 帧前景为空，检查片源是否有纯色/空镜段落")
            if not result["full"] and not result["empty"]:
                print("  结论：分割数据正常，可以直接生成 PDF")

    tiers = build_tiers(hw, info)
    if not tiers:
        print("\n[配置过低] 内存或磁盘不足（低档至少需要 2 GB 内存、1 GB 空闲磁盘），")
        print("           已停止档位选择，请换一台配置更高的电脑再转换。")
        return 2

    print_tiers(tiers, hw)
    chosen = None
    if args.no_convert:
        print("\n（--no-convert：只输出分析结果，未启动转换）")
    elif args.tier:
        chosen = next((t for t in tiers if t["key"] == args.tier), None)
        if chosen is None:
            available = "、".join(t["name"] for t in tiers)
            print(f"\n当前硬件没有 {args.tier} 档，可用：{available}")
            return 1
    elif sys.stdin.isatty():
        chosen = choose_tier(tiers)
    else:
        print("\n（当前不是交互终端，未选择档位；用 --tier low|mid|high 可直接转换）")

    if chosen:
        return launch_conversion(video, audio_path, chosen, hw)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        exit_code = main()
    except Exception as exc:
        print(f"分析失败：{exc}", file=sys.stderr)
        exit_code = 1
    if getattr(sys, "frozen", False) and len(sys.argv) == 1:
        try:
            input("\n按回车键退出...")
        except EOFError:
            pass
    raise SystemExit(exit_code)
