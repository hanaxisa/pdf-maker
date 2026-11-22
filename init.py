# -*- coding: utf-8 -*-
"""一键初始化：自动选择解释器、安装依赖、下载模型、创建工作目录、配置编辑器并自检。

用法：
    python init.py                       # 全自动初始化
    python init.py --check               # 只检查，不改动环境
    python init.py --model small         # 指定 SAM2 模型（tiny/small/base_plus/large）
    python init.py --mirror aliyun       # 默认使用清华镜像，可换成阿里云或 none
    python init.py --interpreter "C:\\...\\Python312\\python.exe"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DIST_INPUT = ROOT / "dist" / "input"
OUTPUT_DIR = ROOT / "output"
REQUIREMENTS = ROOT / "requirements.txt"

CORE_DEPS = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "reportlab": "reportlab",
    "pypdf": "pypdf",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "PIL": "Pillow",
}
ALL_DEPS = {
    **CORE_DEPS,
    "yt_dlp": "yt-dlp",
    "torch": "torch",
    "sam2": "sam2",
    "huggingface_hub": "huggingface_hub",
    "skimage": "scikit-image",
}
MIRRORS = {
    "tsinghua": "https://pypi.tuna.tsinghua.edu.cn/simple",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple",
    "none": "",
}
MODELS = {
    "tiny": ("sam2_hiera_tiny.pt", "facebook/sam2-hiera-tiny"),
    "small": ("sam2_hiera_small.pt", "facebook/sam2-hiera-small"),
    "base_plus": ("sam2_hiera_base_plus.pt", "facebook/sam2-hiera-base-plus"),
    "large": ("sam2_hiera_large.pt", "facebook/sam2-hiera-large"),
}
PROBE_MODULES = ["cv2", "numpy", "reportlab", "pypdf", "imageio_ffmpeg", "torch", "sam2", "yt_dlp"]
PROBE_CODE = (
    "import sys, importlib.util as u, json;"
    "print(json.dumps({'v': list(sys.version_info[:2]),"
    "'m': {m: bool(u.find_spec(m)) for m in " + repr(PROBE_MODULES) + "}}))"
)


def run(cmd: list[str], **kwargs) -> int:
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    sys.stdout.flush()
    sys.stderr.flush()
    return subprocess.run(cmd, **kwargs).returncode


def probe_interpreter(exe: str) -> dict | None:
    try:
        result = subprocess.run(
            [exe, "-c", PROBE_CODE], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=25,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    try:
        exe = str(Path(exe).resolve())
    except Exception:
        pass
    return {"exe": exe, "v": tuple(data["v"]), "m": data["m"]}


def discover_interpreters() -> list[dict]:
    candidates: list[str] = [sys.executable]
    if os.name == "nt":
        for name in ("python", "python3"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
        local = os.environ.get("LOCALAPPDATA")
        if local:
            for path in sorted(Path(local, "Programs", "Python").glob("Python3*/python.exe")):
                candidates.append(str(path))
        try:
            listing = subprocess.run(["py", "-0p"], capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=10).stdout
            candidates += re.findall(r"([A-Za-z]:\\[^\r\n]*?python\.exe)", listing)
        except Exception:
            pass

    unique: dict[str, str] = {}
    for exe in candidates:
        try:
            key = str(Path(exe).resolve()).lower()
        except Exception:
            continue
        unique.setdefault(key, exe)

    found = []
    for exe in unique.values():
        info = probe_interpreter(exe)
        if info:
            found.append(info)
    return found


def score(info: dict) -> tuple[int, int, int]:
    installed = sum(1 for ok in info["m"].values() if ok)
    preferred = 1 if info["v"] in {(3, 10), (3, 11), (3, 12)} else 0
    return installed * 10 + preferred * 3, info["v"][1], info["v"][0]


def choose_interpreter(forced: str | None) -> dict:
    current = probe_interpreter(sys.executable) or {"exe": sys.executable, "v": (0, 0), "m": {}}
    if forced:
        info = probe_interpreter(forced)
        if not info:
            raise SystemExit(f"无法运行指定的解释器：{forced}")
        return info

    candidates = discover_interpreters()
    if not candidates:
        return current
    # 当前解释器在得分相同时优先，避免无意义重启
    best = max(candidates, key=lambda item: (score(item), item["exe"] == sys.executable))
    if best["exe"] != current["exe"] and score(best) > score(current):
        return best
    return current


def relaunch(exe: str, argv: list[str]) -> int:
    forwarded: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--interpreter":
            skip_next = True
            continue
        if arg.startswith("--interpreter="):
            continue
        if arg == "--no-switch":
            continue
        forwarded.append(arg)
    forwarded.append("--no-switch")
    print(f"切换到更合适的解释器：{exe}")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        return subprocess.call([exe, str(Path(__file__).resolve()), *forwarded])
    except Exception as exc:
        print(f"切换失败（{exc}），继续使用当前解释器：{sys.executable}")
        return -1


def ensure_dirs() -> None:
    for path in (CHECKPOINT_DIR, DIST_INPUT / "待分离", DIST_INPUT / "静视频",
                 DIST_INPUT / "纯音频", OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)
    print(f"工作目录已就绪：{DIST_INPUT}")


def missing_deps() -> list[str]:
    return [module for module in ALL_DEPS if importlib.util.find_spec(module) is None]


def install_deps(mirror: str, check_only: bool) -> bool:
    missing = missing_deps()
    core_missing = [module for module in CORE_DEPS if module in missing]
    if not missing:
        print("依赖检查：全部已安装")
        return True

    names = "、".join(ALL_DEPS[module] for module in missing)
    if check_only:
        print(f"依赖检查：缺少 {names}")
        return False

    print(f"安装依赖：{names}")
    base = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    url = MIRRORS.get(mirror, "")
    code = run(base + (["-i", url] if url else []))
    if code != 0 and url:
        print("镜像安装失败，改用默认 PyPI 重试 ...")
        code = run(base)
    if code != 0:
        print("依赖安装失败。请检查网络后重试，或手动执行：")
        print(f'  "{sys.executable}" -m pip install -r "{REQUIREMENTS}"')
        return False

    remaining = missing_deps()
    if remaining:
        print(f"以下依赖仍未就绪：{'、'.join(ALL_DEPS[m] for m in remaining)}")
        return False
    print("依赖安装完成")
    return all(importlib.util.find_spec(module) is not None for module in CORE_DEPS)


def download_model(model: str, check_only: bool) -> bool:
    filename, repo = MODELS[model]
    target = CHECKPOINT_DIR / filename
    if target.is_file() and target.stat().st_size > 10 * 1024 * 1024:
        print(f"模型检查：{filename} 已存在（{target.stat().st_size / 1024 / 1024:.0f} MB）")
        return True
    if check_only:
        print(f"模型检查：缺少 {filename}")
        return False

    print(f"下载模型：{repo} / {filename}（约 150-900 MB，请保持网络连接）")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(CHECKPOINT_DIR))
        size = Path(path).stat().st_size / 1024 / 1024
        print(f"模型已就绪：{path}（{size:.0f} MB）")
        return size > 10
    except Exception as exc:
        print(f"模型下载失败：{exc}")
        print("可手动下载后放到 checkpoints/ 根目录（不要嵌套文件夹）：")
        print(f"  https://hf-mirror.com/{repo}/tree/main")
        return False


def configure_editor(exe: str, check_only: bool) -> None:
    if check_only:
        print(f"编辑器解释器建议：{exe}")
        return
    settings_path = WORKSPACE / ".vscode" / "settings.json"
    data: dict = {}
    if settings_path.is_file():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            print("跳过编辑器配置：现有 .vscode/settings.json 含注释或格式特殊，无法安全写入")
            print(f'请手动设置 "python.defaultInterpreterPath": "{exe}"')
            return
    data["python.defaultInterpreterPath"] = exe
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n",
                                 encoding="utf-8")
        print(f"编辑器解释器已配置：{settings_path}")
    except Exception as exc:
        print(f"无法写入编辑器配置（{exc}），请手动选择解释器：{exe}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BadApplePDF 一键初始化：解释器、依赖、模型、目录、编辑器与自检",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python init.py
  python init.py --check
  python init.py --model small --mirror aliyun
  python init.py --interpreter "C:\\Users\\me\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"
""",
    )
    parser.add_argument("--check", action="store_true", help="只检查，不安装或下载")
    parser.add_argument("--model", choices=list(MODELS), default="tiny", help="要准备的 SAM2 模型")
    parser.add_argument("--mirror", choices=list(MIRRORS), default="tsinghua", help="pip 镜像源")
    parser.add_argument("--interpreter", help="强制使用指定 Python 解释器")
    parser.add_argument("--no-switch", action="store_true", help="不自动切换解释器")
    parser.add_argument("--no-config", action="store_true", help="不写入编辑器解释器配置")
    parser.add_argument("--skip-model", action="store_true", help="跳过模型下载")
    parser.add_argument("--skip-selfcheck", action="store_true", help="结束后不运行自检")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print("BadApplePDF 一键初始化")
    print(f"项目目录：{ROOT}")
    print(f"当前解释器：{sys.executable}")

    if not args.no_switch:
        chosen = choose_interpreter(args.interpreter)
        if os.path.normcase(str(Path(chosen["exe"]))) != os.path.normcase(str(Path(sys.executable))):
            code = relaunch(chosen["exe"], sys.argv[1:])
            if code >= 0:
                return code

    version = sys.version_info
    print(f"使用解释器：{sys.executable}（Python {version.major}.{version.minor}.{version.micro}）")
    if version < (3, 10):
        print("Python 版本过低，请安装 3.10 及以上版本后重新运行。")
        return 1

    ensure_dirs()
    deps_ok = install_deps(args.mirror, args.check)
    model_ok = True if args.skip_model else download_model(args.model, args.check)
    if not args.no_config:
        configure_editor(sys.executable, args.check)

    if args.check:
        print(f"\n检查结果：依赖{'就绪' if deps_ok else '缺失'}；"
              f"模型{'就绪' if model_ok else '缺失'}")

    if not args.skip_selfcheck and not args.check:
        print("\n运行自检 ...")
        selfcheck = ROOT / "selfcheck.py"
        if selfcheck.is_file():
            code = run([sys.executable, str(selfcheck)])
            if code != 0:
                print("自检发现需要处理的问题，请按上面的提示修复。")
                return 1

    if deps_ok and model_ok:
        print("\n初始化完成。下一步：")
        print("  1. 把视频放进 dist/input/待分离/")
        print("  2. python analyze.py          查看素材和推荐参数")
        print("  3. python convert.py          生成 PDF 播放器")
        return 0
    print("\n初始化未完全成功，请根据上面的提示处理后重新运行 python init.py")
    return 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
