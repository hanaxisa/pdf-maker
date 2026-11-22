# -*- coding: utf-8 -*-
"""InitEnv.exe / OneClick.exe 的启动器：定位项目与可用的 Python，再运行对应脚本。

InitEnv.exe   -> init.py      （一键初始化：依赖、模型、目录、编辑器、自检）
OneClick.exe  -> analyze.py   （一键转换：硬件扫描 → 选档 → 自动调用 convert.py 生成 PDF）

这些 exe 需要本机已安装 Python 3.9+；它们会自动挑选依赖最完整、版本最合适的解释器。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROBE_MODULES = ["cv2", "numpy", "reportlab", "pypdf", "imageio_ffmpeg", "torch", "sam2"]
PROBE_CODE = (
    "import sys, importlib.util as u, json;"
    "print(json.dumps({'v': list(sys.version_info[:2]),"
    "'m': {m: bool(u.find_spec(m)) for m in " + repr(PROBE_MODULES) + "}}))"
)


def project_root(exe_dir: Path) -> Path:
    candidates = [exe_dir, exe_dir.parent]
    current = Path.cwd()
    for _ in range(3):
        candidates.append(current)
        current = current.parent
    for base in candidates:
        if (base / "requirements.txt").is_file() and (base / "init.py").is_file():
            return base
    return exe_dir.parent if exe_dir.name.lower() == "dist" else exe_dir


def probe(exe: str) -> dict | None:
    try:
        result = subprocess.run(
            [exe, "-c", PROBE_CODE], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout.strip().splitlines()[-1])
        return {"exe": exe, "v": tuple(data["v"]), "m": data["m"]}
    except Exception:
        return None


def find_interpreters() -> list[dict]:
    found: list[str] = []
    for name in ("python", "python3"):
        path = shutil.which(name)
        if path:
            found.append(path)
    if os.name == "nt":
        try:
            listing = subprocess.run(
                ["py", "-0p"], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=10,
            ).stdout
            found += re.findall(r"([A-Za-z]:\\[^\r\n]*?python\.exe)", listing)
        except Exception:
            pass
        local = os.environ.get("LOCALAPPDATA")
        if local:
            found += [str(p) for p in sorted(Path(local, "Programs", "Python").glob("Python3*/python.exe"))]

    unique: dict[str, str] = {}
    for exe in found:
        try:
            resolved = str(Path(exe).resolve())
        except Exception:
            continue
        unique.setdefault(resolved.lower(), resolved)

    infos = []
    for exe in unique.values():
        info = probe(exe)
        if info and info["v"] >= (3, 9):
            infos.append(info)
    return infos


def score(info: dict) -> tuple[int, int, int, int]:
    installed = sum(1 for ok in info["m"].values() if ok)
    preferred = 1 if info["v"] in {(3, 10), (3, 11), (3, 12)} else 0
    return installed, preferred, info["v"][1], info["v"][0]


def target_script(exe_name: str) -> str:
    lowered = exe_name.lower()
    if "init" in lowered:
        return "init.py"
    if "self" in lowered:
        return "selfcheck.py"
    # OneClick 及其他默认入口统一走 analyze.py（硬件扫描 + 选档 + 转换）
    return "analyze.py"


def main() -> int:
    exe = Path(sys.executable).resolve()
    root = project_root(exe.parent)
    script_name = target_script(exe.stem)
    script = root / script_name
    if not script.is_file():
        print(f"找不到项目脚本：{script}")
        print("请把 exe 放在项目目录或其 dist/ 文件夹中。")
        return 2

    infos = find_interpreters()
    if not infos:
        print("未找到可用的 Python（需要 3.9 及以上）。")
        print("请先安装 Python：https://www.python.org/downloads/")
        return 2

    best = max(infos, key=score)
    print(f"项目目录：{root}")
    print(f"使用解释器：{best['exe']}")
    sys.stdout.flush()
    sys.stderr.flush()
    code = subprocess.call([best["exe"], str(script), *sys.argv[1:]], cwd=str(root))
    if code == 0 and script_name in ("analyze.py", "convert.py") and len(sys.argv) == 1:
        output = root / "output"
        if output.is_dir():
            try:
                os.startfile(str(output))
            except OSError:
                pass
    return code


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    code = main()
    if getattr(sys, "frozen", False) and len(sys.argv) == 1:
        try:
            input("\n按回车键退出...")
        except EOFError:
            pass
    raise SystemExit(code)
