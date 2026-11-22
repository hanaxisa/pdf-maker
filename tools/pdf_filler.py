"""PDF 铺满工具：读取 output 里的分割数据，按源画面比例自动选择填充方式重新生成 PDF。

- 原 PDF 永远不会被改动，新文件写成 `<原文件名>_filled.pdf`
- 「反悔」= 删除新生成的文件（原片从未被动过）
- 自动策略：画面接近 4:3 用 letterbox；偏宽/偏高用 cover（裁边，不变形）
"""

import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

FROZEN = getattr(sys, "frozen", False)
HERE = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent.parent
ROOT = HERE.parent if FROZEN and (HERE.parent / "requirements.txt").is_file() else HERE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import ascii
from contour.sam_segmenter import SegmentResult
from pdf.player_pdf import build_ascii_player, build_char_player, list_audio_tracks, tone_profile

OUTPUT = ROOT / "output"
INPUT = (HERE / "input") if FROZEN else (ROOT / "dist" / "input")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".flv", ".webm", ".avi", ".m4v")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac")

BG = "#101010"
CARD = "#1c1c1c"
BORDER = "#2a2a2a"
TEXT = "#e6e6e6"
SUB = "#8b8b8b"
ACCENT = "#3ddc84"


def list_pdfs():
    if not OUTPUT.is_dir():
        return []
    return sorted((p for p in OUTPUT.glob("*.pdf") if p.is_file()),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def find_npz(pdf: Path):
    """PDF 文件名反查对应的 *_data.npz。"""
    stem = pdf.stem
    candidates = [stem]
    candidates.append(re.sub(r"_(player|fill|filled)(_v\d+)?$", "", stem))
    candidates.append(re.sub(r"_v\d+$", "", stem))
    for name in candidates:
        for suffix in ("_data", ""):
            path = OUTPUT / f"{name}{suffix}.npz"
            if path.is_file():
                return path
    return None


def find_video(base: str):
    for folder in (INPUT / "待分离", INPUT / "静视频", OUTPUT):
        for ext in VIDEO_EXTS:
            candidate = folder / f"{base}{ext}"
            if candidate.is_file():
                return candidate
    return None


def find_audio(base: str, video: Path | None):
    for ext in AUDIO_EXTS:
        candidate = INPUT / "纯音频" / f"{base}{ext}"
        if candidate.is_file():
            return candidate
    return video


def content_box(gray, thresh: int = 25):
    """去掉四周黑边后的有效画面范围。"""
    height, width = gray.shape[:2]
    cols = np.where(gray.max(axis=0) > thresh)[0]
    rows = np.where(gray.max(axis=1) > thresh)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, width, height
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def probe_ratio(video: Path):
    """取视频前中后若干帧的有效画面比例（中位数）。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    ratios = []
    for pos in (0.1, 0.3, 0.5, 0.7, 0.9):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pos))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = content_box(gray)
        if x1 > x0 and y1 > y0:
            ratios.append((x1 - x0) / (y1 - y0))
    cap.release()
    if not ratios:
        raise RuntimeError("无法识别画面内容")
    return float(np.median(ratios))


def read_font_size(pdf: Path, default: float = 10.0) -> float:
    """从 PDF 行文本域（/T (r0)，旧版是 /T (screen)）的 /DA 里读字号。"""
    try:
        text = pdf.read_bytes().decode("latin-1")
        field = re.search(r"/T \((r0|screen)\)(.*?)/AP", text, re.S)
        scope = field.group(2) if field else text
        match = re.search(r"/DA\s*\(/Cour\s+([\d.]+)\s+Tf", scope)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return default


def measure_rendered(npz: Path):
    """量出当前 PDF 里画面实际占了多少字符格（用于判断是否已经铺满）。"""
    result = SegmentResult.load(str(npz))
    total = len(result.frames)
    if total == 0:
        return 0.0, 0.0
    picks = {int(total * p) for p in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)}
    cols_used = rows_used = 0
    shape = None
    for index in sorted(i for i in picks if i < total):
        grid = result.frames[index].grid
        if grid is None:
            continue
        shape = grid.shape
        ink = np.argwhere(grid > 30)
        if ink.size == 0:
            continue
        rows_used = max(rows_used, int(ink[:, 0].max()) - int(ink[:, 0].min()) + 1)
        cols_used = max(cols_used, int(ink[:, 1].max()) - int(ink[:, 1].min()) + 1)
    rows, cols = shape if shape else (ascii.ROWS, ascii.COLS)
    return cols_used / cols, rows_used / rows


def decide(ratio: float, fill_x: float, fill_y: float):
    """结合画面比例和当前实际占屏比例，选择填充方式，返回 (mode, 说明)。"""
    target = 4 / 3
    if fill_x >= 0.97 and fill_y >= 0.97:
        return "letterbox", "当前画面已经铺满屏幕，无需再处理"
    if abs(ratio / target - 1) <= 0.02:
        # 差不到 2% 时用 cover：等比补边会因为差这几像素而在底部/右侧留一整行空白
        return "cover", f"画面比例 {ratio:.2f}:1，与 4:3 只差 {abs(ratio / target - 1) * 100:.1f}%，直接裁掉后可完全铺满"
    if ratio > target:
        loss = (1 - target / ratio) * 100
        direction = "左右"
    else:
        loss = (1 - ratio / target) * 100
        direction = "上下"
    if loss > 30:
        return "letterbox", (f"画面比例 {ratio:.2f}:1，铺满需要裁掉{direction}约 {loss:.0f}%，"
                             f"裁切过多会切掉主体，建议保留黑边（可在下拉框里强制裁剪或拉伸）")
    return "cover", f"画面比例 {ratio:.2f}:1，裁掉{direction}约 {loss:.1f}% 后铺满，画面不变形"


def rebuild(npz: Path, video: Path, audio: Path | None, mode: str, out_pdf: Path,
            credits, progress, audio_track: int = 0, color_mode: str = "high"):
    """用视频按新的填充方式重采样字符网格，再合成为新的 PDF。"""
    result = SegmentResult.load(str(npz))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    # 按时间戳取帧：30fps 源每 2.5 帧取一帧，取整成 2 或 3 都会让音画逐渐偏移。
    dt = 1.0 / max(result.fps, 1)
    next_t = 0.0
    box = None
    index = 0
    for frame in result.frames:
        gray_full = None
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            t = index / src_fps
            index += 1
            if t + 1e-9 >= next_t:
                gray_full = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                next_t += dt
                break
        if gray_full is None:
            break
        if box is None:
            box = content_box(gray_full)
        x0, y0, x1, y1 = box
        crop = gray_full[y0:y1, x0:x1]
        frame.grid = ascii.fit_grid(crop if crop.size else gray_full, mode=mode)
        # 同步重采颜色网格，旧数据重新铺满后也能获得色度感知
        color_crop = raw[y0:y1, x0:x1]
        frame.color_grid = ascii.fit_grid_color(color_crop if color_crop.size else raw, mode=mode)
        progress(f"重采样 {len(result.frames)} 帧…")
    cap.release()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    progress("合成 PDF…")
    # 高色差：彩色/连续色调片用字符画渲染；低色差与黑底剪影走 ASCII 路径。
    high = color_mode == "high"
    if high and tone_profile(result)["mode"] == "color":
        build_char_player(result, str(video), str(out_pdf), credits=credits, credit_sec=3.0,
                          audio_track=audio_track)
    else:
        build_ascii_player(result, audio, str(out_pdf), credits=credits, credit_sec=3.0,
                           audio_track=audio_track, color_analysis=high)
    return out_pdf


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.pdfs = []
        self.analysis = None
        self.busy = False

        root.title("PDF 铺满工具")
        root.configure(bg=BG)
        root.geometry("1000x640")

        tk.Label(root, text="PDF 铺满工具", bg=BG, fg=TEXT,
                 font=("Microsoft YaHei UI", 14, "bold")).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(root, text="选中一个 PDF → 分析 → 生成铺满版（原文件不会被改动）",
                 bg=BG, fg=SUB, font=("Microsoft YaHei UI", 9)).pack(anchor="w", padx=16)

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=10)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self.listbox = tk.Listbox(left, bg=CARD, fg=TEXT, bd=0, highlightthickness=0,
                                  selectbackground="#1b3d2c", selectforeground=TEXT,
                                  font=("Microsoft YaHei UI", 9), activestyle="none")
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_pick)

        right = tk.Frame(body, bg=BG, width=360)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)

        self.info = tk.Text(right, bg=CARD, fg=TEXT, bd=0, height=13, wrap="word",
                            font=("Microsoft YaHei UI", 9), padx=10, pady=10)
        self.info.pack(fill="x")
        self.info.configure(state="disabled")

        tk.Label(right, text="填充方式（自动 = 按实际渲染像素判断）", bg=BG, fg=SUB,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w", pady=(10, 2))
        self.mode = tk.StringVar(value="自动")
        ttk.Combobox(right, values=["自动", "裁剪铺满", "等比补边", "拉伸铺满"],
                     textvariable=self.mode, state="readonly").pack(fill="x")

        tk.Label(right, text="色差版本", bg=BG, fg=SUB,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w", pady=(10, 2))
        self.color_mode = tk.StringVar(value="高色差")
        ttk.Combobox(right, values=["高色差", "低色差"],
                     textvariable=self.color_mode, state="readonly").pack(fill="x")

        tk.Label(right, text="音轨（多音轨视频可在此选择）", bg=BG, fg=SUB,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w", pady=(10, 2))
        self._audio_tracks = []
        self.audio_combo = ttk.Combobox(right, values=["（未选择视频）"], state="disabled")
        self.audio_combo.pack(fill="x")
        self.audio_combo.current(0)

        tk.Label(right, text="片尾字幕（留空则无片尾，多行用 | 分隔）", bg=BG, fg=SUB,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w", pady=(10, 2))
        self.credits = tk.Entry(right, bg=CARD, fg=TEXT, bd=0, insertbackground=TEXT,
                                font=("Microsoft YaHei UI", 9))
        self.credits.pack(fill="x", ipady=5)

        self.btn_build = tk.Button(right, text="生成铺满版", command=self.on_build,
                                   bg="#2f7d4f", fg="white", bd=0, pady=8,
                                   activebackground="#3a9a61", activeforeground="white",
                                   font=("Microsoft YaHei UI", 9, "bold"), cursor="hand2")
        self.btn_build.pack(fill="x", pady=(12, 6))

        self.btn_undo = tk.Button(right, text="反悔（删除铺满版）", command=self.on_undo,
                                  bg=CARD, fg=TEXT, bd=0, pady=8,
                                  activebackground="#2a2a2a", activeforeground=TEXT,
                                  font=("Microsoft YaHei UI", 9), cursor="hand2")
        self.btn_undo.pack(fill="x")

        tk.Button(right, text="打开 output 文件夹", command=self.open_output, bg=CARD,
                  fg=TEXT, bd=0, pady=6, activebackground="#2a2a2a",
                  activeforeground=TEXT, font=("Microsoft YaHei UI", 9),
                  cursor="hand2").pack(fill="x", pady=(6, 0))

        self.status = tk.Label(root, text="", bg=BG, fg=SUB, anchor="w",
                               font=("Microsoft YaHei UI", 9))
        self.status.pack(fill="x", padx=16, pady=(0, 12))

        self.reload()

    def reload(self):
        self.pdfs = list_pdfs()
        self.listbox.delete(0, "end")
        for pdf in self.pdfs:
            size = pdf.stat().st_size / 1024 / 1024
            npz = find_npz(pdf)
            mark = "" if npz else "  [无分割数据]"
            self.listbox.insert("end", f"{pdf.name}    {size:.1f} MB{mark}")
        self.status.configure(text=f"output 目录：{OUTPUT} · 共 {len(self.pdfs)} 个 PDF")

    def on_pick(self, _event=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        pdf = self.pdfs[selection[0]]
        npz = find_npz(pdf)
        if not npz:
            self._show_info(pdf.name, "找不到对应的 *_data.npz，无法重新生成。")
            self.analysis = None
            return
        base = npz.stem[:-5] if npz.stem.endswith("_data") else npz.stem
        video = find_video(base)
        if not video:
            self._show_info(pdf.name, f"找不到源视频（{base}.mp4）。\n它需要在 input/待分离 或 input/静视频 里。")
            self.analysis = None
            return
        self.analysis = None
        self._current = pdf
        self.btn_build.configure(state="disabled", text="分析中…")
        self._show_info(pdf.name, "正在读取数据并分析画面比例…")
        threading.Thread(target=self._analyze, args=(pdf, npz, video), daemon=True).start()

    def _analyze(self, pdf, npz, video):
        try:
            ratio = probe_ratio(video)
            fill_x, fill_y = measure_rendered(npz)
            mode, note = decide(ratio, fill_x, fill_y)
            error = None
        except Exception as exc:
            ratio = fill_x = fill_y = 0.0
            mode, note, error = "letterbox", "", str(exc)
        self.root.after(0, self._analyzed, pdf, npz, video, ratio, fill_x, fill_y, mode, note, error)

    def _analyzed(self, pdf, npz, video, ratio, fill_x, fill_y, mode, note, error):
        if getattr(self, "_current", None) is not pdf:
            return
        self.btn_build.configure(state="normal", text="生成铺满版")
        if error:
            self._show_info(pdf.name, f"分析失败：{error}")
            return
        out = pdf.with_name(f"{pdf.stem}_filled.pdf")
        font = read_font_size(pdf)
        self.analysis = dict(pdf=pdf, npz=npz, video=video, mode=mode, out=out)
        self._refresh_audio_tracks(video)
        self._show_info(pdf.name,
                        f"源视频：{video.name}\n"
                        f"屏幕字号：{font:g}pt（渲染格子 {0.6 * font:.1f} x {1.2 * font:.1f}）\n"
                        f"画面比例：{ratio:.3f} : 1\n"
                        f"当前占屏：横向 {fill_x * 100:.0f}% · 纵向 {fill_y * 100:.0f}%\n"
                        f"自动策略：{ {'letterbox': '等比补边', 'cover': '裁剪铺满', 'stretch': '拉伸'}[mode] }\n\n"
                        f"{note}\n\n"
                        f"输出文件：{out.name}\n（原 PDF 不会被修改）")

    def _refresh_audio_tracks(self, video: Path):
        try:
            tracks = list_audio_tracks(video)
        except Exception:
            tracks = []
        self._audio_tracks = tracks
        values = []
        for t in tracks:
            lang = f" [{t['lang']}]" if t["lang"] else ""
            default = "（默认）" if t["default"] else ""
            values.append(f"音轨 {t['index'] + 1}: {t['codec']} {t['rate']}Hz "
                          f"{t['channels']}{lang}{default}")
        if not values:
            values = ["（无音轨，将生成静音）"]
        self.audio_combo["values"] = values
        self.audio_combo.current(0)
        self.audio_combo.configure(state="readonly" if tracks else "disabled")

    def _show_info(self, title, body):
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", f"{title}\n{'-' * 28}\n{body}")
        self.info.configure(state="disabled")

    def open_output(self):
        import os
        os.startfile(OUTPUT)

    def on_build(self):
        if self.busy or not self.analysis:
            if not self.analysis:
                messagebox.showinfo("提示", "先在左侧选一个 PDF。")
            return
        item = dict(self.analysis)
        override = {"自动": None, "裁剪铺满": "cover", "等比补边": "letterbox",
                    "拉伸铺满": "stretch"}.get(self.mode.get())
        if override:
            item["mode"] = override
        credits = [part for part in self.credits.get().split("|") if part.strip()]
        audio_track = self.audio_combo.current()
        if audio_track < 0 or not self._audio_tracks:
            audio_track = 0
        color_mode = "high" if self.color_mode.get() == "高色差" else "low"
        self.busy = True
        self.btn_build.configure(state="disabled", text="生成中…")
        threading.Thread(target=self._worker,
                         args=(item, credits, audio_track, color_mode), daemon=True).start()

    def _worker(self, item, credits, audio_track, color_mode):
        def progress(text):
            self.root.after(0, self.status.configure, {"text": text})
        try:
            audio = find_audio(item["npz"].stem.replace("_data", ""), item["video"])
            rebuild(item["npz"], item["video"], audio, item["mode"], item["out"],
                    credits, progress, audio_track=audio_track, color_mode=color_mode)
            self.root.after(0, self._done, item, None)
        except Exception as exc:
            self.root.after(0, self._done, item, str(exc))

    def _done(self, item, error):
        self.busy = False
        self.btn_build.configure(state="normal", text="生成铺满版")
        if error:
            self.status.configure(text="生成失败")
            messagebox.showerror("失败", error)
            return
        self.reload()
        self.status.configure(text=f"已生成：{item['out'].name}（原 PDF 未改动）")
        messagebox.showinfo("完成", f"已生成：\n{item['out']}\n\n原 PDF 未改动，可用「反悔」删除。")

    def on_undo(self):
        if not self.analysis:
            messagebox.showinfo("提示", "先在左侧选一个 PDF。")
            return
        out = self.analysis["out"]
        if not out.is_file():
            messagebox.showinfo("提示", f"没有找到铺满版文件：\n{out.name}")
            return
        if not messagebox.askyesno("反悔", f"删除 {out.name}？\n原 PDF 不会受影响。"):
            return
        out.unlink()
        self.reload()
        self.status.configure(text=f"已删除 {out.name}")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
