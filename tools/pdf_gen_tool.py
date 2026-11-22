"""PDF 生成工具：选视频 + 分割数据，直接生成单页交互 PDF。

- 可切换高色差 / 低色差两种渲染
- 多音轨视频可选音轨
- 只读取已有 *_data.npz，不跑分割推理；已有文件不会被改动
"""

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

FROZEN = getattr(sys, "frozen", False)
HERE = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent.parent
ROOT = HERE.parent if FROZEN and (HERE.parent / "requirements.txt").is_file() else HERE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ascii
from contour.sam_segmenter import SegmentResult
from pdf.player_pdf import (build_ascii_player, build_char_player,
                            list_audio_tracks, tone_profile)

OUTPUT = ROOT / "output"
INPUT = (HERE / "input") if FROZEN else (ROOT / "dist" / "input")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".flv", ".webm", ".avi", ".m4v")

BG = "#101010"
CARD = "#1c1c1c"
TEXT = "#e6e6e6"
SUB = "#8b8b8b"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.busy = False
        self._audio_tracks = []

        root.title("PDF 生成工具")
        root.configure(bg=BG)
        root.geometry("760x560")

        tk.Label(root, text="PDF 生成工具", bg=BG, fg=TEXT,
                 font=("Microsoft YaHei UI", 14, "bold")).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(root, text="选视频 + 分割数据 → 生成单页交互 PDF（不会改动已有文件）",
                 bg=BG, fg=SUB, font=("Microsoft YaHei UI", 9)).pack(anchor="w", padx=16)

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        self.video = tk.StringVar()
        self.npz = tk.StringVar()
        self.credits = tk.StringVar()

        self._file_row(body, "视频文件", self.video, self.browse_video, 0)
        self._file_row(body, "分割数据（*_data.npz）", self.npz, self.browse_npz, 1)

        tk.Label(body, text="色差版本", bg=BG, fg=SUB,
                 font=("Microsoft YaHei UI", 8)).grid(row=2, column=0, sticky="w", pady=(12, 2))
        self.color_mode = tk.StringVar(value="高色差")
        combo = ttk.Combobox(body, values=["高色差", "低色差"], textvariable=self.color_mode,
                             state="readonly")
        combo.grid(row=3, column=0, columnspan=2, sticky="ew")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_output())

        tk.Label(body, text="音轨（多音轨视频可在此选择）", bg=BG, fg=SUB,
                 font=("Microsoft YaHei UI", 8)).grid(row=4, column=0, sticky="w", pady=(12, 2))
        self.audio_combo = ttk.Combobox(body, values=["（未选择视频）"], state="disabled")
        self.audio_combo.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.audio_combo.current(0)

        tk.Label(body, text="片尾字幕（留空则无片尾，多行用 | 分隔）", bg=BG, fg=SUB,
                 font=("Microsoft YaHei UI", 8)).grid(row=6, column=0, sticky="w", pady=(12, 2))
        tk.Entry(body, textvariable=self.credits, bg=CARD, fg=TEXT, bd=0,
                 insertbackground=TEXT, font=("Microsoft YaHei UI", 9)).grid(
            row=7, column=0, columnspan=2, sticky="ew", ipady=5)

        self.out_label = tk.Label(body, text="", bg=BG, fg=SUB, anchor="w",
                                  font=("Microsoft YaHei UI", 8), justify="left")
        self.out_label.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(14, 0))

        self.btn = tk.Button(body, text="生成 PDF", command=self.on_generate,
                             bg="#2f7d4f", fg="white", bd=0, pady=9,
                             activebackground="#3a9a61", activeforeground="white",
                             font=("Microsoft YaHei UI", 10, "bold"), cursor="hand2")
        self.btn.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(12, 6))

        tk.Button(body, text="打开 output 文件夹", command=self.open_output, bg=CARD,
                  fg=TEXT, bd=0, pady=6, activebackground="#2a2a2a",
                  activeforeground=TEXT, font=("Microsoft YaHei UI", 9),
                  cursor="hand2").grid(row=10, column=0, columnspan=2, sticky="ew")

        body.columnconfigure(0, weight=1)

        self.status = tk.Label(root, text="", bg=BG, fg=SUB, anchor="w",
                               font=("Microsoft YaHei UI", 9))
        self.status.pack(fill="x", padx=16, pady=(0, 12))
        self._refresh_output()

    def _file_row(self, parent, label, var, command, row):
        tk.Label(parent, text=label, bg=BG, fg=SUB,
                 font=("Microsoft YaHei UI", 8)).grid(row=row * 2, column=0, sticky="w",
                                                      pady=(0 if row == 0 else 12, 2))
        tk.Entry(parent, textvariable=var, bg=CARD, fg=TEXT, bd=0, insertbackground=TEXT,
                 font=("Microsoft YaHei UI", 9)).grid(row=row * 2 + 1, column=0,
                                                      sticky="ew", ipady=5)
        tk.Button(parent, text="浏览…", command=command, bg=CARD, fg=TEXT, bd=0, padx=10,
                  activebackground="#2a2a2a", activeforeground=TEXT,
                  font=("Microsoft YaHei UI", 9), cursor="hand2").grid(
            row=row * 2 + 1, column=1, padx=(6, 0))

    def browse_video(self):
        folder = INPUT / "待分离" if (INPUT / "待分离").is_dir() else INPUT
        path = filedialog.askopenfilename(
            title="选择视频", initialdir=str(folder),
            filetypes=[("视频", " ".join(f"*{e}" for e in VIDEO_EXTS)), ("所有文件", "*.*")],
        )
        if not path:
            return
        self.video.set(path)
        candidate = OUTPUT / f"{Path(path).stem}_data.npz"
        if candidate.is_file():
            self.npz.set(str(candidate))
        self._refresh_audio(Path(path))
        self._refresh_output()

    def browse_npz(self):
        path = filedialog.askopenfilename(
            title="选择分割数据", initialdir=str(OUTPUT), filetypes=[("分割数据", "*.npz")],
        )
        if path:
            self.npz.set(path)
            self._refresh_output()

    def _refresh_audio(self, video: Path):
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

    def _output_path(self) -> Path | None:
        video = self.video.get().strip().strip('"')
        if not video:
            return None
        suffix = "" if self.color_mode.get() == "高色差" else "_low"
        return OUTPUT / f"{Path(video).stem}_player{suffix}.pdf"

    def _refresh_output(self):
        out = self._output_path()
        self.out_label.configure(text=f"输出：{out}" if out else "输出：选择视频后自动确定")

    def on_generate(self):
        if self.busy:
            return
        video = Path(self.video.get().strip().strip('"'))
        npz = Path(self.npz.get().strip().strip('"'))
        if not video.is_file():
            messagebox.showinfo("提示", "请先选择一个视频文件。")
            return
        if not npz.is_file():
            messagebox.showinfo("提示", "找不到分割数据（*_data.npz）。\n请先完成分割或手动选择。")
            return
        out = self._output_path()
        credits = [part for part in self.credits.get().split("|") if part.strip()]
        audio_track = self.audio_combo.current()
        if audio_track < 0 or not self._audio_tracks:
            audio_track = 0
        color_mode = "high" if self.color_mode.get() == "高色差" else "low"
        self.busy = True
        self.btn.configure(state="disabled", text="生成中…")
        self.status.configure(text="正在读取分割数据…")
        threading.Thread(target=self._worker,
                         args=(npz, video, out, credits, audio_track, color_mode),
                         daemon=True).start()

    def _worker(self, npz, video, out, credits, audio_track, color_mode):
        def progress(text):
            self.root.after(0, self.status.configure, {"text": text})

        try:
            result = SegmentResult.load(str(npz))
            model = str(getattr(result, "model_name", "") or "tiny")
            ascii.set_grid_size(*ascii.grid_size_for_model(model))
            progress("渲染中…")
            high = color_mode == "high"
            if high and tone_profile(result)["mode"] == "color":
                build_char_player(result, str(video), str(out), credits=credits,
                                  credit_sec=3.0, audio_track=audio_track)
            else:
                build_ascii_player(result, video, str(out), credits=credits, credit_sec=3.0,
                                   audio_track=audio_track, color_analysis=high)
            self.root.after(0, self._done, out, None)
        except Exception as exc:
            self.root.after(0, self._done, out, str(exc))

    def _done(self, out, error):
        self.busy = False
        self.btn.configure(state="normal", text="生成 PDF")
        if error:
            self.status.configure(text="生成失败")
            messagebox.showerror("失败", error)
            return
        self.status.configure(text=f"已生成：{out.name}")
        messagebox.showinfo("完成", f"已生成：\n{out}\n\n请使用 Adobe Acrobat 打开。")

    def open_output(self):
        import os
        OUTPUT.mkdir(exist_ok=True)
        os.startfile(OUTPUT)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
