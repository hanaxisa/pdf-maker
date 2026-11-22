"""表情包选择器：把 input/表情包 里勾选的 gif 转成 mp4，放到 input/待分离。"""

import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from uuid import uuid4

from PIL import Image, ImageTk

BG = "#101010"
CARD = "#1c1c1c"
CARD_SEL = "#1b3d2c"
BORDER = "#2a2a2a"
BORDER_SEL = "#3ddc84"
TEXT = "#e6e6e6"
SUB = "#8b8b8b"
THUMB = 132


def input_root() -> Path:
    """定位 input 目录：exe 同级的 input，脚本运行时兼容 dist/input。"""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent.parent
    for candidate in (base / "input", base / "dist" / "input"):
        if candidate.is_dir():
            return candidate
    return base / "input"


def find_ffmpeg():
    """按顺序找 ffmpeg：exe 同级 bin/ → 系统 PATH → imageio_ffmpeg 自带。"""
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) \
        else Path(__file__).resolve().parent.parent
    for candidate in (base / "bin" / "ffmpeg.exe", base / "ffmpeg.exe"):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def list_gifs(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".gif"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def unique_path(folder: Path, stem: str, suffix: str) -> Path:
    dst = folder / f"{stem}{suffix}"
    index = 2
    while dst.exists():
        dst = folder / f"{stem} ({index}){suffix}"
        index += 1
    return dst


def gif_to_mp4(gif_path: Path, out_dir: Path, loops: int = 3, ffmpeg: str = None) -> Path:
    """把 gif 转成 mp4（不改分辨率，只补齐偶数边长），返回输出路径。

    多帧动图按 loops 循环拼接；单帧静态图没有动画可循环，固定输出 2 秒。
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg，请把 ffmpeg.exe 放到程序同级的 bin 目录")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = unique_path(out_dir, gif_path.stem, ".mp4")
    try:
        with Image.open(gif_path) as im:
            animated = getattr(im, "n_frames", 1) > 1
    except Exception:
        animated = True

    scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    temp_png = None
    if animated:
        if loops > 1:
            cmd += ["-stream_loop", str(loops - 1)]
        cmd += ["-i", str(gif_path), "-vf", scale]
    else:
        # 单帧 gif 没有动画可循环，先导出 png 再按静止画面生成 2 秒
        try:
            temp_png = Path(tempfile.gettempdir()) / f"sticker_{uuid4().hex[:10]}.png"
            with Image.open(gif_path) as im:
                im.convert("RGB").save(temp_png)
            source = temp_png
        except Exception:
            source = gif_path
        cmd += ["-f", "image2", "-loop", "1", "-framerate", "12", "-t", "2",
                "-i", str(source), "-vf", scale, "-r", "12"]
    cmd += ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
            "-movflags", "+faststart", "-an", str(out)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    finally:
        if temp_png is not None:
            temp_png.unlink(missing_ok=True)
    if result.returncode != 0 or not out.exists():
        raise RuntimeError((result.stderr or "ffmpeg 转换失败").strip().splitlines()[-1])
    return out


class Card(tk.Frame):
    def __init__(self, master, path: Path, on_toggle):
        super().__init__(master, bg=CARD, highlightthickness=2,
                         highlightbackground=BORDER)
        self.path = path
        self.selected = tk.BooleanVar(value=False)
        self.on_toggle = on_toggle

        image = self._thumb(path)
        tk.Label(self, image=image, bg=CARD, bd=0).pack(padx=6, pady=(6, 2))
        self._image = image

        tk.Label(self, text=path.name, bg=CARD, fg=TEXT, font=("Microsoft YaHei UI", 8),
                 wraplength=THUMB, justify="center").pack(padx=4)
        tk.Label(self, text=f"{path.stat().st_size / 1024:.0f} KB", bg=CARD, fg=SUB,
                 font=("Microsoft YaHei UI", 8)).pack(pady=(0, 6))

        for widget in (self, *self.winfo_children()):
            widget.bind("<Button-1>", self._click)

    def _thumb(self, path: Path):
        try:
            with Image.open(path) as im:
                im.seek(0)
                frame = im.convert("RGBA")
                frame.thumbnail((THUMB, THUMB), Image.LANCZOS)
                canvas = Image.new("RGBA", (THUMB, THUMB), (10, 10, 10, 255))
                canvas.alpha_composite(frame, ((THUMB - frame.width) // 2,
                                               (THUMB - frame.height) // 2))
                return ImageTk.PhotoImage(canvas.convert("RGB"))
        except Exception:
            return ImageTk.PhotoImage(Image.new("RGB", (THUMB, THUMB), (30, 30, 30)))

    def _click(self, _event=None):
        self.selected.set(not self.selected.get())
        self._paint()
        self.on_toggle()

    def _paint(self):
        on = self.selected.get()
        self.configure(highlightbackground=BORDER_SEL if on else BORDER)
        self.configure(bg=CARD_SEL if on else CARD)
        for child in self.winfo_children():
            child.configure(bg=CARD_SEL if on else CARD)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.input = input_root()
        self.sticker_dir = self.input / "表情包"
        self.staging = self.input / "待分离"
        self.sticker_dir.mkdir(parents=True, exist_ok=True)
        self.cards = []
        self.busy = False

        root.title("表情包选择器 · gif 转 mp4")
        root.configure(bg=BG)
        root.geometry("920x720")

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(header, text="表情包选择器", bg=BG, fg=TEXT,
                 font=("Microsoft YaHei UI", 14, "bold")).pack(side="left")
        tk.Label(header, text="点击卡片选中 → 转成 mp4 存进待分离", bg=BG, fg=SUB,
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=10)

        bar = tk.Frame(root, bg=BG)
        bar.pack(fill="x", padx=16, pady=(0, 8))
        for text, command in (("全选", self.select_all), ("全不选", self.clear_all),
                              ("刷新", self.reload), ("打开表情包文件夹", self.open_folder)):
            tk.Button(bar, text=text, command=command, bg=CARD, fg=TEXT, bd=0,
                      padx=14, pady=6, activebackground="#2a2a2a",
                      activeforeground=TEXT, font=("Microsoft YaHei UI", 9),
                      cursor="hand2").pack(side="left", padx=(0, 8))

        self.save_btn = tk.Button(bar, text="转成 mp4 并存进待分离", command=self.convert_selected,
                                  bg="#2f7d4f", fg="white", bd=0, padx=18, pady=6,
                                  activebackground="#3a9a61", activeforeground="white",
                                  font=("Microsoft YaHei UI", 9, "bold"), cursor="hand2")
        self.save_btn.pack(side="right")

        self.loops = tk.StringVar(value="3")
        loop_box = tk.Frame(bar, bg=BG)
        loop_box.pack(side="right", padx=10)
        tk.Label(loop_box, text="循环", bg=BG, fg=SUB,
                 font=("Microsoft YaHei UI", 9)).pack(side="left")
        tk.Spinbox(loop_box, from_=1, to=20, width=3, textvariable=self.loops, bd=0,
                   bg=CARD, fg=TEXT, justify="center",
                   font=("Microsoft YaHei UI", 9)).pack(side="left", padx=4)
        tk.Label(loop_box, text="遍", bg=BG, fg=SUB,
                 font=("Microsoft YaHei UI", 9)).pack(side="left")

        wrapper = tk.Frame(root, bg=BG)
        wrapper.pack(fill="both", expand=True, padx=16)
        self.canvas = tk.Canvas(wrapper, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(wrapper, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=BG)
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._layout)
        self.canvas.bind_all("<MouseWheel>", self._wheel)

        self.status = tk.Label(root, text="", bg=BG, fg=SUB, anchor="w",
                               font=("Microsoft YaHei UI", 9))
        self.status.pack(fill="x", padx=16, pady=(6, 12))

        self.reload()

    def _wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _layout(self, event):
        self.window.configure(width=event.width)
        self._grid()

    def _grid(self):
        width = max(self.canvas.winfo_width(), 1)
        per_row = max(1, width // (THUMB + 40))
        for i, card in enumerate(self.cards):
            card.grid(row=i // per_row, column=i % per_row, padx=8, pady=8, sticky="n")

    def reload(self):
        for card in self.cards:
            card.destroy()
        self.cards = []
        files = list_gifs(self.sticker_dir)
        for path in files:
            self.cards.append(Card(self.inner, path, self.update_status))
        if not files:
            tk.Label(self.inner, text=f"这里还没有 gif\n把表情包放进：{self.sticker_dir}",
                     bg=BG, fg=SUB, font=("Microsoft YaHei UI", 10),
                     justify="left").grid(row=0, column=0, padx=8, pady=20, sticky="w")
        self._grid()
        self.update_status()

    def select_all(self):
        for card in self.cards:
            card.selected.set(True)
            card._paint()
        self.update_status()

    def clear_all(self):
        for card in self.cards:
            card.selected.set(False)
            card._paint()
        self.update_status()

    def update_status(self):
        chosen = sum(1 for c in self.cards if c.selected.get())
        self.status.configure(
            text=f"表情包 {len(self.cards)} 个 · 已选 {chosen} 个 · 输出目录：{self.staging}")
        self.save_btn.configure(text=f"转成 mp4 并存进待分离 ({chosen})" if chosen
                                else "转成 mp4 并存进待分离")

    def open_folder(self):
        import os
        os.startfile(self.sticker_dir)

    def convert_selected(self):
        if self.busy:
            return
        chosen = [c.path for c in self.cards if c.selected.get()]
        if not chosen:
            messagebox.showinfo("提示", "先点几张表情包再存。")
            return
        try:
            loops = max(1, int(self.loops.get()))
        except ValueError:
            loops = 3
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            messagebox.showerror("缺少 ffmpeg", "没找到 ffmpeg，请把 ffmpeg.exe 放到程序同级的 bin 目录。")
            return
        self.busy = True
        self.save_btn.configure(state="disabled", text="转换中…")
        threading.Thread(target=self._worker, args=(chosen, loops, ffmpeg),
                         daemon=True).start()

    def _worker(self, chosen, loops, ffmpeg):
        done, failed = 0, []
        for index, path in enumerate(chosen, 1):
            self.root.after(0, self.status.configure,
                            {"text": f"正在转换 {index}/{len(chosen)}：{path.name}"})
            try:
                gif_to_mp4(path, self.staging, loops, ffmpeg)
                done += 1
            except Exception as exc:
                failed.append(f"{path.name}：{exc}")
        self.root.after(0, self._finish, done, failed)

    def _finish(self, done, failed):
        self.busy = False
        self.save_btn.configure(state="normal")
        self.clear_all()
        message = f"已生成 {done} 个 mp4，输出到：\n{self.staging}"
        if failed:
            message += "\n\n失败：\n" + "\n".join(failed[:6])
            messagebox.showwarning("完成（部分失败）", message)
        else:
            messagebox.showinfo("完成", message)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
