# -*- coding: utf-8 -*-
"""将 SAM2 分割结果生成真正的单页 PDF 播放器。

画面使用分割结果中的前景灰度，播放器复用 make_pdf_direct.py 的
Doom 风格引擎：单页 JavaScript 刷帧、播放/暂停、进度跳转、音量、
片尾字幕和内嵌 MP3 音轨。

PDF JavaScript 与媒体播放需要使用 Adobe Acrobat 等兼容阅读器，
浏览器内置 PDF 阅读器通常只能显示第一帧。
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

import ascii
import make_pdf_direct as player_engine

# 彩色模式的 8 级字符表：小字号下相邻等级越少、墨量差越大，层次越明显
COLOR_RAMP = list(" .:-=+*#")
# 高调画面 10 级：在 13.3pt 下仍能分开，再往上加 o/x/& 会糊成噪声
HIGHLIGHT_RAMP = list(" .:-=+*#%@")


def _edge_magnitude(cg):
    """Lab 三通道 Sobel 的最大梯度，固定卷积核，不随帧自适应。"""
    lab = cv2.cvtColor(cg, cv2.COLOR_BGR2LAB).astype(np.float32)
    mag = np.zeros(cg.shape[:2], dtype=np.float32)
    for ch in range(3):
        gx = cv2.Sobel(lab[:, :, ch], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[:, :, ch], cv2.CV_32F, 0, 1, ksize=3)
        mag = np.maximum(mag, np.abs(gx) + np.abs(gy))
    return mag


def _content_box(result, thresh: int = 25) -> tuple[int, int, int, int]:
    """从分割数据的原始灰度里找出真实画面区域，去掉左右/上下黑边。"""
    frames = result.frames
    height, width = frames[0].gray.shape
    step = max(1, len(frames) // 12)
    acc_c = np.zeros(width, dtype=np.uint8)
    acc_r = np.zeros(height, dtype=np.uint8)
    for frame in frames[::step]:
        acc_c = np.maximum(acc_c, frame.gray.max(axis=0))
        acc_r = np.maximum(acc_r, frame.gray.max(axis=1))
    cols = np.where(acc_c > thresh)[0]
    rows = np.where(acc_r > thresh)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, width, height
    return (max(0, int(cols.min()) + 1), max(0, int(rows.min()) + 1),
            min(width, int(cols.max())), min(height, int(rows.max()) + 1))


def tone_profile(result, sample=24):
    """分析片源色调，返回全片统一的调色参数。

    - silhouette：黑底纯色剪影（Bad Apple），中间调很少，用固定阈值压成干净黑白。
    - color     ：彩色/连续色调，用全片直方图均衡（只算一次，不逐帧漂移）。

    彩色模式还会利用采集时存下的 color_grid（每格平均 BGR）：
      - 以画面四周边缘的中位色作为背景色，计算每格与背景的 Lab 色差 ΔE；
      - 高调画面（大面积亮背景）里黄色主体与白色背景亮度几乎相同，
        此时提高色差权重，让主体从背景里“浮”出来；
      - 普通/暗调画面只给色差很小的权重，保住原本的明暗层次。
    """
    frames = result.frames
    if not frames:
        return dict(mode="silhouette", lo=0, hi=255, lut=None, has_color=False,
                    chroma_w=0.0, bg_lab=None, de_hi=1.0, bright_ratio=0.0)
    step = max(1, len(frames) // sample)
    picked = frames[::step][:sample]
    mids, values = [], []
    for frame in picked:
        source = frame.grid if frame.grid is not None else frame.gray
        v = np.asarray(source, dtype=np.float32).ravel()
        if v.size == 0:
            continue
        values.append(v)
        mids.append(float(((v >= 30) & (v <= 225)).mean()))
    if not values:
        return dict(mode="silhouette", lo=0, hi=255, lut=None, has_color=False,
                    chroma_w=0.0, bg_lab=None, de_hi=1.0, bright_ratio=0.0)
    mid_ratio = float(np.median(mids))
    mode = "color" if mid_ratio >= 0.20 else "silhouette"

    # 全片直方图均衡：裁掉两端 1% 后把 CDF 拉满，只生成一次 LUT
    hist = np.zeros(256, dtype=np.float64)
    for v in values:
        vv = v[v > 1]
        if vv.size < v.size * 0.5:
            vv = v
        hist += np.bincount(vv.astype(np.uint8).ravel(), minlength=256)
    cdf = hist.cumsum()
    total = cdf[-1]
    lo_bin = int(np.searchsorted(cdf, total * 0.01))
    hi_bin = int(np.searchsorted(cdf, total * 0.99))
    if hi_bin - lo_bin < 16:
        lo_bin, hi_bin = 0, 255
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i <= lo_bin:
            lut[i] = 0
        elif i >= hi_bin:
            lut[i] = 255
        else:
            lut[i] = int(round((cdf[i] - cdf[lo_bin]) * 255.0 /
                               max(1.0, cdf[hi_bin] - cdf[lo_bin])))
    bright_ratio = float(np.median([
        float((v[v > 1] > 220).mean()) if (v > 1).any() else 0.0 for v in values
    ]))

    color_frames = [f.color_grid for f in picked if f.color_grid is not None]
    has_color = mode == "color" and len(color_frames) == len(picked)
    chroma_w = 0.78 if bright_ratio >= 0.25 else 0.25
    bg_lab = None
    de_hi = 1.0
    s_hi = 0.15
    if color_frames:
        # 全片统一的饱和度上限：奶龙这类黄色主体 S 高、灰色背景 S 低
        svals = [cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[:, :, 1].ravel()
                 for c in color_frames]
        s_hi = max(0.15, float(np.percentile(np.concatenate(svals), 99)) / 255.0)
    # 高调画面的分层统计：主体内 2%/98% 墨量与边缘强度上限，全片固定
    tone_lo, tone_hi, edge_hi = 0.0, 1.0, 1.0
    if has_color:
        subject_vals, edge_vals = [], []
        for frame in picked:
            c = frame.color_grid
            g = frame.grid if frame.grid is not None else frame.gray
            if g.shape[:2] != (ascii.ROWS, ascii.COLS):
                g = cv2.resize(g, (ascii.COLS, ascii.ROWS), interpolation=cv2.INTER_AREA)
                c = cv2.resize(c, (ascii.COLS, ascii.ROWS), interpolation=cv2.INTER_AREA)
            luma = lut[g].astype(np.float32) / 255.0
            sat = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
            sn = np.clip(sat / max(1e-6, s_hi), 0, 1)
            base = sn * (0.45 + 0.55 * (1.0 - luma))
            gate = sn > 0.10
            if gate.any():
                subject_vals.append(base[gate])
                edge_vals.append(_edge_magnitude(c)[gate])
        if subject_vals:
            tone_lo, tone_hi = np.percentile(np.concatenate(subject_vals), (2, 98))
            edge_hi = max(1e-3, float(np.percentile(np.concatenate(edge_vals), 98)))
            if tone_hi - tone_lo < 0.05:
                tone_lo, tone_hi = 0.0, 1.0
    if has_color:
        borders = []
        flat_deltas = []
        for frame in picked:
            c = frame.color_grid
            g = frame.grid if frame.grid is not None else frame.gray
            if g.shape[:2] != c.shape[:2]:
                g = cv2.resize(g, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid = g > 8  # 排除 letterbox 黑边
            ys = np.where(valid.any(axis=1))[0]
            xs = np.where(valid.any(axis=0))[0]
            if not ys.size or not xs.size:
                continue
            y0, y1, x0, x1 = ys[0], ys[-1], xs[0], xs[-1]
            ph = max(2, (y1 - y0 + 1) // 10)
            pw = max(2, (x1 - x0 + 1) // 10)
            # 取内容区四角的小块做背景色，避免黑边和主体边缘干扰
            for cy, cx in ((y0, x0), (y0, x1 - pw + 1),
                           (y1 - ph + 1, x0), (y1 - ph + 1, x1 - pw + 1)):
                patch = c[cy:cy + ph, cx:cx + pw]
                pmask = valid[cy:cy + ph, cx:cx + pw]
                if pmask.any():
                    borders.append(patch[pmask])
        if not borders:
            for frame in picked:
                c = frame.color_grid
                borders.append(np.concatenate([
                    c[:2].reshape(-1, 3), c[-2:].reshape(-1, 3),
                    c[:, :2].reshape(-1, 3), c[:, -2:].reshape(-1, 3),
                ]))
        bg_bgr = np.median(np.concatenate(borders), axis=0).astype(np.uint8)
        bg_lab = cv2.cvtColor(bg_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)
        bg_lab = bg_lab.reshape(3).astype(np.float32)
        for frame in picked:
            c = frame.color_grid
            g = frame.grid if frame.grid is not None else frame.gray
            if g.shape[:2] != c.shape[:2]:
                g = cv2.resize(g, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST)
            lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)
            delta = np.sqrt(((lab - bg_lab) ** 2).sum(axis=2))
            flat_deltas.append(delta[g > 8].ravel() if (g > 8).any() else delta.ravel())
        de_hi = max(8.0, float(np.percentile(np.concatenate(flat_deltas), 99)))
    return dict(mode=mode, lo=lo_bin, hi=hi_bin, lut=lut, has_color=has_color,
                chroma_w=chroma_w, bg_lab=bg_lab, de_hi=de_hi, s_hi=s_hi,
                tone_lo=tone_lo, tone_hi=tone_hi, edge_hi=edge_hi,
                bright_ratio=bright_ratio, high_key=bright_ratio >= 0.25)


def segment_to_ascii_frames(result, contrast=None, blur=None, autolevel=None,
                            bg_threshold=None, color_analysis=True):
    """把分割数据转换成播放器需要的 ASCII 字符帧。

    默认按片源内容自动选择调色方式：

    剪影模式（Bad Apple 这类黑底纯色画面）：
      - autolevel=False：不做逐帧百分位色阶，避免频闪
      - blur=0：不做高斯模糊，防止边缘产生光晕
      - contrast=5.0 + bg_threshold=35：低灰度清零，高灰度陡拉升，轮廓干净

    彩色/连续色调模式：
      - 全片直方图均衡，只算一次，杜绝逐帧漂移
      - 有 color_grid 时，按“背景色差 ΔE”加权：高调画面权重 0.78，
        普通画面权重 0.25，黄色主体和白色背景也能拉开层次
      - 字符表扩到 10 级（" .:-=+*#%@"），灰阶过渡更细

    color_analysis=False 时只按亮度出字符，不做色差加权（低色差版）。

    优先使用采集阶段从原始分辨率视频采样好的 grid；只有旧数据（没有 grid
    字段）时才退化为从分割灰度缩放。
    """
    profile = tone_profile(result)
    color = profile["mode"] == "color"
    if color and not color_analysis:
        profile = dict(profile, has_color=False, chroma_w=0.0, bg_lab=None,
                       de_hi=1.0, tone_lo=0.0, tone_hi=1.0, edge_hi=1.0,
                       high_key=False)
    if color:
        ramp = HIGHLIGHT_RAMP if profile["high_key"] else COLOR_RAMP
    else:
        ramp = ascii.TEN_ON_DARK
    if color and not color_analysis:
        print(f"[player] 彩色（低色差）：仅按亮度渲染 {profile['lo']}~{profile['hi']}")
    elif color:
        print(f"[player] 检测到彩色/连续色调：直方图均衡 {profile['lo']}~{profile['hi']}，"
              f"高亮占比 {profile['bright_ratio'] * 100:.0f}%，"
              + (f"色度权重 {profile['chroma_w']:.2f}" if profile["has_color"]
                 else "无颜色网格，仅亮度"))
    else:
        print("[player] 检测到黑底剪影：固定阈值 35，对比度 5.0")
    box = _content_box(result)
    frames = []
    for frame in result.frames:
        grid = getattr(frame, "grid", None)
        if grid is not None and grid.shape == (ascii.ROWS, ascii.COLS):
            small = grid
        elif grid is not None:
            # 旧采集数据的网格尺寸不同（如 48 行），重采样到当前网格即可
            small = cv2.resize(grid, (ascii.COLS, ascii.ROWS), interpolation=cv2.INTER_AREA)
        else:
            source = frame.gray
            crop = source[box[1]:box[3], box[0]:box[2]]
            if crop.size == 0:
                crop = source
            small = ascii.fit_grid(crop, mode="cover")
        if blur and blur > 0:
            small = cv2.GaussianBlur(small, (0, 0), blur)

        if not color:
            values = ascii.enhance(small, 5.0, False, bg_threshold=35)
            levels = (values.astype("int32") * len(ramp) // 256).clip(0, len(ramp) - 1)
            frames.append(["".join(ramp[i] for i in row) for row in levels.tolist()])
            continue

        luma = profile["lut"][small].astype(np.float32) / 255.0
        cg = getattr(frame, "color_grid", None)
        if profile["has_color"] and cg is not None:
            if cg.shape[:2] != (ascii.ROWS, ascii.COLS):
                cg = cv2.resize(cg, (ascii.COLS, ascii.ROWS), interpolation=cv2.INTER_AREA)
            lab = cv2.cvtColor(cg, cv2.COLOR_BGR2LAB).astype(np.float32)
            delta = np.sqrt(((lab - profile["bg_lab"]) ** 2).sum(axis=2))
            chroma = np.clip(delta / profile["de_hi"], 0, 1)
            bar = cg.max(axis=2) <= 16
            if profile["high_key"]:
                # 高调画面（亮背景）：饱和度当主墨量，灰白背景自然留空；
                # 再做三层增强（全部使用全片固定参数，零逐帧闪烁）：
                #   1. 固定核 unsharp 提升局部对比
                #   2. 主体 2%~98% 分位拉到满量程，得到清晰的亮度分层
                #   3. Lab 边缘层补五官和轮廓
                # 3x3 中值滤波去掉源画面的网点纹理和孤立脏点
                cg = cv2.medianBlur(cg, 3)
                sat = cv2.cvtColor(cg, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
                sn = np.clip(sat / max(1e-6, profile["s_hi"]), 0, 1)
                base = sn * (0.45 + 0.55 * (1.0 - luma))
                base = np.clip(base + 0.6 * (base - cv2.GaussianBlur(base, (0, 0), 1.0)), 0, 1)
                tone = np.clip((base - profile["tone_lo"]) /
                               max(1e-6, profile["tone_hi"] - profile["tone_lo"]), 0, 1)
                gate = sn > 0.10
                tone[~gate] = 0
                # 压暗中间调，给眼睛/嘴/轮廓留出高墨量的头部空间
                tone = np.power(tone, 2.0)
                edge = np.clip(_edge_magnitude(cg) / max(1e-6, profile["edge_hi"]), 0, 1)
                tone = np.clip(tone + 0.10 * edge * gate, 0, 1)
                tone[tone < 0.06] = 0
            else:
                tone = np.clip(0.82 * luma + 0.18 * chroma, 0, 1)
            tone[bar] = 0  # letterbox 黑边不参与墨量计算
        else:
            tone = luma
        # letterbox 黑边 / 纯黑区域直接留空，避免在高调分支被当成“暗部”涂满
        tone[small <= 1] = 0
        levels = (np.clip(tone, 0, 1) * (len(ramp) - 1)).round().astype("int32")
        frames.append(["".join(ramp[i] for i in row) for row in levels.tolist()])
    if not frames:
        raise RuntimeError("分割结果中没有可播放的帧")
    return frames


def _run_ffmpeg(command):
    return subprocess.run(
        command, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def list_audio_tracks(source_media):
    """列出媒体文件中的音轨，返回按 ffmpeg `0:a:N` 顺序的列表。

    每项为 dict(index, codec, rate, channels, lang, default)；无音轨时返回 []。
    用于在 CLI/GUI 里让用户选择音轨，index 即传给 prepare_player_audio 的 audio_track。
    """
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(source_media)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, OSError):
        return []
    text = result.stderr.decode("utf-8", errors="replace")
    tracks = []
    pattern = re.compile(
        r"Stream #(\d+):(\d+)(?:\[[^\]]*\])?(?:\(([^)]*)\))?:\s*Audio:\s*"
        r"(\w+).*?,\s*(\d+)\s+Hz,\s*([^,\n]+)"
    )
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        tracks.append(dict(
            index=len(tracks),
            stream_index=int(match.group(2)),
            codec=match.group(4),
            rate=int(match.group(5)),
            channels=match.group(6).strip(),
            lang=(match.group(3) or "").strip(),
            default="(default)" in line,
        ))
    return tracks


def prepare_player_audio(source_media, seconds, audio_track=0):
    """从视频/音频提取并转码为 MP3；无音轨或未提供时生成等长静音。

    通用视频通常是 AAC 音轨，不能直接复制到 MP3，因此这里统一重编码。
    audio_track 选择第几条音轨（从 0 开始，对应 ffmpeg 的 `0:a:N`），
    越界或提取失败时回退为等长静音。
    """
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    seconds = max(0.1, float(seconds))
    handle, temp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(handle)
    track_idx = int(audio_track)

    def silence():
        _run_ffmpeg([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", f"{seconds:.4f}", "-c:a", "libmp3lame", "-b:a", "128k",
            temp_path,
        ])

    try:
        if source_media:
            try:
                _run_ffmpeg([
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-t", f"{seconds:.4f}", "-i", str(source_media),
                    "-map", f"0:a:{track_idx}", "-vn",
                    "-c:a", "libmp3lame", "-b:a", "128k",
                    temp_path,
                ])
            except (subprocess.CalledProcessError, OSError):
                print(f"[player] 未读取到可用音轨（音轨 {track_idx}），改用等长静音")
                silence()
        else:
            print("[player] 未提供音频，生成等长静音")
            silence()
        with open(temp_path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def build_char_player(
        result, video_path, out_path, credits=None, credit_sec=3.0, audio_track=0):
    """彩色视频字符画播放器：饱和度主体 + 暗度强调，168x63 网格。

    背景是中性灰，饱和度接近 0，能干净留空；暗度层只补很暗的部分
    （黑阴影、眼周）。不做逐帧色阶拉伸，全片只算一次参数，避免闪烁。
    """
    from pdf import char_render
    # char_render 固定输出 168x63 字符帧；但 make_pdf_direct 的 RLE 往返校验
    # 与页面布局（字号/行高/画行循环/JS 行数）都依赖全局 ascii.COLS/ROWS，
    # 而该全局值按采集模型档位设置（tiny 是 120x45）。若不对齐，
    # encode_segments 会抛“分段行数错误”，build_pdf 布局也会错位。
    import ascii as _ascii
    _ascii.set_grid_size(char_render.COLS, char_render.ROWS)

    frames = char_render.char_frames(video_path, result.fps, len(result.frames))
    count = min(len(frames), len(result.frames))
    frames = frames[:count]
    print(f"[player] 字符画正片 {count} 帧 / {count / result.fps:.2f} 秒")
    segments = player_engine.encode_segments(frames)

    credit_lines = list(credits or [])
    credit_seconds = credit_sec if credit_lines else 0
    audio_seconds = count / result.fps + credit_seconds
    audio_data = prepare_player_audio(video_path, audio_seconds, audio_track=audio_track)
    print(f"[player] 内嵌 MP3 {len(audio_data) / 1024 / 1024:.2f} MB")

    pdf_data, _ = player_engine.build_pdf(
        frames, segments, result.fps, credit_lines, credit_seconds, audio_data,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_data)
    print(f"[player] 完成 {out_path}  {len(pdf_data) / 1024 / 1024:.2f} MB")
    return out_path


def build_ascii_player(
        result, source_media, out_path, credits=None, credit_sec=3.0, audio_track=0,
        contrast=None, blur=None, autolevel=None, bg_threshold=None,
        color_analysis=True):
    """分割结果 + 原视频音轨 -> 单页交互 PDF 播放器。"""
    frames = segment_to_ascii_frames(result, contrast, blur, autolevel, bg_threshold,
                                     color_analysis=color_analysis)
    print(f"[player] ASCII 正片 {len(frames)} 帧 / {len(frames) / result.fps:.2f} 秒")
    segments = player_engine.encode_segments(frames)

    credit_lines = list(credits or [])
    credit_seconds = credit_sec if credit_lines else 0
    audio_seconds = len(frames) / result.fps + credit_seconds
    audio_data = prepare_player_audio(source_media, audio_seconds, audio_track=audio_track)
    print(f"[player] 内嵌 MP3 {len(audio_data) / 1024 / 1024:.2f} MB")

    pdf_data, _ = player_engine.build_pdf(
        frames, segments, result.fps, credit_lines, credit_seconds, audio_data,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_data)
    print(f"[player] 完成 {out_path}  {len(pdf_data) / 1024 / 1024:.2f} MB")
    print("[player] 请使用 Adobe Acrobat 打开；浏览器 PDF 阅读器不支持播放脚本")
    return out_path
