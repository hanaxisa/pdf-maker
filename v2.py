# -*- coding: utf-8 -*-
"""Bad Apple PDF 播放器 V2：160x120 1-bit 位图帧 -> 两种播放引擎

  A (js)  : 单页 + 缓存 2630 个隐藏按钮注释，JS 定时切显隐
  B (dur) : 多页 + /Dur 自动放映，全屏演示模式自动翻页
"""
import argparse
import os
import zlib

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject, BooleanObject, DecodedStreamObject, DictionaryObject,
    EncodedStreamObject, FloatObject, NameObject, NumberObject, TextStringObject,
)

PAGE_W = 480                        # 页面宽度固定，高度按网格比例推算


def page_size(gw, gh):
    return PAGE_W, PAGE_W * gh / gw


def probe_size(video):
    cap = cv2.VideoCapture(video)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return int(w), int(h)


def detect_content_box(video, thresh=25, sample_step=24):
    """扫描采样帧，找出真实画面区域（去掉左右/上下纯黑边）"""
    cap = cv2.VideoCapture(video)
    acc_c = np.zeros(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    acc_r = np.zeros(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % sample_step == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            acc_c = np.maximum(acc_c, g.max(axis=0))
            acc_r = np.maximum(acc_r, g.max(axis=1))
        i += 1
    cap.release()
    cx = np.where(acc_c > thresh)[0]
    ry = np.where(acc_r > thresh)[0]
    if len(cx) == 0 or len(ry) == 0:
        return 0, 0, int(acc_c.size), int(acc_r.size)
    return (int(cx.min()) + 2, int(ry.min()) + 2,
            int(cx.max()) + 1 - 2, int(ry.max()) + 1 - 2)


def extract_bits(video, duration, fps, gw, gh, box=None):
    """抽帧 + 灰度 + Otsu 二值化，返回 [numpy bool(gh,gw)]，True=白"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dt = 1.0 / fps
    next_t = 0.0
    total = int(duration * fps) if duration > 0 else 10 ** 9
    out, idx = [], 0
    while len(out) < total:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / src_fps
        if t + 1e-9 >= next_t:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if box:
                gray = gray[box[1]:box[3], box[0]:box[2]]
            small = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)
            _, bit = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            out.append(bit > 0)
            next_t += dt
        idx += 1
    cap.release()
    return out


def pack(frame):
    """bool 矩阵 -> 1-bit 打包字节，MSB 在前，1=白"""
    return np.packbits(frame.astype(np.uint8), axis=1).tobytes()


def to_pil(frame):
    return Image.fromarray(np.where(frame, 255, 0).astype(np.uint8)).convert("1")


# ---------------------------------------------------------------- B: 多页 /Dur
def build_dur(frames, fps, out, pw, ph, video_title=""):
    tmp = out + ".tmp.pdf"
    c = canvas.Canvas(tmp, pagesize=(pw, ph), pageCompression=1)
    for f in frames:
        c.drawImage(ImageReader(to_pil(f)), 0, 0, pw, ph)
        c.showPage()
    c.save()

    reader = PdfReader(tmp)
    writer = PdfWriter()
    writer.append(reader)
    for p in writer.pages:
        p[NameObject("/Dur")] = FloatObject(1.0 / fps)
        p[NameObject("/Trans")] = DictionaryObject({
            NameObject("/S"): NameObject("/Replace"),
            NameObject("/D"): FloatObject(0),
            NameObject("/Dm"): NameObject("/H"),
        })
    writer._root_object[NameObject("/PageMode")] = NameObject("/FullScreen")
    writer._root_object[NameObject("/NonFullScreenPageMode")] = NameObject("/UseNone")
    if video_title:
        writer.add_metadata({"/Title": video_title})
    with open(out, "wb") as fh:
        writer.write(fh)
    os.remove(tmp)


# ---------------------------------------------------------------- A: 单页 JS
def _img_xobject(writer, packed, gw, gh):
    s = EncodedStreamObject()
    s.update({
        NameObject("/Type"): NameObject("/XObject"),
        NameObject("/Subtype"): NameObject("/Image"),
        NameObject("/Width"): NumberObject(gw),
        NameObject("/Height"): NumberObject(gh),
        NameObject("/ColorSpace"): NameObject("/DeviceGray"),
        NameObject("/BitsPerComponent"): NumberObject(1),
        NameObject("/Filter"): NameObject("/FlateDecode"),
    })
    s._data = zlib.compress(packed, 9)
    return writer._add_object(s)


def _appearance(writer, img_ref, gw, gh):
    a = DecodedStreamObject()
    a.set_data(f"q {gw} 0 0 {gh} 0 0 cm /Im0 Do Q".encode())
    a.update({
        NameObject("/Type"): NameObject("/XObject"),
        NameObject("/Subtype"): NameObject("/Form"),
        NameObject("/FormType"): NumberObject(1),
        NameObject("/BBox"): ArrayObject([NumberObject(0), NumberObject(0),
                                         NumberObject(gw), NumberObject(gh)]),
        NameObject("/Resources"): DictionaryObject({
            NameObject("/XObject"): DictionaryObject({NameObject("/Im0"): img_ref}),
        }),
    })
    return writer._add_object(a)


def build_js(frames, fps, out, gw, gh, pw, ph, video_title=""):
    tmp = out + ".tmp.pdf"
    c = canvas.Canvas(tmp, pagesize=(pw, ph), pageCompression=1)
    c.showPage()
    c.save()

    writer = PdfWriter()
    writer.append(PdfReader(tmp))
    page = writer.pages[0]
    annots = ArrayObject()
    fields = ArrayObject()

    for i, f in enumerate(frames):
        img = _img_xobject(writer, pack(f), gw, gh)
        ap = _appearance(writer, img, gw, gh)
        w = DictionaryObject({
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Widget"),
            NameObject("/Rect"): ArrayObject([NumberObject(0), NumberObject(0),
                                              NumberObject(pw), NumberObject(ph)]),
            NameObject("/F"): NumberObject(0 if i == 0 else 2),   # 2 = Hidden
            NameObject("/FT"): NameObject("/Btn"),
            NameObject("/T"): TextStringObject(f"f{i}"),
            NameObject("/P"): page.indirect_reference,
            NameObject("/AP"): DictionaryObject({NameObject("/N"): ap}),
            NameObject("/MK"): DictionaryObject({
                NameObject("/BC"): ArrayObject(), NameObject("/BG"): ArrayObject(),
            }),
            NameObject("/Border"): ArrayObject([NumberObject(0), NumberObject(0), NumberObject(0)]),
        })
        ref = writer._add_object(w)
        annots.append(ref)
        fields.append(ref)

    page[NameObject("/Annots")] = annots
    writer._root_object[NameObject("/AcroForm")] = DictionaryObject({
        NameObject("/Fields"): fields,
        NameObject("/NeedAppearances"): BooleanObject(False),
        NameObject("/DA"): TextStringObject("/Helv 0 Tf 0 g"),
    })

    n = len(frames)
    ms = round(1000.0 / fps)
    js = (
        "var N=%d, cur=0, fs=[];\n"
        "for(var i=0;i<N;i++) fs[i]=this.getField('f'+i);\n"
        "function step(){ fs[cur].display=display.hidden; cur=(cur+1)%%N; fs[cur].display=display.visible; }\n"
        "app.setInterval('step()', %d);\n"
    ) % (n, ms)
    writer._root_object[NameObject("/OpenAction")] = DictionaryObject({
        NameObject("/S"): NameObject("/JavaScript"),
        NameObject("/JS"): TextStringObject(js),
    })
    if video_title:
        writer.add_metadata({"/Title": video_title})
    with open(out, "wb") as fh:
        writer.write(fh)
    os.remove(tmp)


def main():
    p = argparse.ArgumentParser(description="Bad Apple PDF 播放器 V2")
    p.add_argument("video")
    p.add_argument("-m", "--mode", choices=["js", "dur", "both"], default="both")
    p.add_argument("-o", "--outdir", default="out")
    p.add_argument("--duration", type=float, default=10, help="秒，0=全片")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--grid", default="160xauto", help="宽x高；高填 auto 则按画面比例推算")
    p.add_argument("--crop", choices=["auto", "none"], default="auto", help="自动裁掉纯黑边")
    a = p.parse_args()

    box = None
    if a.crop == "auto":
        box = detect_content_box(a.video)
        print(f"画面有效区域: x{box[0]}..{box[2]} y{box[1]}..{box[3]} "
              f"({box[2]-box[0]}x{box[3]-box[1]})")
    sw, sh = probe_size(a.video)
    if box:
        sw, sh = box[2] - box[0], box[3] - box[1]
    gw_s, gh_s = a.grid.split("x")
    gw = int(gw_s)
    gh = max(1, round(gw * sh / sw)) if gh_s.lower() == "auto" else int(gh_s)
    pw, ph = page_size(gw, gh)
    os.makedirs(a.outdir, exist_ok=True)
    print(f"网格 {gw}x{gh}, 页面 {pw}x{ph:.0f}pt")
    print(f"抽帧 {a.duration or '全片'}s @ {a.fps}fps ...")
    frames = extract_bits(a.video, a.duration, a.fps, gw, gh, box)
    print(f"共 {len(frames)} 帧")

    tag = "full" if a.duration <= 0 else f"{int(a.duration)}s"
    title = "Bad Apple!! PDF Player"
    if a.mode in ("dur", "both"):
        o = os.path.join(a.outdir, f"badapple_v2_dur_{tag}.pdf")
        build_dur(frames, a.fps, o, pw, ph, title)
        print(f"[B 多页/Dur] {o}  {os.path.getsize(o)/1024/1024:.2f} MB  {len(frames)} 页")
    if a.mode in ("js", "both"):
        o = os.path.join(a.outdir, f"badapple_v2_js_{tag}.pdf")
        build_js(frames, a.fps, o, gw, gh, pw, ph, title)
        print(f"[A 单页/JS ] {o}  {os.path.getsize(o)/1024/1024:.2f} MB  {len(frames)} 帧")


if __name__ == "__main__":
    main()
