# -*- coding: utf-8 -*-
"""Bad Apple PDF：DoomPDF 式逐行文本域播放器 + Acrobat legacy MP3 时钟。"""
import argparse
import os
import subprocess
import tempfile
import zlib

from PIL import Image, ImageDraw, ImageFont

import ascii
from v2 import detect_content_box

PAGE_W, PAGE_H = 960, 768
SCREEN_H = 720
CTRL_H = PAGE_H - SCREEN_H
# 屏幕区按「一行一个只读单行文本域」铺满：单行域不会换行、也不会被自动缩放。
# 字号取 COLS 列正好 960pt、行高取 720/ROWS，域左边界放在 -2，Acrobat 默认 2pt
# 内边距正好把正文字推到 x=0，最后一列贴着页面右边缘，不会留黑边。
#
# 注意：字号/行高必须在 build_pdf 时按当时的 ascii.COLS/ROWS 现算。
# ascii.set_grid_size() 可能在模块 import 之后才被调用（高/中/低档网格不同），
# 若在模块级算成常量，档位切换后字号会停留在旧值，导致每行超宽被裁切、行数超页。
def row_metrics(advance=0.6):
    """返回当前网格对应的 (字号, 行高)，均以 pt 为单位。

    advance 是字形的步进宽度（相对字号）：Courier 0.6，点阵字体 1.0。
    """
    return PAGE_W / (ascii.COLS * advance), SCREEN_H / ascii.ROWS


SEG_FRAMES = 4
PDF_HEADER = b"%PDF-1.6\n%\xe2\xe3\xcf\xd3\n"


def pdf_escape(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_hex(s):
    return s.encode("latin-1").hex().upper()


def rle_encode(line):
    out = []
    start = 0
    while start < len(line):
        ch = line[start]
        # 计数只由数字和小写字母组成；大写字母、标点是安全的填充字符
        if ch.isdigit() or ("a" <= ch <= "z") or ch in "|~":
            raise ValueError(f"字符不能用于此 RLE：{ch!r}")
        end = start + 1
        while end < len(line) and line[end] == ch:
            end += 1
        digits = ""
        n = end - start
        while n:
            n, rem = divmod(n, 36)
            digits = "0123456789abcdefghijklmnopqrstuvwxyz"[rem] + digits
        out.append((digits or "0") + ch)
        start = end
    return "".join(out)


def rle_decode(encoded):
    out = []
    i = 0
    while i < len(encoded):
        j = i
        while j < len(encoded) and (encoded[j].isdigit() or "a" <= encoded[j] <= "z"):
            j += 1
        if j == len(encoded):
            raise ValueError("RLE 缺少字符")
        count = int(encoded[i:j], 36)
        out.append(encoded[j] * count)
        i = j + 1
    line = "".join(out)
    if len(line) != ascii.COLS:
        raise ValueError(f"RLE 行宽错误：{len(line)}")
    return line


def encode_segment(lines4):
    blocks = []
    previous = None
    for lines in lines4:
        encoded_rows = []
        for row, line in enumerate(lines):
            if previous is not None and line == previous[row]:
                encoded_rows.append("")
            else:
                encoded_rows.append(rle_encode(line))
        blocks.append("|".join(encoded_rows))
        previous = lines
    return "~".join(blocks)


def decode_segment(payload):
    frames = []
    previous = None
    for block in payload.split("~"):
        cells = block.split("|")
        if len(cells) != ascii.ROWS:
            raise ValueError("分段行数错误")
        current = []
        for row, cell in enumerate(cells):
            if previous is not None and not cell:
                current.append(previous[row])
            else:
                current.append(rle_decode(cell))
        frames.append(current)
        previous = current
    return frames


def encode_segments(frames):
    segments = []
    largest = 0
    for start in range(0, len(frames), SEG_FRAMES):
        expected = frames[start:start + SEG_FRAMES]
        payload = encode_segment(expected)
        if decode_segment(payload) != expected:
            raise AssertionError(f"分段 {start // SEG_FRAMES} RLE 往返校验失败")
        size = len(payload.encode("latin-1"))
        if size >= 65535:
            raise ValueError(f"分段 {start // SEG_FRAMES} 超过 65535 字节：{size}")
        largest = max(largest, size)
        segments.append(payload)
    print(f"数据分段 {len(segments)} 个，最大 {largest} 字节；RLE 往返校验通过")
    return segments


JS_TEMPLATE = r"""
var baDoc = this;
var baFps = __FPS__;
var baFrames = __FRAMES__;
var baMainDur = __DURATION__;
var baCreditDur = __CREDITS__;
var baRows = [];
var baLast = [];
var baStatus = null;
var baCredits = null;
var baCacheSeg = -1;
var baCache = [];
var baFrame = -1;
var baPlayer = null;
var baPlayerAlive = false;
var baPlaying = false;
var baFinished = false;
var baInCredits = false;
var baCreditEnd = 0;
var baCreditLeft = baCreditDur * 1000;
var baOffset = 0;
var baStarted = 0;
var baMediaClock = true;
var baVolume = 100;
var baTimer = null;
var baReady = false;
var baStatusAt = 0;
var baAudioError = "";
var baAudioRetry = 0;
var baPlayStartMs = 0;
var baAudioOk = false;
var baStep = "start";
var baRendition = null;
var baRenditionAnnot = null;

function baMs() { return (new Date()).getTime(); }
function baPad(n) {
    var s = "" + n;
    while (s.length < 4) s = "0" + s;
    return s;
}
function baRle(s) {
    var out = "", i = 0;
    while (i < s.length) {
        var j = i;
        while (j < s.length) {
            var c = s.charCodeAt(j);
            if ((c >= 48 && c <= 57) || (c >= 97 && c <= 122)) j++;
            else break;
        }
        var n = parseInt(s.substring(i, j), 36);
        var ch = s.charAt(j);
        var parts = [];
        for (var k = 0; k < n; k++) parts.push(ch);
        out += parts.join("");
        i = j + 1;
    }
    return out;
}
function baDecodeSegment(s) {
    var result = [], previous = null;
    var blocks = s.split("~");
    for (var f = 0; f < blocks.length; f++) {
        var cells = blocks[f].split("|");
        var current = [];
        for (var r = 0; r < __ROWS__; r++) {
            if (previous !== null && cells[r].length === 0) current[r] = previous[r];
            else current[r] = baRle(cells[r]);
        }
        result.push(current);
        previous = current;
    }
    return result;
}
function baLoadSegment(seg) {
    if (seg === baCacheSeg) return;
    var field = baDoc.getField("d" + baPad(seg));
    if (!field) throw new Error("missing d" + baPad(seg));
    baCache = baDecodeSegment(field.value);
    baCacheSeg = seg;
}
function baDraw(frame) {
    if (frame < 0) frame = 0;
    if (frame >= baFrames) frame = baFrames - 1;
    var seg = Math.floor(frame / 4);
    baLoadSegment(seg);
    var lines = baCache[frame - seg * 4];
    // One single-line field per row: rewrite only changed rows. A single-line
    // field never wraps and never gets auto-scaled, so the picture fills the page.
    for (var r = 0; r < __ROWS__; r++) {
        if (baLast[r] !== lines[r]) {
            baRows[r].value = lines[r];
            baLast[r] = lines[r];
        }
    }
    baFrame = frame;
}
function baFmt(t) {
    if (t < 0) t = 0;
    var n = Math.floor(t);
    var m = Math.floor(n / 60);
    var s = n % 60;
    return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
}
function baShowStatus(label) {
    if (!baStatus) return;
    var msg = label + "  " + baFmt(baOffset) + " / " +
        baFmt(baMainDur) + "  VOL " + Math.round(baVolume) + "%" +
        (baPlayer && baPlayerAlive && baMediaClock ? "  M" : "  T") +
        (baAudioError ? "  ERR:" + baAudioError + " v" + app.viewerVersion : "");
    if (baStatus.value !== msg) baStatus.value = msg;
}
function baClock() {
    if (!baPlaying) return baOffset;
    var elapsed = baOffset + (baMs() - baStarted) / 1000;
    if (baPlayer && baPlayerAlive && baMediaClock) {
        try {
            if (baPlayer.isOpen) {
                var w = baPlayer.where();
                if (w && typeof w.time === "number" && isFinite(w.time) && w.time >= 0) {
                    // where().time should be seconds. If it looks like ms or frames, drop it.
                    var diff = w.time - elapsed;
                    if (diff < 0) diff = -diff;
                    if (diff <= 0.6) return w.time;
                    baMediaClock = false;
                }
            }
        } catch (e) {
            baMediaClock = false;
        }
    }
    return elapsed;
}
function baClosePlayer() {
    // Drop our reference BEFORE closing: once Acrobat closes a player the
    // object is invalidated, and any later property access raises a fatal
    // "Object has died" error dialog.
    var p = baPlayer;
    baPlayer = null;
    baPlayerAlive = false;
    baMediaClock = false;
    if (!p) return;
    try { p.close(); } catch (e1) {}
}
function baPausePlayer() {
    // Keep the player alive across pause so resume never recreates it.
    if (!baPlayer || !baPlayerAlive) return;
    try {
        if (baPlayer.isOpen) baPlayer.pause();
    } catch (e) {
        baClosePlayer();
    }
}
function baOpenCodeName(c) {
    try {
        for (var k in app.media.openCode) {
            if (app.media.openCode[k] === c) return k;
        }
    } catch (e) {}
    return "" + c;
}
function baFindAnnot() {
    if (baRenditionAnnot) return baRenditionAnnot;
    // Doc.media has getAnnot only (no getAnnots); fetch the Screen annotation by title.
    try {
        var a = baDoc.media.getAnnot({nPage: 0, cAnnotTitle: "baAudio"});
        if (a) return a;
    } catch (e1) {}
    return null;
}
function baGrabEvent() {
    // The Play/Pause button is itself a Rendition action, so event.action
    // carries a ready-made rendition/annot and we skip the name lookup.
    try {
        if (event && event.action) {
            if (event.action.rendition) baRendition = event.action.rendition;
            if (event.action.annot) baRenditionAnnot = event.action.annot;
        }
    } catch (e) {}
}
function baToggleBtn() {
    baGrabEvent();
    baToggle();
}
function baReplayBtn() {
    baGrabEvent();
    baReplay();
}
function baFindRendition() {
    if (baRendition) return baRendition;
    var names = ["baAudio", "badapple audio", "badapple.mp3"];
    for (var i = 0; i < names.length; i++) {
        try {
            var r = baDoc.media.getRendition(names[i]);
            if (r) { baRendition = r; return r; }
        } catch (e1) {}
    }
    return null;
}
function baOpenPlayers() {
    // app.media.getPlayers() lists available player *types* (PlayerInfo), not
    // opened players. Open players live in doc.media.getOpenPlayers().
    try {
        var list = baDoc.media.getOpenPlayers(baDoc);
        if (list) return list;
    } catch (e0) {}
    try { return baDoc.media.getOpenPlayers(); } catch (e1) {}
    return null;
}
function baMp3Players() {
    try {
        var list = baOpenPlayers();
        return list ? list.length : 0;
    } catch (e) { return -1; }
}
function baAdoptPlayer() {
    // Acrobat may already have opened a player for this screen annotation.
    // Only one player per annotation is allowed, so reuse it instead of
    // creating a second one (createPlayer returns null in that case).
    var list = baOpenPlayers();
    if (!list || !list.length) return null;
    var annot = null;
    try { annot = baFindAnnot(); } catch (e0) {}
    var fallback = null;
    for (var i = 0; i < list.length; i++) {
        var p = list[i];
        var a = null;
        try { a = p.annot; } catch (e1) { continue; }  // destroyed / closed document
        try {
            var d = p.doc;
            if (d && d !== baDoc) continue;
        } catch (e2) {}
        if (annot && a) {
            try { if (a === annot) return p; } catch (e3) {}
        }
        if (!fallback) fallback = p;
    }
    return fallback;
}
function baOpenAudio(t) {
    baAudioError = "";
    // 1) Reuse our live player: seek + play.
    if (baPlayer && baPlayerAlive) {
        try {
            if (!baPlayer.isOpen) throw new Error("closed");
        } catch (ed) {
            baClosePlayer();
        }
    }
    if (baPlayer && baPlayerAlive) {
        try {
            try { baPlayer.seek(t); } catch (es) {}
            baPlayer.play();
            baMediaClock = true;
            return;
        } catch (er) {
            baClosePlayer();
        }
    }
    // 2) Adopt a player Acrobat already created for our Screen annotation.
    var adopted = baAdoptPlayer();
    if (adopted) {
        baPlayer = adopted;
        baPlayerAlive = true;
        try {
            try { if (!baPlayer.isOpen) baPlayer.open(); } catch (eo0) {}
            try { baPlayer.seek(t); } catch (es0) {}
            baPlayer.play();
            baMediaClock = true;
            return;
        } catch (ea) {
            baClosePlayer();
        }
    }
    var annot = baFindAnnot();
    if (!annot) throw new Error("no annot");
    // rendition may be null when the name tree lookup fails; Acrobat can still
    // resolve the clip from the annotation itself, so keep going without it.
    var rendition = baFindRendition();
    var st = {
        autoPlay: true,
        startAt: t,
        showUI: false,
        visible: false,
        // MediaSettings.volume: 0 = muted, 100 = full volume.
        volume: baVolume
    };
    try {
        var wt = app.media.windowType.docked;
        if (wt !== undefined && wt !== null) st.windowType = wt;
    } catch (ew) {}
    var ev = null;
    try {
        ev = new app.media.Events({
            onReady: function () {
                try {
                    if (baPlayer && baPlayerAlive && baPlayer.isOpen && !baPlayer.isPlaying) baPlayer.play();
                } catch (er) {}
            }
        });
    } catch (ee) {}
    var args = {
        annot: annot,
        fromUser: true,
        settings: st
    };
    if (rendition) args.rendition = rendition;
    if (ev) args.events = ev;
    try { baPlayer = app.media.createPlayer(args); } catch (ec) { baPlayer = null; }
    if (!baPlayer) {
        // Inside a rendition action Acrobat already knows which annotation and
        // clip to use, so the no-argument createPlayer() is the safest call.
        try {
            if (event && event.action && event.action.rendition) {
                baPlayer = app.media.createPlayer();
            }
        } catch (ec2) { baPlayer = null; }
    }
    if (!baPlayer) throw new Error("create null r" + (rendition ? 1 : 0) +
                                   " p" + baMp3Players());
    baPlayerAlive = true;
    var res = null;
    try { res = baPlayer.open(); } catch (eo) {
        baPlayer = null;
        baPlayerAlive = false;
        throw new Error("open ex");
    }
    if (res && typeof res.code === "number" && app.media.openCode &&
            typeof app.media.openCode.success === "number") {
        if (res.code !== app.media.openCode.success) {
            baPlayer = null;
            baPlayerAlive = false;
            throw new Error("open " + baOpenCodeName(res.code));
        }
    }
    baMediaClock = true;
    // open() may not start playback, so call play() explicitly.
    // Do not gate on !isPlaying: some Acrobat builds report isPlaying=true
    // right after open() while the player is actually paused (silent).
    try {
        if (baPlayer && baPlayerAlive) baPlayer.play();
    } catch (ep) {}
}
function baPlay() {
    if (baFinished) { baReplay(); return; }
    if (baPlaying) return;
    if (baInCredits) {
        baCreditEnd = baMs() + baCreditLeft;
        baPlaying = true;
        baShowStatus("CREDITS");
        return;
    }
    baAudioRetry = 0;
    baPlayStartMs = baMs();
    baAudioOk = false;
    try {
        // Player.open() starts playback itself. Do not seek/play right after it.
        baOpenAudio(baOffset);
    } catch (e) {
        baClosePlayer();
        baAudioError = ("" + (e.message || e)).substring(0, 40);
    }
    baStarted = baMs();
    baPlaying = true;
    baShowStatus("PLAY");
}
function baPause() {
    if (!baPlaying) return;
    if (baInCredits) {
        baCreditLeft = Math.max(0, baCreditEnd - baMs());
        baPlaying = false;
        baShowStatus("PAUSE");
        return;
    }
    baOffset = Math.min(baMainDur, baClock());
    baPausePlayer();
    baPlaying = false;
    baStarted = baMs();
    baDraw(Math.floor(baOffset * baFps));
    baShowStatus("PAUSE");
}
function baToggle() { if (baPlaying) baPause(); else baPlay(); }
function baEnterCredits() {
    baOffset = baMainDur;
    baClosePlayer();
    baInCredits = true;
    baCreditLeft = baCreditDur * 1000;
    baCreditEnd = baMs() + baCreditLeft;
    try { if (baCredits) baCredits.hidden = false; } catch (e0) {}
    if (baCreditDur <= 0) {
        baInCredits = false;
        baPlaying = false;
        baFinished = true;
        baShowStatus("FINISHED - REPLAY");
    } else {
        baShowStatus("CREDITS");
    }
}
function baReplay() {
    baClosePlayer();
    baOffset = 0;
    baStarted = baMs();
    baPlaying = false;
    baFinished = false;
    baInCredits = false;
    try { if (baCredits) baCredits.hidden = true; } catch (e0) {}
    baDraw(0);
    baPlay();
}
function baSeek(delta) {
    var fromCredits = baInCredits || baFinished;
    var resume = baPlaying;
    var t = fromCredits ? baMainDur : baClock();
    var next = t + delta;
    var lastTime = Math.max(0, baMainDur - 0.001);
    if (next < 0) next = 0;
    if (next > lastTime) next = lastTime;
    if (fromCredits) {
        baInCredits = false;
        baFinished = false;
        baPlaying = false;
        try { if (baCredits) baCredits.hidden = true; } catch (e0) {}
    }
    baOffset = next;
    baStarted = baMs();
    if (resume) {
        try { baOpenAudio(next); } catch (e) {
            baAudioError = ("" + (e.message || e)).substring(0, 40);
        }
    }
    baDraw(Math.floor(next * baFps));
    baPlaying = resume;
    baShowStatus(resume ? "PLAY" : "SEEK");
}
function baVolumeStep(delta) {
    baVolume += delta;
    if (baVolume < 0) baVolume = 0;
    if (baVolume > 100) baVolume = 100;
    // Do not reopen the player here (that kills the old player object).
    // Just nudge the live player's media settings; ignore if unsupported.
    if (baPlayer && baPlayerAlive) {
        try { baPlayer.settings.volume = baVolume; } catch (ev) {}
    }
    baShowStatus(baPlaying ? "PLAY" : "VOLUME");
}
function baExportAudio() {
    // Guaranteed audio path: extract the embedded MP3 and let the OS player
    // play it. Works even when Acrobat blocks legacy PDF multimedia.
    try { baDoc.exportDataObject({ cName: "badapple.mp3", nLaunch: 2 }); }
    catch (e) { baShowStatus("AUDIO EXPORT FAIL"); }
}
function baTick() {
    try {
        baTickCore();
    } catch (e) {
        baPlaying = false;
        if (baStatus) baStatus.value = "TICK ERROR: " + (e.message || e);
    }
}
function baTickCore() {
    if (!baPlaying) return;
    if (baInCredits) {
        if (baMs() >= baCreditEnd) {
            baInCredits = false;
            baPlaying = false;
            baFinished = true;
            baShowStatus("FINISHED - REPLAY");
        }
        return;
    }
    var t = baClock();
    // Audio watchdog: if the media clock never takes over, the player is
    // stuck at 0s (picture moves, no sound). Reopen it, at most twice.
    if (!baInCredits) {
        if (baMediaClock) {
            baAudioOk = true;
        } else if (!baAudioOk && baPlayerAlive && baAudioRetry < 2 &&
                   baMs() - baPlayStartMs > 1200) {
            baAudioRetry++;
            baPlayStartMs = baMs();
            try { baOpenAudio(baOffset); } catch (e) {
                baAudioError = ("" + (e.message || e)).substring(0, 40);
            }
        } else if (!baAudioOk && !baAudioError && baAudioRetry >= 2 &&
                   baMs() - baPlayStartMs > 2000) {
            // Player was created but never advanced: Acrobat blocked legacy media.
            baAudioError = "MEDIA BLOCKED - allow multimedia";
        }
    }
    if (t >= baMainDur) { baEnterCredits(); return; }
    var frame = Math.floor(t * baFps);
    if (frame !== baFrame) baDraw(frame);
    baOffset = t;
    baStarted = baMs();
    var now = baStarted;
    if (now - baStatusAt > 250) {
        baShowStatus("PLAY");
        baStatusAt = now;
    }
}
function baShutdown() {
    baPlaying = false;
    baClosePlayer();
    if (baTimer) {
        try { app.clearInterval(baTimer); } catch (e) {}
        baTimer = null;
    }
}
function baInit() {
    if (baReady) return;
    baReady = true;
    baStatus = baDoc.getField("status");
    try {
        baStep = "screen";
        for (var r = 0; r < __ROWS__; r++) {
            var fld = baDoc.getField("r" + r);
            if (!fld) throw new Error("missing r" + r);
            baRows.push(fld);
            baLast.push("");
            // Set explicit colors so Acrobat does not flash a white background.
            try {
                fld.fillColor = ["RGB", 0, 0, 0];
                fld.textColor = ["RGB", 0.8, 0.8, 0.8];
            } catch (ec) {}
        }
        baStep = "credits";
        baCredits = baDoc.getField("credits");
        baStep = "draw";
        baDraw(0);
        baStep = "status";
        baShowStatus("READY - CLICK PLAY");
        // Show the viewer version so media problems can be diagnosed from a screenshot.
        try { baStatus.value = "READY - CLICK PLAY  v" + app.viewerVersion; } catch (ev3) {}
        baStep = "timer";
        var tickExpr = "try{baTick();}catch(e){baPlaying=false;if(baStatus)baStatus.value='TICK ERROR: '+(e.message||e);}";
        try {
            baTimer = app.setInterval(tickExpr, Math.round(1000 / baFps));
        } catch (et) {
            baTimer = app.setInterval(tickExpr, 100);
        }
    } catch (e) {
        if (baStatus) baStatus.value = "INIT ERROR@" + baStep + ": " + (e.message || e);
    }
}
"""


def make_js(frame_count, fps, credit_sec):
    duration = frame_count / fps
    return (JS_TEMPLATE
            .replace("__FPS__", str(fps))
            .replace("__FRAMES__", str(frame_count))
            .replace("__ROWS__", str(ascii.ROWS))
            .replace("__DURATION__", f"{duration:.8f}")
            .replace("__CREDITS__", f"{credit_sec:.3f}"))


class PdfBuilder:
    def __init__(self):
        self.objects = {}
        self.next_id = 1

    def reserve(self):
        obj_id = self.next_id
        self.next_id += 1
        return obj_id

    def set(self, obj_id, body):
        self.objects[obj_id] = body.encode("latin-1") if isinstance(body, str) else body

    def add(self, body):
        obj_id = self.reserve()
        self.set(obj_id, body)
        return obj_id

    def stream(self, entries, data):
        body = (f"<< {entries} /Length {len(data)} >>\nstream\n".encode("latin-1")
                + data + b"\nendstream\nendobj\n")
        return self.add(body)

    def render(self, root_id):
        if set(self.objects) != set(range(1, self.next_id)):
            raise AssertionError("PDF 对象编号有空洞")
        out = bytearray(PDF_HEADER)
        offsets = {}
        for obj_id in sorted(self.objects):
            body = self.objects[obj_id]
            if not body.endswith(b"endobj\n"):
                body += b"endobj\n"
            offsets[obj_id] = len(out)
            out.extend(f"{obj_id} 0 obj\n".encode("latin-1"))
            out.extend(body)
            out.extend(b"\n")
        xref = len(out)
        out.extend(f"xref\n0 {self.next_id}\n".encode("latin-1"))
        out.extend(b"0000000000 65535 f \n")
        for obj_id in range(1, self.next_id):
            out.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode("latin-1"))
        out.extend((f"trailer\n<< /Size {self.next_id} /Root {root_id} 0 R >>\n"
                    f"startxref\n{xref}\n%%EOF\n").encode("latin-1"))
        return bytes(out)


def credits_image(lines, width, height):
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 40)
    step = 88
    y = (height - step * len(lines)) // 2
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        draw.text(((width - (box[2] - box[0])) // 2, y), line,
                  font=font, fill=(204, 204, 204))
        y += step
    return img.tobytes()


def add_button(pdf, page_id, font_id, name, label, action, rect,
               rendition=None, screen=None):
    x1, y1, x2, y2 = rect
    width, height = x2 - x1, y2 - y1
    from reportlab.pdfbase.pdfmetrics import stringWidth
    size = 10
    tx = (width - stringWidth(label, "Helvetica", size)) / 2
    content = (f"q 0.12 0.12 0.12 rg 0 0 {width} {height} re f "
               f"0.38 0.38 0.38 RG 0.8 w 0.5 0.5 {width - 1} {height - 1} re S "
               f"BT /Helv {size} Tf 0.88 0.88 0.88 rg "
               f"1 0 0 1 {tx:.2f} {(height - size) / 2 + 2:.2f} Tm "
               f"({pdf_escape(label)}) Tj ET Q\n").encode("latin-1")
    ap = pdf.stream(("/Type /XObject /Subtype /Form /FormType 1 "
                     f"/BBox [0 0 {width} {height}] /Resources << /Font << /Helv {font_id} 0 R >> >>"),
                    content)
    action = ("try{" + action + ";}catch(e){if(baStatus)"
              "baStatus.value='BTN ERROR: '+(e.message||e);}")
    if rendition:
        # /R 必须是指向 Rendition 字典的间接引用；直接内联字典 Acrobat 不认。
        # /JS 让 Acrobat 把这个按钮当成「播放媒体 + 自定义 JavaScript」动作，
        # 脚本执行时 event.action.rendition / event.action.annot 才可用，
        # app.media.createPlayer() 也才有媒体上下文。
        act = (f"/A << /S /Rendition /R {rendition} 0 R /OP 0 /AN {screen} 0 R "
               f"/JS ({action}) >>")
    else:
        act = f"/A << /S /JavaScript /JS ({action}) >>"
    return pdf.add(f"""<< /Type /Annot /Subtype /Widget /FT /Btn /T ({name})
/P {page_id} 0 R /Rect [{' '.join(str(v) for v in rect)}] /F 4 /Ff 65536
/H /N /BS << /W 0 >> /AP << /N {ap} 0 R >>
/DA (/Helv 10 Tf 0.88 0.88 0.88 rg)
/MK << /CA ({pdf_escape(label)}) /BG [0.12 0.12 0.12] /BC [0.38 0.38 0.38] >>
{act} >>""")


def add_text_appearance(pdf, font_id, font_name, text, width, height,
                        font_size, bg, fg, tx, ty):
    content = (f"q {bg[0]} {bg[1]} {bg[2]} rg 0 0 {width} {height} re f "
               f"BT /{font_name} {font_size} Tf {fg[0]} {fg[1]} {fg[2]} rg "
               f"1 0 0 1 {tx} {ty} Tm ({pdf_escape(text)}) Tj ET Q\n").encode("latin-1")
    return pdf.stream(("/Type /XObject /Subtype /Form /FormType 1 "
                       f"/BBox [0 0 {width} {height}] "
                       f"/Resources << /Font << /{font_name} {font_id} 0 R >> >>"), content)


def build_pdf(frames, segments, fps, credit_lines, credit_sec, audio_data):
    pdf = PdfBuilder()
    catalog = pdf.reserve()
    pages = pdf.reserve()
    page = pdf.reserve()
    content_id = pdf.reserve()
    courier = pdf.reserve()
    helv = pdf.reserve()
    screen = pdf.reserve()
    credits = pdf.reserve()

    page_content = (f"0 0 0 rg 0 0 {PAGE_W} {PAGE_H} re f\n"
                    f"0.07 0.07 0.07 rg 0 0 {PAGE_W} {CTRL_H} re f\n"
                    f"0.25 0.25 0.25 RG 0.5 w 0 {CTRL_H} m {PAGE_W} {CTRL_H} l S\n")
    pdf.set(content_id, (f"<< /Length {len(page_content)} >>\nstream\n"
                         + page_content + "endstream\nendobj\n"))
    pdf.set(courier, "<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")
    pdf.set(helv, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    screen_font, screen_name = courier, "Cour"

    js = make_js(len(frames), fps, credit_sec).encode("latin-1")
    doc_js = pdf.add(f"<< /S /JavaScript /JS ({pdf_escape(js.decode('latin-1'))}) >>")
    open_action = pdf.add("<< /S /JavaScript /JS (baInit();) >>")

    embedded = pdf.stream("/Type /EmbeddedFile /Subtype /audio#2Fmpeg "
                          f"/Params << /Size {len(audio_data)} >>", audio_data)
    filespec = pdf.add(f"""<< /Type /Filespec /F (badapple.mp3) /UF (badapple.mp3)
/EF << /F {embedded} 0 R /UF {embedded} 0 R >> >>""")
    # Rendition / MediaClip 直写进 Rendition 字典，再用间接引用被 Screen 注释
    # 和 /Names /Renditions 同时引用：/R 按 PDF 规范必须指向间接对象，
    # 内联字典会让 Acrobat 认不出这个媒体注释。
    rendition = (f"<< /S /MR /N (baAudio) "
                 f"/C << /Type /MediaClip /S /MCD /N (baAudio) /CT (audio/mpeg) "
                 f"/D {filespec} 0 R /P << /TF (TEMPACCESS) >> >> >>")
    rendition_ref = pdf.add(rendition)
    pdf.set(screen, f"""<< /Type /Annot /Subtype /Screen /T (baAudio) /NM (baAudio)
/P {page} 0 R /Rect [446 12 452 36] /F 4
/A << /S /Rendition /R {rendition_ref} 0 R /OP 0 /AN {screen} 0 R >> >>""")
    # 同时在文档级 Renditions 名称树里登记一份，
    # 让 JS 的 doc.media.getRendition("baAudio") 在拿不到 event.action 时也能找到。
    # 名称树本身必须是字典（<< /Names [...] >>），直接写数组 Acrobat 不认。
    rendition_names = (f"/Renditions << /Names [(baAudio) {rendition_ref} 0 R] >>")

    row_font, row_lead = row_metrics(0.6)
    row_ids = []
    first = frames[0]
    if len(first) != ascii.ROWS or any(len(line) != ascii.COLS for line in first):
        raise ValueError(f"字符帧尺寸 {len(first)}x{len(first[0])} 与当前网格 "
                         f"{ascii.ROWS}x{ascii.COLS} 不一致")
    # 一行一个只读单行文本域：Acrobat 只对「多行域」做缩小适配，单行域永远按 DA 字号
    # 原样绘制。屏幕区 y 从 CTRL_H 到 PAGE_H，共 SCREEN_H 高，每行 row_lead。
    # 域从 -2 到 968：Acrobat 排版时会留 2pt 内边距，正文正好落在 x=0..960，
    # 最后一列贴着页面右边缘，不会出现空列。
    da = f"/{screen_name} {row_font:g} Tf 0.800 0.800 0.800 rg"
    rect_x0, rect_x1 = -2, PAGE_W + 8
    box_w = rect_x1 - rect_x0
    for index in range(ascii.ROWS):
        row_top = PAGE_H - index * row_lead
        row_bottom = row_top - row_lead
        line = first[index]
        row_ap = add_text_appearance(
            pdf, screen_font, screen_name, line, box_w, row_lead, row_font,
            (0, 0, 0), (0.8, 0.8, 0.8), 2, (row_lead - row_font) / 2)
        row_ids.append(pdf.add(
            f"""<< /Type /Annot /Subtype /Widget /FT /Tx /T (r{index})
/P {page} 0 R /Rect [{rect_x0} {row_bottom:g} {rect_x1:g} {row_top:g}] /F 4 /Ff 1 /Q 0
/DA ({pdf_escape(da)}) /MK << /BC [0 0 0] /BG [0 0 0] >> /BS << /W 0 >>
/AP << /N {row_ap} 0 R >> /V <{pdf_hex(line)}> >>"""))

    buttons = [
        ("replay", "Replay", "baReplay()", (108, 9, 184, 39)),
        ("back", "-5s", "baSeek(-5)", (192, 9, 254, 39)),
        ("forward", "+5s", "baSeek(5)", (262, 9, 324, 39)),
        ("voldown", "Vol-", "baVolumeStep(-10)", (332, 9, 384, 39)),
        ("volup", "Vol+", "baVolumeStep(10)", (392, 9, 444, 39)),
    ]
    # Play/Pause 和 Replay 必须是 Rendition 动作（带自定义 JavaScript）：
    # 只有在这种动作里 event.action.rendition / event.action.annot 才存在，
    # app.media.createPlayer() 也才有媒体上下文。若只挂普通 JavaScript 动作，
    # Acrobat 会返回 null（状态栏 create null r0）并弹「无效自变量」。
    button_ids = [add_button(pdf, page, helv, "toggle", "Play/Pause", "baToggleBtn()",
                            (12, 9, 100, 39), rendition=rendition_ref, screen=screen)]
    button_ids.append(add_button(pdf, page, helv, "replay", "Replay",
                                 "baReplayBtn()", (108, 9, 184, 39),
                                 rendition=rendition_ref, screen=screen))
    button_ids += [add_button(pdf, page, helv, *spec) for spec in buttons[1:]]
    # Guaranteed audio: click AUD to extract the embedded MP3 and play it in
    # the system player (does not depend on Acrobat's legacy media support).
    button_ids.append(add_button(pdf, page, helv, "audio", "AUD", "baExportAudio()",
                                 (900, 9, 948, 39)))
    status_ap = add_text_appearance(pdf, helv, "Helv", "READY - CLICK PLAY", 440, 30,
                                    11, (0.07, 0.07, 0.07), (0.8, 0.8, 0.8), 8, 10)
    status = pdf.add(f"""<< /Type /Annot /Subtype /Widget /FT /Tx /T (status)
/P {page} 0 R /Rect [454 9 896 39] /F 4 /Ff 1 /Q 0
/DA (/Helv 11 Tf 0.800 0.800 0.800 rg)
/MK << /BC [0.2 0.2 0.2] /BG [0.07 0.07 0.07] >> /BS << /W 0 >>
/AP << /N {status_ap} 0 R >> /V <{pdf_hex('READY - CLICK PLAY')}> >>""")

    rgb = credits_image(credit_lines, PAGE_W, SCREEN_H)
    image = pdf.stream(("/Type /XObject /Subtype /Image "
                        f"/Width {PAGE_W} /Height {SCREEN_H} /ColorSpace /DeviceRGB "
                        "/BitsPerComponent 8 /Filter /FlateDecode"), zlib.compress(rgb, 9))
    ap_content = f"q {PAGE_W} 0 0 {SCREEN_H} 0 0 cm /Im0 Do Q\n".encode("latin-1")
    credits_ap = pdf.stream(("/Type /XObject /Subtype /Form /FormType 1 "
                             f"/BBox [0 0 {PAGE_W} {SCREEN_H}] "
                             f"/Resources << /XObject << /Im0 {image} 0 R >> >>"), ap_content)
    pdf.set(credits, f"""<< /Type /Annot /Subtype /Widget /FT /Btn /T (credits)
/P {page} 0 R /Rect [0 {CTRL_H} {PAGE_W} {PAGE_H}] /F 2 /Ff 65536
/BS << /W 0 >> /AP << /N {credits_ap} 0 R >> >>""")

    data_ids = []
    for index, payload in enumerate(segments):
        # 必须是带 widget 的隐藏字段：Acrobat 的 getField() 找不到没有 widget 的纯字段。
        data_ids.append(pdf.add(
            f"""<< /Type /Annot /Subtype /Widget /FT /Tx /T (d{index:04d})
/P {page} 0 R /Rect [0 0 1 1] /F 2 /Ff 1
/DA (/Cour 12 Tf 0 0 0 rg) /V <{pdf_hex(payload)}> >>"""))

    annots = row_ids + [screen] + button_ids + [status, credits] + data_ids
    fields = row_ids + button_ids + [status, credits] + data_ids
    pdf.set(page, f"""<< /Type /Page /Parent {pages} 0 R
/MediaBox [0 0 {PAGE_W} {PAGE_H}] /CropBox [0 0 {PAGE_W} {PAGE_H}] /Rotate 0
/Contents {content_id} 0 R /Resources << /Font << /Cour {courier} 0 R /Helv {helv} 0 R /{screen_name} {screen_font} 0 R >> >>
/Annots [{' '.join(f'{i} 0 R' for i in annots)}]
/AA << /C << /S /JavaScript /JS (baShutdown();) >> >> >>""")
    pdf.set(pages, f"<< /Type /Pages /Count 1 /Kids [{page} 0 R] >>")
    pdf.set(catalog, f"""<< /Type /Catalog /Pages {pages} 0 R
/OpenAction {open_action} 0 R
/Names << /JavaScript << /Names [(badapple.js) {doc_js} 0 R] >> {rendition_names}
/EmbeddedFiles << /Names [(badapple.mp3) {filespec} 0 R] >> >>
/AcroForm << /Fields [{' '.join(f'{i} 0 R' for i in fields)}]
/DA (/{screen_name} {row_font:g} Tf 0.8 0.8 0.8 rg)
/DR << /Font << /Cour {courier} 0 R /Helv {helv} 0 R /{screen_name} {screen_font} 0 R >> >>
/NeedAppearances false >> >>""")
    return pdf.render(catalog), js


def prepare_audio(path, seconds):
    import imageio_ffmpeg
    handle, temp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(handle)
    try:
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
               "-i", path, "-map", "0:a:0", "-t", f"{seconds:.4f}", "-c:a", "copy", temp_path]
        subprocess.run(cmd, check=True)
        with open(temp_path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(description="DoomPDF 式 Bad Apple PDF 播放器")
    parser.add_argument("video")
    parser.add_argument("-o", "--out", default="out/badapple_doom.pdf")
    parser.add_argument("--audio", default="out/audio_cut.mp3")
    parser.add_argument("--duration", type=float, default=218.4167)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--contrast", type=float, default=1.7)
    parser.add_argument("--blur", type=float, default=0.4)
    parser.add_argument("--inset", type=int, default=10)
    parser.add_argument("--credits", default="对照视频《Bad Apple!!》制作|作者 @蒴奈|视频作者：哈纳西飒")
    parser.add_argument("--credit-sec", type=float, default=3.0)
    parser.add_argument("--no-autolevel", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    box = detect_content_box(args.video)
    if box and args.inset > 0:
        box = (box[0] + args.inset, box[1] + args.inset,
               box[2] - args.inset, box[3] - args.inset)
    print(f"裁切框 {box}  网格 {ascii.COLS}x{ascii.ROWS}  对比度 {args.contrast}  模糊 {args.blur}")
    extracted, _, _ = ascii.extract_text(
        args.video, args.duration, args.fps, box, args.contrast,
        not args.no_autolevel, "mono", "ten", args.blur, "stable", smooth=18)
    frames = [lines for lines, _ in extracted]
    if not frames:
        raise RuntimeError("没有提取到正片帧")
    print(f"正片 {len(frames)} 帧 / {len(frames) / args.fps:.4f} 秒")
    segments = encode_segments(frames)
    credit_lines = [s for s in args.credits.split("|") if s]
    audio_seconds = len(frames) / args.fps + (args.credit_sec if credit_lines else 0)
    audio_data = prepare_audio(args.audio, audio_seconds)
    print(f"内嵌 MP3 {len(audio_data) / 1024 / 1024:.2f} MB（{audio_seconds:.4f} 秒裁剪）")
    pdf_data, _ = build_pdf(frames, segments, args.fps, credit_lines,
                            args.credit_sec if credit_lines else 0, audio_data)
    with open(args.out, "wb") as fh:
        fh.write(pdf_data)
    print(f"完成 {args.out}  {len(pdf_data) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
