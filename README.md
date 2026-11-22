# Bad Apple PDF 播放器

> **使用前注意：请先打开终端，进入本文件所在的 `BadApplePDF/` 文件夹，再执行下面的安装和转换命令。不要在 `checkpoints/` 文件夹中运行 `pip`。**

把《Bad Apple!!》做成纯 PDF 播放器的开源项目。

视频演示：待补充（发布后在此填写 B 站演示链接）

> 打包好的 EXE 不放进代码仓库，请到 [Releases](https://github.com/hanaxisa/pdf-maker/releases) 页面下载。

## 功能

本项目提供三种把视频转成 PDF 的方式：

| 模块 | 说明 |
|------|------|
| `ascii.py` | 灰度 → ASCII 字符密度，逐帧一页 PDF（经典方案） |
| `v2.py` | 1-bit 位图帧 → PDF 播放器，支持 JS 定时翻页 / Dur 自动放映两种引擎 |
| `contour/` + `pdf/` | **大模型方案**：SAM2 自动识别前景轮廓 + 提取灰度，默认生成带音频和播放控制的单页 ASCII PDF 播放器，也支持静态 PDF |

## 大模型方案（contour + pdf）

核心思路：用 **SAM2（Segment Anything Model 2）** 对视频逐帧做前景分割，
模型自己判断哪些像素属于角色（前景），并保留前景像素的原始灰度值，
再把灰度映射为 ASCII 字符帧，生成支持播放、暂停、进度跳转和音量调节的单页 PDF 播放器。
也可以选择静态 PDF 模板，输出 ASCII、轮廓线或灰度填充页面。

### 处理流程

```
视频 ──► SAM2 分割 ──► 前景掩码 + 灰度 ──► .npz ──► ASCII 字符帧 + 音频 ──► 单页 PDF 播放器
         (大模型)        (模型产出)                  (可复用数据)
```

### 安装

**一键初始化（推荐）**

在 `BadApplePDF/` 目录下运行：

```powershell
python init.py
```

脚本会自动完成：

- 选择本机依赖最完整的 Python（例如 3.12），不合适时自动切换解释器并重启自身
- 安装 `requirements.txt` 中缺失的依赖（默认清华镜像，失败自动回退官方 PyPI）
- 从 `hf-mirror.com` 下载所选 SAM2 权重到 `checkpoints/`（默认 tiny）
- 创建 `dist/input/{待分离,静视频,纯音频,表情包}` 和 `output/` 目录
- 写入 `.vscode/settings.json` 的 `python.defaultInterpreterPath`（已有复杂配置时只提示、不覆盖）
- 最后自动运行 `selfcheck.py` 验证环境

常用参数：

| 参数 | 说明 |
|------|------|
| `--check` | 只检查，不安装、不下载、不写配置 |
| `--model small` | 预下载指定模型：`tiny` / `small` / `base_plus` / `large` |
| `--mirror aliyun` | 切换 pip 镜像，可选 `tsinghua` / `aliyun` / `none` |
| `--interpreter "路径"` | 强制使用指定 Python 解释器 |
| `--skip-model` | 跳过模型下载 |
| `--no-config` | 不写编辑器解释器配置 |

**手动安装**

打开终端后，**先切换到项目根目录**，也就是 `requirements.txt` 所在的 `BadApplePDF/` 文件夹。不要在 `C:\Users\...` 或 `checkpoints/` 目录中执行安装。

请根据终端类型选择下面**其中一组**命令，并将路径替换成你实际存放项目的位置。

**PowerShell（提示符通常以 `PS` 开头）：**

```powershell
Set-Location -LiteralPath "e:\新建文件夹\bad apple\BadApplePDF"
python -m pip install -r requirements.txt
```

**Windows CMD（提示符通常类似 `C:\Users\...>`）：**

```bat
cd /d "e:\新建文件夹\bad apple\BadApplePDF"
python -m pip install -r requirements.txt
```

注意：`/d` 只能用于 CMD，不能在 PowerShell 中使用。

**注意：`pip install` 只安装 Python 依赖，不会下载模型。** 安装完成后，`checkpoints/` 中只有 `.gitkeep` 占位文件是正常的。

模型会在首次运行 `python convert.py` 并开始视频分割时，自动从 `hf-mirror.com` 下载到 `checkpoints/`。下载期间请保持网络连接，不要提前关闭终端。

**编辑器里运行报 `ModuleNotFoundError` 怎么办？** 如果终端里 `python -c "import cv2"` 正常，但点编辑器右上角运行却报找不到模块，说明编辑器选中的 Python 不是装依赖的那个。按 `Ctrl+Shift+P` → `Python: Select Interpreter`，选择与终端一致的 3.12 解释器（例如 `C:\Users\<用户名>\AppData\Local\Programs\Python\Python312\python.exe`），再重新运行。也可以直接运行 `python selfcheck.py`，它会显示当前解释器并给出可切换的解释器路径。

也可以手动下载权重，并将 `.pt` 文件直接放在 `checkpoints/` 根目录中，不要再嵌套一层文件夹：

- tiny:  `sam2_hiera_tiny.pt`
- small: `sam2_hiera_small.pt`
- base_plus: `sam2_hiera_base_plus.pt`
- large: `sam2_hiera_large.pt`

### 按 BV 号下载与音视频分离（可选）

Python 脚本与 EXE 共用同一个工作目录 `dist/input/`，所有输出都使用源视频的主文件名，方便对应查找和批量保存：

| 目录 | 用途 |
|------|------|
| `dist/input/待分离/` | 放入手动收集的视频；下载脚本也把含音轨的原视频保存在这里 |
| `dist/input/静视频/` | 保存去掉音轨的视频，画面无损复制，保留原视频扩展名 |
| `dist/input/纯音频/` | 保存从原视频提取并转码的同名 MP3 |

**按 BV 号下载（默认只下载原视频）：**

```powershell
python download_bilibili.py BV1AbCdEfGhI
python download_bilibili.py "https://www.bilibili.com/video/BV1AbCdEfGhI/"
```

上面的 BV 号仅演示格式，请替换成实际 BV 号。也可以一次传入多个 BV 号进行批量下载。以下载 `BV1AbCdEfGhI` 为例，默认输出只有一个文件：

```text
dist/input/待分离/BV1AbCdEfGhI.mp4   # 原视频（画面 + 音轨）
```

转换脚本可以直接读取原视频里的音轨，所以默认不需要分离。确实需要静视频或纯音频时再加 `--split`：

```powershell
python download_bilibili.py BV1AbCdEfGhI --split
```

```text
dist/input/待分离/BV1AbCdEfGhI.mp4   # 原视频
dist/input/静视频/BV1AbCdEfGhI.mp4   # 静视频（仅画面）
dist/input/纯音频/BV1AbCdEfGhI.mp3   # 纯音频
```

已下载的原视频会直接复用；默认跳过已存在的分离结果。需要重新分离时添加 `--split --overwrite`。仅支持无需登录即可公开访问的视频，不处理付费或受限内容。

**只分离本地视频：**

1. 把视频放入 `dist/input/待分离/`。
2. 执行批量分离：

```powershell
python separate_media.py
```

也可以只处理一个或多个指定文件：

```powershell
python separate_media.py "dist/input/待分离/demo.mp4"
python separate_media.py demo.mp4 other.mkv
python separate_media.py --overwrite
```

### EXE 怎么用

`dist/` 中提供七个程序：

| 程序 | 作用 | 是否需要 Python |
|------|------|------------------|
| `OneClick.exe` | 一键转换：硬件扫描 → 选档 → 自动分割采集 + 生成 PDF | 是，自动选择 Python 3.9+ |
| `MediaSeparator.exe` | 把视频拆成「静视频 + 纯音频」 | 否，内置 FFmpeg |
| `BilibiliDownloader.exe` | 按 BV 号下载 B 站视频（默认只存原视频，`--split` 可分离） | 否，内置 FFmpeg |
| `InitEnv.exe` | 一键初始化：装依赖、下模型、建目录、自检 | 是，自动选择 Python 3.9+ |
| `PDF生成工具.exe` | 选视频 / npz、低或高色差、音轨、片尾字幕，直接生成播放器 PDF | 否 |
| `PDF铺满工具.exe` | 把 PDF 页面铺满到指定纸张大小 | 否 |
| `gif转mp4.exe` | GIF 表情包转 MP4 | 否 |

MediaSeparator、BilibiliDownloader、PDF生成工具、PDF铺满工具、gif转mp4 是独立单文件程序；InitEnv 和 OneClick 是启动器，会自动找到依赖最完整的解释器，再把参数原样交给 `init.py` / `analyze.py`。

MediaSeparator 和 BilibiliDownloader 都遵循同一条规则：在 **EXE 自己所在的目录**下创建并使用下面三个文件夹，输出文件与源视频同名，方便一一对应。

```text
dist/
├── MediaSeparator.exe
├── BilibiliDownloader.exe
├── InitEnv.exe
├── OneClick.exe
├── PDF生成工具.exe
├── PDF铺满工具.exe
├── gif转mp4.exe
└── input/
    ├── 待分离/   # 你放进去的原始视频；下载器也把下载的原视频存这里
    ├── 静视频/   # 输出：去音轨的视频，画面直接复制，扩展名不变
    └── 纯音频/   # 输出：同名 MP3（192kbps）
```

#### 用法零：OneClick.exe（一键转换）

最核心的入口。把视频放进 `dist/input/待分离/`，双击 `OneClick.exe`，程序会依次完成：

**1. 硬件扫描**

自动检测并显示：
- CPU 型号与线程数
- 内存总量与当前可用量
- 显卡型号、显存总量与已用、当前负载（%）、驱动版本
- 磁盘剩余空间
- 已有的 SAM2 权重（tiny / base_plus / large）

**2. 视频分析**

读取视频的分辨率、帧率、时长，并采样计算平均亮度、黑边范围、音轨音量，输出推荐的画面裁剪参数。

**3. 三档选档**

根据硬件自动给出 low / mid / high 三档（或更少，见降级规则），每档显示：

| 档位 | 模型 | 分辨率 | ASCII 网格 | 关键帧间隔 | 采样点 | 帧率 | 设备 |
|------|------|--------|-----------|-----------|--------|------|------|
| 低档 | tiny | 128px | 120×45 | 16 | 6 | 10 | CPU |
| 中档 | tiny | 192px | 120×45 | 10 | 8 | 12 | CPU |
| 高档(CUDA) | large | 512px | 168×63 | 4 | 8 | 15 | GPU |
| 高档(CPU) | base_plus | 384px | 144×54 | 6 | 8 | 14 | CPU |

> ASCII 网格越大细节越丰富、字体越小。高档 168×63 比中档 120×45 多约 96% 的字符单元，能捕捉更细的线条和小细节；同时默认采用 `autolevel=False` + `bg_threshold=35` + `contrast=5.0` 的干净管线，消除背景频闪和边缘光晕。

每档还会显示：
- **预计用时**：区间形式（如「约 5–8 分钟」），基于模型大小、分辨率、采样点数、设备、CPU 核心数估算
- **死机概率**：基于内存占用比，分 1–3%（安全）、2–5%、5–12%、12–25%、25–45%、45–70% 六档
- **缺失权重提示**：若该档所需的 `.pt` 不在 `checkpoints/`，会标注「需下载」

**4. 降级规则**

- 高档会根据显存/内存自动降级（large→base_plus→更小分辨率）；
- 若高档画质仍不如中档，则只显示「低档 + 高档」两档；
- 若高档连低档画质都达不到，则拒绝转换并提示硬件不足；
- 检测到 GPU 负载 ≥50%（如正在玩游戏），高档用时会乘以 8 倍粗估，并标注「GPU 被占用」。

**5. 转换**

输入档位编号后，程序自动拼装参数并调用 `convert.py` 运行 SAM2 采集数据、生成 PDF，完成后自动打开 `output/`。全程不需要输入任何路径。

**命令行参数**（适用于 OneClick.exe 和 `python analyze.py`）：

```bat
OneClick.exe --tier high                  直接按高档转换，跳过交互选档
OneClick.exe --no-convert                 只输出硬件扫描与分析结果，不启动转换
OneClick.exe dist\input\待分离\demo.mp4   指定视频
OneClick.exe --seconds 60                 只采样前 60 秒画面（分析更快）
OneClick.exe --data output\demo_data.npz  额外检查已有分割数据的前景占比
```

> 提示：低/中档固定在 CPU 上运行，不受 GPU 被占用影响，适合先出预览；高档使用 GPU，速度快但需要显存。

#### 用法一：MediaSeparator.exe（分离本地视频）

1. 双击 `MediaSeparator.exe`。第一次运行且文件夹为空时，它只会建好三个目录并提示「没有待分离视频」，这是正常的。
2. 把要处理的视频（mp4 / mkv / mov / avi / flv / wmv / webm / m4v）复制到 `dist/input/待分离/`。
3. 再次双击 `MediaSeparator.exe`，它会依次处理该目录下的**全部视频**，跑完后按回车关闭窗口。
4. 结果分别在 `dist/input/静视频/` 和 `dist/input/纯音频/`。

也可以把单个视频文件直接**拖到 exe 图标上**处理；连续拖入多少个就处理多少个。注意这种方式窗口不会停下来等你按回车。

已存在的输出默认跳过，想重新生成时用命令行加 `--overwrite`：

```bat
MediaSeparator.exe --overwrite
MediaSeparator.exe input\待分离\demo.mp4 --overwrite
```

#### 用法二：BilibiliDownloader.exe（按 BV 号下载）

1. 双击 `BilibiliDownloader.exe`。
2. 按提示输入 BV 号或完整视频链接，例如 `BV1AbCdEfGhI` 或 `https://www.bilibili.com/video/BV1AbCdEfGhI/`；一次要下多个就用空格隔开。
3. 回车后下载原视频，默认只生成一个文件：

```text
dist/input/待分离/BV1AbCdEfGhI.mp4   # 原视频（画面 + 音轨）
```

需要静视频或纯音频时加 `--split`：

```text
dist/input/待分离/BV1AbCdEfGhI.mp4   # 原视频
dist/input/静视频/BV1AbCdEfGhI.mp4   # 静视频（仅画面）
dist/input/纯音频/BV1AbCdEfGhI.mp3   # 纯音频
```

直接在终端里带参数运行，可免去交互输入，适合批量：

```bat
BilibiliDownloader.exe BV1AbCdEfGhI BV2AbCdEfGhI
BilibiliDownloader.exe "https://www.bilibili.com/video/BV1AbCdEfGhI/" --split --overwrite
```

#### 用法三：InitEnv.exe（一键初始化）

双击 `InitEnv.exe`，自动完成依赖安装、模型下载、目录创建和自检。需要本机已装 Python 3.9+；启动器会优先选择依赖最全的解释器（例如 3.12）。

```bat
InitEnv.exe --check
InitEnv.exe --model small --mirror aliyun
```

参数与 `python init.py` 相同，详见上面的“一键初始化”章节。

#### 常见问题

- **画面被压窄/拉长、左右有大黑边**：播放器现在会先从分割数据里自动裁掉原视频黑边，再按字符单元 1:2 的真实比例等比缩放。旧 PDF 不会自动变化，需要用 `python convert.py --reuse-data` 重新生成。
- **没有声音，状态栏显示 `ERR:open failPlayerSecurityPrompt`**：这是 Acrobat 的多媒体安全提示没能弹出导致的（旧版播放器主动关闭了提示）。新版已恢复提示，首次点 Play 时在弹窗里选“允许”即可。若仍失败，打开 Acrobat 首选项 → 多媒体/多媒体（旧版），勾选“允许多媒体操作”，并把 PDF 所在目录加入信任位置。
- **偶尔闪屏**：新版已恢复媒体安全提示、给文本行显式设置黑底、状态栏只在内容变化时刷新，减少整屏闪白。旧 PDF 需要重新生成。
- **只支持公开视频**：需要登录、付费、会员或其它权限限制的内容无法下载，会报失败并继续处理下一个。
- **重复运行不会重复下载**：已存在的原视频直接复用，已存在的静视频/音频默认跳过；要重做加 `--overwrite`。
- **目录位置固定**：程序只认自己旁边的 `input/`，把视频放在 `C:\Users\...` 或别处不会被自动发现（拖拽或命令行传路径除外）。
- **不要只复制 exe 到别处后又期望找到旧输出**：输出始终跟着 exe 所在目录走。
- **InitEnv.exe / OneClick.exe 找不到脚本**：它们会从自身目录向上查找含 `requirements.txt` 和 `init.py` 的项目根目录，请把 exe 留在 `dist/` 或项目内。若提示未找到 Python，请先安装 Python 3.9+。

重新打包（PyInstaller）：

```powershell
python -m PyInstaller --noconfirm --onefile --console --name InitEnv --distpath dist --workpath build\init --specpath build exe_launcher.py
python -m PyInstaller --noconfirm --onefile --console --name OneClick --distpath dist --workpath build\oneclick --specpath build exe_launcher.py
python -m PyInstaller --noconfirm --onefile --console --name MediaSeparator --distpath dist --workpath build\mediaseparator --specpath build separate_media.py
python -m PyInstaller --noconfirm --onefile --console --name BilibiliDownloader --distpath dist --workpath build\bilibili --specpath build download_bilibili.py
python -m PyInstaller --noconfirm --onefile --windowed --name "PDF生成工具" --distpath dist --workpath build\pdfgen --specpath build tools\pdf_gen_tool.py
python -m PyInstaller --noconfirm --onefile --windowed --name "PDF铺满工具" --distpath dist --workpath build\filler --specpath build tools\pdf_filler.py
python -m PyInstaller --noconfirm --onefile --windowed --name "gif转mp4" --distpath dist --workpath build\picker --specpath build tools\gif_picker.py
```

### 一键转换（推荐）

最省事的方式是双击 `dist/OneClick.exe`，按提示回车即可。

也可以手动操作：

1. 把视频放入 `dist/input/待分离/` 文件夹。
2. 运行：

```bash
python convert.py
```

3. 按中文提示选择视频、处理范围和输出类型。默认生成单页 PDF 播放器，也可选择逐帧静态 PDF，或“只采集数据”仅生成 `.npz`。生成的文件和中间数据保存在 `output/`。

转换开始前会自动打印一段分析结果（分辨率、黑边、平均亮度、音轨音量），加 `--no-analyze` 可跳过。

也可以直接使用命令行：

```bash
# 只转换前 10 秒为单页交互播放器，适合先测试效果
python convert.py dist/input/demo.mp4 --test

# 转换完整视频，默认生成带播放控制和内嵌音频的单页 ASCII PDF 播放器
python convert.py dist/input/demo.mp4

# 只分析并采集分割数据，不生成 PDF；之后用 --reuse-data 合成
python convert.py dist/input/demo.mp4 --collect

# 使用独立音频文件；不指定时读取输入视频音轨，无音轨则生成等长静音
python convert.py dist/input/demo.mp4 --audio dist/input/bgm.mp3

# 改为生成绿色矢量轮廓静态 PDF
python convert.py dist/input/demo.mp4 --static --mode contour --theme green

# 复用已有分割数据，生成静态灰度填充 PDF，无需重新运行 SAM2
python convert.py dist/input/demo.mp4 --reuse-data --static --mode filled

# 升级旧数据：从已存的灰度+mask 重新生成 ASCII 网格（cover 模式+应用 mask），无需重跑 SAM2
python convert.py dist/input/demo.mp4 --reuse-data --refit-grid
```

播放器请使用 **Adobe Acrobat** 打开，并按提示启用 JavaScript 和多媒体；浏览器内置 PDF 阅读器无法完整播放。只有添加 `--static` 才会生成逐帧一页的静态 PDF。

### 硬件扫描、选档与自检

`analyze.py` 是 OneClick.exe 的核心，也可手动运行：

```powershell
# 硬件扫描 + 视频分析 + 三档选档 + 转换（交互）
python analyze.py

# 只扫描和分析，不启动转换
python analyze.py --no-convert

# 直接按指定档位转换，跳过交互
python analyze.py --tier high

# 只采样前 60 秒画面，分析更快
python analyze.py dist/input/待分离/demo.mp4 --seconds 60

# 已有分割数据时，一并检查前景占比是否正常
python analyze.py --data output/demo_data.npz
```

`analyze.py` 会打印硬件信息、推荐档位及每档的预计用时和死机概率，并在选档后自动调用 `convert.py`。

环境出问题时跑自检：

```powershell
# 环境、依赖、模型、目录、视频音频、输出全面自检
python selfcheck.py

# 自动创建缺失目录 / 额外检查 CUDA
python selfcheck.py --fix
python selfcheck.py --deep

# 出问题时把报错内容交给它诊断原因
python selfcheck.py --error "ModuleNotFoundError: No module named 'sam2'"
python selfcheck.py --log error.txt
```

`selfcheck.py` 用 `[PASS]` / `[WARN]` / `[FAIL]` 标注结果，并对问题给出处理命令。`init.py` 末尾会自动运行一次自检。

### 分步使用（进阶）

**第一步：用 SAM2 分割视频**

```bash
# 关键帧模式（推荐，CPU 上可接受）：每 10 帧用 SAM2，其余用 Otsu
python -m contour.sam_segmenter dist/input/demo.mp4 -o out/segmented.npz \
    --duration 0 --fps 12 --model tiny --max-dim 192 \
    --keyframe-interval 10 --points-per-side 8

# 全帧 SAM2（精度最高但慢，建议 GPU）
python -m contour.sam_segmenter dist/input/demo.mp4 -o out/segmented.npz \
    --keyframe-interval 0
```

**第二步：生成静态 PDF（可选）**

```bash
# ASCII 模式（与现有视频风格一致）
python -m pdf.contour_pdf out/segmented.npz -o out/badapple.pdf --mode ascii

# 轮廓线模式（矢量线条）
python -m pdf.contour_pdf out/segmented.npz -o out/badapple.pdf --mode contour

# 灰度填充模式（色块）
python -m pdf.contour_pdf out/segmented.npz -o out/badapple.pdf --mode filled
```

### PDF 渲染模式说明

| 模式 | 效果 | 适用场景 |
|------|------|----------|
| `ascii` | 前景灰度映射为 8 级字符密度表 ` .:-=+*#` | 复刻经典 Bad Apple ASCII 风格 |
| `contour` | OpenCV `findContours` 提取轮廓，矢量线条绘制 | 线条艺术风格 |
| `filled` | 按前景灰度填充半透明色块 | 灰度图风格 |

## 经典方案（ascii.py）

```bash
python ascii.py dist/input/demo.mp4 -o out/badapple_ascii.pdf \
    --theme mono --ramp ten --fps 12 --bgmode stable
```

主题可选：`mono`（黑白）、`green`（绿屏）、`amber`（琥珀）、`blue`（蓝屏）、`paper`（白纸黑字）

## 播放器方案（v2.py）

```bash
# 两种引擎都生成
python v2.py dist/input/demo.mp4 -m both --grid 160xauto

# 仅 JS 单页播放器
python v2.py dist/input/demo.mp4 -m js

# 仅多页 /Dur 自动放映
python v2.py dist/input/demo.mp4 -m dur
```

## 项目结构

```
BadApplePDF/
├── init.py                 # 一键初始化：解释器、依赖、模型、目录、编辑器配置
├── exe_launcher.py         # InitEnv.exe / OneClick.exe 的启动器
├── convert.py              # 转换引擎（被 analyze.py 调用，也可独立运行）
├── analyze.py              # 硬件扫描 + 三档选档 + ETA/死机概率 + 调用 convert.py
├── selfcheck.py            # 环境与素材自检（由 init.py 末尾调用）
├── separate_media.py       # 本地视频批量/单独分离音视频（MediaSeparator.exe）
├── download_bilibili.py    # 按 BV 号下载视频（默认只存原视频）
├── dist/                   # EXE 打包产物 + 统一工作目录
│   ├── MediaSeparator.exe
│   ├── BilibiliDownloader.exe
│   ├── InitEnv.exe
│   ├── OneClick.exe
│   ├── PDF铺满工具.exe
│   ├── gif转mp4.exe
│   └── input/
│       ├── 待分离/         # 待处理视频及下载的原视频
│       ├── 静视频/         # 分离后的无音轨视频
│       └── 纯音频/         # 分离后的同名 MP3
├── build/                  # PyInstaller 构建缓存（可删除）
├── output/                 # 一键转换的 PDF 和分割数据输出到这里
├── checkpoints/            # SAM2 权重目录（tiny / base_plus / large）
├── ascii.py                # 经典 ASCII 方案
├── v2.py                   # PDF 播放器（JS / Dur）
├── make_pdf_direct.py      # Doom 风格单页播放器
├── contour/
│   ├── __init__.py
│   └── sam_segmenter.py    # SAM2 前景分割 + 灰度提取
├── pdf/
│   ├── __init__.py
│   ├── player_pdf.py       # SAM2 数据 → 单页交互 PDF 播放器
│   └── contour_pdf.py      # 静态 PDF 模板（ascii / contour / filled）
├── requirements.txt
├── LICENSE
└── README.md
```

## 性能说明

SAM2 单帧处理时间取决于模型、分辨率和设备。OneClick.exe 会在选档前给出每档的预计用时区间，实际时间与估算误差通常在 ±28% 以内。

- **低档**（tiny/128px/CPU）：几分钟内完成，占用最低，适合快速预览
- **中档**（tiny/192px/CPU）：约 10–30 分钟
- **高档**（large/512px/CUDA）：GPU 上数分钟完成，画质最高

无独显时高档回退到 CPU 的 base_plus/384px，用时会显著增加。GPU 被游戏或渲染占用（负载 ≥50%）时，高档用时会按 8 倍粗估，建议关闭占用程序后再跑高档。

## 致谢

- 对照视频《Bad Apple!!》
- 视频作者：哈纳西飒
- SAM2: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)

## License

MIT
