# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['../tools/pdf_gen_tool.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 本工具只读取 npz 复用数据，不跑 SAM2 推理；排除训练/推理依赖可把体积降到百兆级
    excludes=['torch', 'torchvision', 'torchaudio', 'sam2', 'hydra', 'omegaconf',
              'onnxruntime', 'tensorboard', 'triton'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PDF生成工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
