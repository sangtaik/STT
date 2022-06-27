# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata

datas = [('symspell_jamo_dict.txt', '.'), ('Assets/vocab.json', 'Assets'), ('Assets/vocab_jamos.json', 'Assets'), ('Assets/test_data.wav', 'Assets'), ('Assets/vocab_chars.json', 'Assets')]
datas += collect_data_files('torch')
datas += collect_data_files('tensorflow')
datas += collect_data_files('transformers')
datas += collect_data_files('librosa')
datas += copy_metadata('torch')
datas += copy_metadata('tqdm')
datas += copy_metadata('tensorflow')
datas += copy_metadata('transformers')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('importlib_metadata')
datas += copy_metadata('librosa')


block_cipher = None


a = Analysis(
    ['main_run.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['pytorch', 'tensorflow', 'transformers', 'sklearn.utils._cython_blas', 'sklearn.utils._typedefs', 'sklearn.neighbors._partition_nodes', 'scipy.special.cython_special'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main_run',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main_run',
)
