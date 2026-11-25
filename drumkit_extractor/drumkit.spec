# PyInstaller spec for Drumkit Extractor.
# Run via build_windows.bat (or: pyinstaller drumkit.spec).
#
#   lite build (default):  two single-file .exe, ffmpeg bundled, HPSS separation
#   full build (DRUMKIT_FULL=1): adds Demucs + PyTorch + model weights,
#                                built as a folder (a single 2 GB .exe would
#                                have to unpack itself on every start)
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

FULL = os.environ.get("DRUMKIT_FULL") == "1"
HERE = Path(SPECPATH)

datas, binaries, hidden = [], [], []

ffmpeg = HERE / "vendor" / "ffmpeg.exe"
if not ffmpeg.is_file():
    raise SystemExit("vendor/ffmpeg.exe missing - run build_windows.bat, it downloads it.")
binaries.append((str(ffmpeg), "."))

excludes = ["torch", "torchaudio", "demucs"] if not FULL else []
if FULL:
    for pkg in ("demucs", "dora", "julius", "openunmix", "torchaudio", "lameenc", "soundfile"):
        try:
            d, b, h = collect_all(pkg)
            datas += d; binaries += b; hidden += h
        except Exception as e:
            print(f"(skipping {pkg}: {e})")
    models = HERE / "vendor" / "models"
    if models.is_dir():
        datas.append((str(models), "models"))
    else:
        print("WARNING: no pre-downloaded Demucs model - it will download on first use.")

a = Analysis(
    ["drumkit_extractor.py"],
    pathex=[str(HERE)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden + ["scipy.signal", "scipy.ndimage", "matplotlib.backends.backend_agg"],
    excludes=excludes + ["PyQt5", "PyQt6", "PySide2", "PySide6", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if not FULL:
    # Double-click app (no console window)
    gui = EXE(pyz, a.scripts, a.binaries, a.datas, [],
              name="DrumkitExtractor", console=False, upx=False)
    # Command-line version (shows progress in the terminal)
    cli = EXE(pyz, a.scripts, a.binaries, a.datas, [],
              name="drumkit-cli", console=True, upx=False)
else:
    gui = EXE(pyz, a.scripts, [], exclude_binaries=True,
              name="DrumkitExtractor", console=False, upx=False)
    cli = EXE(pyz, a.scripts, [], exclude_binaries=True,
              name="drumkit-cli", console=True, upx=False)
    COLLECT(gui, cli, a.binaries, a.datas, name="DrumkitExtractor-full", upx=False)
