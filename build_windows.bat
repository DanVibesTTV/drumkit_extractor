@echo off
REM ==========================================================================
REM  Drumkit Extractor - Windows build
REM
REM    build_windows.bat          lite build:  dist\DrumkitExtractor.exe  (~120 MB)
REM                                            dist\drumkit-cli.exe
REM    build_windows.bat full     with Demucs: dist\DrumkitExtractor-full\  (~2 GB)
REM
REM  Needs: Windows 10/11, Python 3.10-3.12 from python.org (with tcl/tk,
REM         the default), internet connection during the build only.
REM ==========================================================================
setlocal
cd /d "%~dp0"

set FULL=0
if /i "%1"=="full" set FULL=1

REM ---- find Python -------------------------------------------------------
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
  echo Python not found. Install it from https://www.python.org/downloads/ and re-run.
  exit /b 1
)

REM ---- virtual environment ----------------------------------------------
if not exist .venv (
  echo [1/4] Creating virtual environment...
  %PY% -m venv .venv || exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
echo [2/4] Installing dependencies...
python -m pip install numpy scipy matplotlib pyinstaller || exit /b 1
if "%FULL%"=="1" (
  python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu || exit /b 1
  python -m pip install demucs soundfile || exit /b 1
)

REM ---- ffmpeg -----------------------------------------------------------
if not exist vendor mkdir vendor
if not exist vendor\ffmpeg.exe (
  echo [3/4] Downloading ffmpeg...
  powershell -NoProfile -Command ^
    "$ErrorActionPreference='Stop';" ^
    "Invoke-WebRequest 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'vendor\ffmpeg.zip';" ^
    "Expand-Archive 'vendor\ffmpeg.zip' 'vendor\ffmpeg_tmp' -Force;" ^
    "Copy-Item (Get-ChildItem 'vendor\ffmpeg_tmp' -Recurse -Filter ffmpeg.exe | Select-Object -First 1).FullName 'vendor\ffmpeg.exe';" ^
    "Remove-Item 'vendor\ffmpeg.zip','vendor\ffmpeg_tmp' -Recurse -Force" || exit /b 1
)

REM ---- Demucs model weights (full build) --------------------------------
if "%FULL%"=="1" if not exist vendor\models (
  echo Downloading Demucs model...
  set TORCH_HOME=%CD%\vendor\models
  python -c "from demucs.pretrained import get_model; get_model('htdemucs')" || exit /b 1
)

REM ---- build ------------------------------------------------------------
echo [4/4] Building executables...
set DRUMKIT_FULL=%FULL%
pyinstaller --noconfirm --clean drumkit.spec || exit /b 1

echo.
if "%FULL%"=="1" (
  echo Done:  dist\DrumkitExtractor-full\DrumkitExtractor.exe
) else (
  echo Done:  dist\DrumkitExtractor.exe   and   dist\drumkit-cli.exe
)
endlocal
