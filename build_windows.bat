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

REM ---- find a supported Python (3.10 - 3.12) -----------------------------
REM  Newer versions (3.13+) lack ready-made packages for PyTorch / Demucs'
REM  dependencies, so pip would try to compile them from source and fail.
REM  PYTHON_EXE can be set to force a specific interpreter (used on GitHub).
set PY=
if defined PYTHON_EXE set PY="%PYTHON_EXE%"
if not defined PY for %%V in (3.12 3.11 3.10) do (
  if not defined PY py -%%V -c "pass" >nul 2>nul && set PY=py -%%V
)
if not defined PY (
  echo No Python 3.10 - 3.12 found. Install Python 3.12 from
  echo https://www.python.org/downloads/ and re-run this script.
  exit /b 1
)
%PY% -c "import sys; v=sys.version_info[:2]; print('Using Python %%d.%%d' %% v); sys.exit(0 if (3,10)<=v<=(3,12) else 1)" || (
  echo This Python version is not supported - please use Python 3.10 - 3.12.
  exit /b 1
)

REM ---- virtual environment ----------------------------------------------
REM  Recreate it if it was made with an unsupported Python earlier.
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" || (
    echo Removing old virtual environment with unsupported Python...
    rmdir /s /q .venv
  )
)
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
