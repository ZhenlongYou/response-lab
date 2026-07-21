@echo off
setlocal EnableExtensions

rem ResponseLab Windows x64 build entry.
rem Run this file from Explorer or a cmd.exe window.
rem It creates a local venv, verifies the source, and produces dist\ResponseLab\ResponseLab.exe.

cd /d "%~dp0"
if errorlevel 1 goto :fail

rem Prefer the Python Launcher with 3.11; fall back to python when py is unavailable.
where py >nul 2>&1
if errorlevel 1 (
    set "BOOTSTRAP_PYTHON=python"
) else (
    set "BOOTSTRAP_PYTHON=py -3.11"
)

%BOOTSTRAP_PYTHON% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.11 or newer is required. Install 64-bit Python, then run again.
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating project virtual environment...
    %BOOTSTRAP_PYTHON% -m venv .venv
    if errorlevel 1 goto :fail
)

set "PYTHON=%CD%\.venv\Scripts\python.exe"

echo Installing build dependencies...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%PYTHON%" -m pip install -e ".[dev]"
if errorlevel 1 goto :fail
"%PYTHON%" -m pip install "pyinstaller>=6.0"
if errorlevel 1 goto :fail

echo Running source verification...
"%PYTHON%" -m pytest -q
if errorlevel 1 goto :fail
"%PYTHON%" -m ruff check .
if errorlevel 1 goto :fail
"%PYTHON%" -m compileall -q main.py src tests examples
if errorlevel 1 goto :fail
"%PYTHON%" main.py --self-test
if errorlevel 1 goto :fail
"%PYTHON%" main.py --gui-smoke-test
if errorlevel 1 goto :fail

if exist "dist\ResponseLab" (
    echo Existing dist\ResponseLab will be replaced by PyInstaller.
)

echo Building Windows EXE...
"%PYTHON%" -m PyInstaller ^
  --noconfirm --clean --windowed --onedir ^
  --name ResponseLab ^
  --paths src ^
  --add-data "src\response_lab\assets;response_lab\assets" ^
  --collect-all pyqtgraph ^
  --collect-all scipy ^
  main.py
if errorlevel 1 goto :fail

if not exist "dist\ResponseLab\ResponseLab.exe" (
    echo ERROR: PyInstaller completed without dist\ResponseLab\ResponseLab.exe.
    goto :fail
)

echo.
echo SUCCESS: dist\ResponseLab\ResponseLab.exe
echo Deliver the entire dist\ResponseLab folder, not only the EXE file.
echo Before release, complete the clean-machine and Keysight AG10 read-back checks in docs\WINDOWS_EXE_BUILD_HANDOFF.md.
exit /b 0

:fail
echo.
echo BUILD FAILED. Read the first ERROR above; do not ship the EXE until it is resolved.
exit /b 1
