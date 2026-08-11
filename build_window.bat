@echo off
setlocal EnableExtensions

rem ResponseLab Windows x64 build entry.
rem Run this file from Explorer or a cmd.exe window.
rem It creates a local venv, verifies the source, and produces dist\ResponseLab\ResponseLab.exe.

cd /d "%~dp0"
if errorlevel 1 goto :fail

rem A valid existing project venv is self-contained; only bootstrap Python when creating it.
if exist ".venv\Scripts\python.exe" goto :validate_venv

rem Prefer a valid x64 Python already selected on PATH (including actions/setup-python).
rem Fall back to the Windows Python Launcher when PATH has no suitable interpreter.
set "BOOTSTRAP_PYTHON="
where python >nul 2>&1
if errorlevel 1 goto :try_python_launcher
python -c "import struct, sys, sysconfig; raise SystemExit(0 if sys.version_info >= (3, 11) and struct.calcsize('P') == 8 and sysconfig.get_platform().lower() == 'win-amd64' else 1)" >nul 2>&1
if errorlevel 1 goto :try_python_launcher
set "BOOTSTRAP_PYTHON=python"
goto :bootstrap_python_ready

:try_python_launcher
where py >nul 2>&1
if errorlevel 1 goto :bootstrap_python_unavailable
py -3 -c "import struct, sys, sysconfig; raise SystemExit(0 if sys.version_info >= (3, 11) and struct.calcsize('P') == 8 and sysconfig.get_platform().lower() == 'win-amd64' else 1)" >nul 2>&1
if errorlevel 1 goto :bootstrap_python_unavailable
set "BOOTSTRAP_PYTHON=py -3"
goto :bootstrap_python_ready

:bootstrap_python_unavailable
echo ERROR: Windows x64 Python 3.11 or newer is required. Install x64 Python, then run again.
goto :fail

:bootstrap_python_ready
%BOOTSTRAP_PYTHON% -c "import struct, sys, sysconfig; raise SystemExit(0 if sys.version_info >= (3, 11) and struct.calcsize('P') == 8 and sysconfig.get_platform().lower() == 'win-amd64' else 1)"
if errorlevel 1 (
    echo ERROR: Windows x64 Python 3.11 or newer is required. Install x64 Python, then run again.
    goto :fail
)

echo Creating project virtual environment...
%BOOTSTRAP_PYTHON% -m venv .venv
if errorlevel 1 goto :fail

:validate_venv
set "PYTHON=%CD%\.venv\Scripts\python.exe"

rem Re-check the interpreter actually used for testing/building; reject x86, ARM64, and versions <3.11.
"%PYTHON%" -c "import struct, sys, sysconfig; raise SystemExit(0 if sys.version_info >= (3, 11) and struct.calcsize('P') == 8 and sysconfig.get_platform().lower() == 'win-amd64' else 1)"
if errorlevel 1 (
    echo ERROR: Project .venv must use Windows x64 Python 3.11 or newer.
    echo Delete .venv, install x64 Python, and run build_window.bat again.
    goto :fail
)

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
