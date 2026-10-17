@echo off
title Camera Calibration GUI
cd /d "%~dp0"

rem ---- choose Python interpreter ----
set "PY=python"
if exist "C:\Users\Lenovo\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" set "PY=C:\Users\Lenovo\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if exist "%~dp0pylibs" set "PYTHONPATH=%~dp0pylibs"

rem ---- check dependencies ----
"%PY%" -c "import cv2, tkinter" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Missing dependencies for: %PY%
  echo Install with: pip install opencv-python numpy pillow
  pause
  exit /b 1
)

"%PY%" "%~dp0gui_app.py"
if errorlevel 1 (
  echo.
  echo Program exited with an error. See message above.
  pause
)
