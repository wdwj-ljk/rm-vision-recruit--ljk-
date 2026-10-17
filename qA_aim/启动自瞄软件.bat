@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
"C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe" auto_aim.py --mode menu
if errorlevel 1 (
  echo.
  echo [ERROR] Program failed. See messages above. Run diagnose.bat for details.
)
pause
