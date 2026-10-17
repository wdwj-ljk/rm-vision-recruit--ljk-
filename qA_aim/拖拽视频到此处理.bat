@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if "%~1"=="" goto usage
"C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe" auto_aim.py --mode drop --video "%~1"
if errorlevel 1 (
  echo.
  echo [ERROR] Program failed. See messages above. Run diagnose.bat for details.
)
pause
exit /b

:usage
echo.
echo Usage - drag an mp4 video file onto this bat file.
echo The program detects red/blue by filename automatically,
echo then asks how to process it.
echo.
pause
