@echo off
title Aimbot Server Installer
echo ================================
echo  Aimbot Server - Auto Installer
echo ================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Downloading installer...
    curl -o python_installer.exe https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo Installing Python (this may take a minute)...
    python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del python_installer.exe
    echo Python installed.
)

:: Manually add Python to PATH for this session (in case PATH wasn't refreshed)
set PATH=%PATH%;C:\Program Files\Python311;C:\Program Files\Python311\Scripts
set PATH=%PATH%;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311
set PATH=%PATH%;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\Scripts

echo.
echo Installing required packages (this may take several minutes)...
python -m pip install --upgrade pip
python -m pip install ultralytics torch opencv-python numpy

echo.
echo ================================
echo  Installation complete!
echo  Your IP addresses:
ipconfig | findstr "IPv4"
echo.
echo  Tell the gaming PC to use one
echo  of the IPs shown above.
echo ================================
echo.
pause
