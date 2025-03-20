@echo off
echo Checking for Python installation...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python and try again.
    pause
    exit /b
)

echo Python is installed.

:: Set up environment variables
set "ENV_DIR=%CD%\venv"
set "DIST_DIR=%CD%\dist"
set "DESKTOP=%USERPROFILE%\Desktop"

:: Check if virtual environment exists, if not, create it
if not exist "%ENV_DIR%" (
    echo Creating a new virtual environment...
    python -m venv "%ENV_DIR%"
)

:: Activate the virtual environment
call "%ENV_DIR%\Scripts\activate"

echo Installing required packages...
pip install --upgrade pip
pip install numpy==1.23.0 tensorflow pyinstaller

echo Generating executable...
pyinstaller --onefile --windowed --distpath "%DIST_DIR%" GUI_main.py

echo Build completed! Your .exe is in the dist/ folder.

:: Create a shortcut on Desktop
set "SHORTCUT=%DESKTOP%\GUI_main.lnk"
set "EXE_PATH=%DIST_DIR%\GUI_main.exe"

powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT%'); $s.TargetPath='%EXE_PATH%'; $s.Save()"

echo Shortcut created on Desktop!

:: Deactivate virtual environment
deactivate

echo Done! Press any key to exit.
pause

