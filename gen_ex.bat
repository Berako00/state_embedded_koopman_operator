@echo off
@echo off
pip install pyinstaller
set "DESKTOP=%USERPROFILE%\Desktop"
pyinstaller --onefile --windowed --distpath "%DESKTOP%" GUI_main.py

