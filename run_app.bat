@echo off
cd /d %~dp0
call tts_env\Scripts\activate.bat
python ui.py
pause