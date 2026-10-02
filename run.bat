@echo off
rem Запуск без «ручных» телодвижений: %* — аргументы CLI.
rem Примеры:
rem   run.bat run twitch.tv/tarik --whisper-model medium
rem   run.bat run 123456789 --limit-seconds 900 --max-clips 5 --vertical
rem   run.bat serve --port 8765
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"
python -m twitch_ai_clipper %*
endlocal
