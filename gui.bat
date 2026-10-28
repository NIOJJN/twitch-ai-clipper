@echo off
rem Графический интерфейс: без ввода команд. Окно с настройками и живым логом.
setlocal
cd /d "%~dp0"
where pythonw >nul 2>nul
if %errorlevel%==0 (
  start "" pythonw -m twitch_ai_clipper gui
) else (
  python -m twitch_ai_clipper gui
)
endlocal
