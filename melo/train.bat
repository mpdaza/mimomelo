@echo off
setlocal

rem Run one training session with the example configuration.
set "SCRIPT_DIR=%~dp0"
set "CONFIG=%SCRIPT_DIR%data\example\config.json"
set "MODEL_NAME=example"

cd /d "%SCRIPT_DIR%"
python "%SCRIPT_DIR%train.py" -c "%CONFIG%" -m "%MODEL_NAME%"
exit /b %ERRORLEVEL%
