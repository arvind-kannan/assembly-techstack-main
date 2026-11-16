@echo off
rem Starts the DASG HMI from its own folder (config.ini paths are relative to the repo root).
rem Activates the repo .venv (uv sync) when it exists.
if exist "%~dp0..\.venv\Scripts\activate.bat" call "%~dp0..\.venv\Scripts\activate.bat"
python "%~dp0dasg_hmi.py"
pause
