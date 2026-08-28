@echo off
cd /d "%~dp0"
python scripts/load_to_sqlite.py
pause
