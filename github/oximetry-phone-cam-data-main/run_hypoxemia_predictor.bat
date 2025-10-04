@echo off
cd /d "%~dp0"
echo Starting Hypoxemia Prediction Software...
echo.
python hypoxemia_predictor.py
echo.
echo Press any key to exit...
pause >nul
