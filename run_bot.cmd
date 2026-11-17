@echo off
rem Starts the stocks-in-play paper-trading bot for one trading day.
cd /d "%~dp0"
"C:\Users\korad\StockAnalyzerData\venv\Scripts\python.exe" -m backend.bot.inplay_bot >> "C:\Users\korad\StockAnalyzerData\bot\console.log" 2>&1
