@echo off
rem Runs both paper-trading bots for the rest of today's session on this PC.
cd /d "%~dp0"
"C:\Users\korad\StockAnalyzerData\venv\Scripts\python.exe" -m backend.bot.run_day >> "C:\Users\korad\StockAnalyzerData\bot\console.log" 2>&1
