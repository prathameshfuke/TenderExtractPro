@echo off
echo Starting TenderExtractPro...
start "FastAPI Backend" cmd /k "cd /d D:\TenderExtractPro && venv\Scripts\activate && uvicorn api.main:app --reload --port 8000"
timeout /t 3 /nobreak > nul
start "React Frontend" cmd /k "cd /d D:\TenderExtractPro\frontend && npm run dev"
echo.
echo  FastAPI : http://localhost:8000
echo  React   : http://localhost:5173
echo.
