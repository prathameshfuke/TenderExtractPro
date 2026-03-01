@echo off
echo Starting TenderExtractPro...
echo.
echo [1/2] Starting FastAPI backend on port 8000...
start "TenderExtractPro API" cmd /k "cd /d D:\TenderExtractPro && venv\Scripts\activate && venv\Scripts\uvicorn.exe api.main:app --reload --port 8000 --log-level info"
timeout /t 4 /nobreak > nul

echo [2/2] Starting React frontend on port 5173...
start "TenderExtractPro UI" cmd /k "cd /d D:\TenderExtractPro\frontend && npm run dev"
timeout /t 2 /nobreak > nul

echo.
echo  ==========================================
echo   TenderExtractPro is starting up...
echo  ==========================================
echo   API  : http://localhost:8000
echo   UI   : http://localhost:5173
echo   Docs : http://localhost:8000/docs
echo  ==========================================
echo.
echo  Wait ~30s for the embedding model to load on first run.
echo  (Subsequent runs are faster due to HuggingFace cache.)
echo.
