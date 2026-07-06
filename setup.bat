@echo off
echo ============================================================
echo  KanoonSaathi — Demo Setup
echo ============================================================

echo.
echo [1/4] Installing Python backend dependencies...
cd backend
pip install -r requirements.txt

echo.
echo [2/4] Downloading spaCy English model...
python -m spacy download en_core_web_sm

echo.
echo [3/4] Creating required directories...
if not exist "storage\documents" mkdir storage\documents
if not exist "storage\faiss_index" mkdir storage\faiss_index
if not exist "data" mkdir data

echo.
echo [4/4] Installing frontend dependencies...
cd ..\frontend
call npm install

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To start:
echo    Backend:  cd backend && python -m uvicorn main:app --reload
echo    Frontend: cd frontend && npm run dev
echo.
echo  Then open: http://localhost:5173
echo ============================================================
pause
