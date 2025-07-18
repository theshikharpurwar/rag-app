@echo off
echo ===============================================
echo   RAG Application - Multimodal Setup (Local)
echo ===============================================
echo.

REM Get the directory of the batch file
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Step 1: Checking Docker...
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)
echo ✓ Docker is running

echo.
echo Step 2: Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Ollama is not running. Please start Ollama.
    pause
    exit /b 1
)
echo ✓ Ollama is running

echo.
echo Step 3: Starting core services (MongoDB, Qdrant)...
docker-compose up -d mongo qdrant
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to start core services.
    pause
    exit /b 1
)

echo.
echo Step 4: Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo.
echo Step 5: Setting up Python environment...
cd python

echo Setting up local Nomic models (no API key required)...
echo ✓ Using local models - completely offline after download!

echo.
echo Installing Python dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install Python dependencies.
    pause
    exit /b 1
)

echo.
echo Setting up multimodal collections...
python setup_multimodal.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to setup multimodal collections.
    pause
    exit /b 1
)

echo.
echo Testing multimodal setup...
python test_multimodal.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Multimodal test failed. Check your setup.
) else (
    echo ✓ Multimodal setup successful!
)

cd ..

echo.
echo Step 6: Starting full application...
docker-compose up -d
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to start application.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo   RAG Application Started Successfully!
echo ===============================================
echo.
echo Access your application:
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:5000
echo   Qdrant:   http://localhost:6333
echo.
echo Multimodal Features Available (Local Models):
echo   Process PDFs: python python\compute_embeddings_multimodal.py "file.pdf" --pdf_id "id"
echo   Query:        python python\local_llm_multimodal.py "question" --pdf_id "id"
echo.
echo Use 'docker-compose down' to stop the application.
echo.
pause
