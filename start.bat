@echo off
echo Starting the RAG Application...

REM Get the directory of the batch file
set "SCRIPT_DIR=%~dp0"
echo Current directory: %SCRIPT_DIR%
cd /d "%SCRIPT_DIR%"
echo Working directory set to: %CD%

echo.
echo Step 1: Checking if Docker is running...
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Docker does not appear to be running.
    echo Please start Docker Desktop before continuing.
    pause
    exit /b 1
)
echo Docker is running.

echo.
echo Step 2: Checking if Ollama is running...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Ollama does not appear to be running.
    echo Please start Ollama before continuing.
    pause
    exit /b 1
)
echo Ollama is running.

echo.
echo Step 3: Checking for tinyllama model...
curl -s http://localhost:11434/api/tags | findstr "tinyllama" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: The tinyllama model is not available in Ollama.
    echo Please run "ollama pull tinyllama" before continuing.
    pause
    exit /b 1
)
echo The tinyllama model is available.

echo.
echo Step 4: Checking for docker-compose.yml file...
if not exist "docker-compose.yml" (
    echo.
    echo ERROR: docker-compose.yml file not found in the current directory.
    echo Expected at: %CD%\docker-compose.yml
    pause
    exit /b 1
)
echo Found docker-compose.yml file.

echo.
echo Step 5: Stopping any existing containers...
docker compose down

echo.
echo Step 6: Building and starting the application (this may take a few minutes)...
docker compose up --build -d

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Failed to start the application.
    echo Please check the error messages above for details.
    pause
    exit /b 1
)

echo.
echo Application is now running!
echo.
echo Access the web interface at: http://localhost:3000
echo.
pause 