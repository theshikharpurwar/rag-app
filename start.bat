@echo off
echo Starting the RAG Application...

REM Path to Docker
set "DOCKER=C:\Program Files\Docker\Docker\resources\bin\docker"

REM Path to Ollama
set "OLLAMA=C:\Users\Shikhar Purwar\AppData\Local\Programs\Ollama\ollama.exe"

echo.
echo Step 1: Stopping any existing containers...
"%DOCKER%" compose down

echo.
echo Step 2: Building and starting the application (this may take a few minutes)...
"%DOCKER%" compose up --build -d

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Failed to start the application.
    echo Please make sure Docker Desktop is running.
    pause
    exit /b 1
)

echo.
echo Application is now running!
echo.
echo Access the web interface at: http://localhost:3000
echo.
pause 