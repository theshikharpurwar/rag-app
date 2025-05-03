@echo off
setlocal EnableDelayedExpansion

echo ========================================================
echo RAG Application Setup and Launcher
echo ========================================================
echo.

:: Set colors for better UI
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "MAGENTA=[95m"
set "CYAN=[96m"
set "RESET=[0m"

:: Check if running as administrator
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %RED%This script needs to be run as Administrator for some operations.%RESET%
    echo Please right-click on this file and select "Run as administrator".
    echo.
    pause
    exit /b 1
)

echo %CYAN%Step 1: Checking for Docker Desktop installation...%RESET%
echo.

:: Check if Docker is installed
where docker >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %RED%Docker is not installed or not in PATH.%RESET%
    echo.
    echo %YELLOW%Would you like to download Docker Desktop?%RESET% (Y/N)
    set /p INSTALL_DOCKER=
    if /i "!INSTALL_DOCKER!"=="Y" (
        echo.
        echo %CYAN%Opening Docker Desktop download page...%RESET%
        start "" "https://www.docker.com/products/docker-desktop"
        echo.
        echo Please install Docker Desktop, then restart your computer.
        echo After installation, run this script again.
        pause
        exit /b 1
    ) else (
        echo.
        echo %YELLOW%Docker is required to run this application.%RESET%
        echo Please install Docker Desktop manually and run this script again.
        pause
        exit /b 1
    )
)

echo %GREEN%Docker is installed.%RESET%
echo.

echo %CYAN%Step 2: Checking if Docker Desktop is running...%RESET%
echo.

:: Try to run a simple Docker command to check if Docker is available
docker ps >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %RED%Docker Desktop does not appear to be running.%RESET%
    echo.
    echo %YELLOW%Would you like to start Docker Desktop now?%RESET% (Y/N)
    set /p START_DOCKER=
    if /i "!START_DOCKER!"=="Y" (
        echo.
        echo %CYAN%Starting Docker Desktop...%RESET%
        start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        echo Waiting for Docker to start (this may take a minute)...
        
        set MAX_RETRIES=30
        set RETRY_COUNT=0
        
        :DOCKER_START_WAIT
        timeout /t 2 >nul
        set /a RETRY_COUNT+=1
        echo Checking Docker status... Attempt !RETRY_COUNT! of %MAX_RETRIES%
        
        docker ps >nul 2>&1
        if %ERRORLEVEL% NEQ 0 (
            if !RETRY_COUNT! LSS %MAX_RETRIES% (
                goto DOCKER_START_WAIT
            ) else (
                echo.
                echo %RED%Docker Desktop did not start successfully within the expected time.%RESET%
                echo Please start Docker Desktop manually, then run this script again.
                pause
                exit /b 1
            )
        ) else (
            echo.
            echo %GREEN%Docker Desktop started successfully!%RESET%
        )
    ) else (
        echo.
        echo %YELLOW%Please start Docker Desktop manually, then run this script again.%RESET%
        pause
        exit /b 1
    )
) else (
    echo %GREEN%Docker Desktop is running.%RESET%
)
echo.

echo %CYAN%Step 3: Checking for Ollama installation...%RESET%
echo.

:: Check if Ollama is installed
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %RED%Ollama is not installed or not in PATH.%RESET%
    echo.
    echo %YELLOW%Would you like to download Ollama?%RESET% (Y/N)
    set /p INSTALL_OLLAMA=
    if /i "!INSTALL_OLLAMA!"=="Y" (
        echo.
        echo %CYAN%Opening Ollama download page...%RESET%
        start "" "https://ollama.com/download"
        echo.
        echo Please install Ollama, then run this script again.
        pause
        exit /b 1
    ) else (
        echo.
        echo %YELLOW%Ollama is required to run this application.%RESET%
        echo Please install Ollama manually and run this script again.
        pause
        exit /b 1
    )
) else (
    echo %GREEN%Ollama is installed.%RESET%
)
echo.

echo %CYAN%Step 4: Checking if Ollama is running...%RESET%
echo.

:: Check if Ollama is running by attempting to list models
ollama list >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %RED%Ollama is not running.%RESET%
    echo.
    echo %YELLOW%Would you like to start Ollama now?%RESET% (Y/N)
    set /p START_OLLAMA=
    if /i "!START_OLLAMA!"=="Y" (
        echo.
        echo %CYAN%Starting Ollama...%RESET%
        start "" "%LOCALAPPDATA%\Ollama\ollama.exe"
        echo Waiting for Ollama to start (this may take a minute)...
        
        set MAX_RETRIES=15
        set RETRY_COUNT=0
        
        :OLLAMA_START_WAIT
        timeout /t 2 >nul
        set /a RETRY_COUNT+=1
        echo Checking Ollama status... Attempt !RETRY_COUNT! of %MAX_RETRIES%
        
        ollama list >nul 2>&1
        if %ERRORLEVEL% NEQ 0 (
            if !RETRY_COUNT! LSS %MAX_RETRIES% (
                goto OLLAMA_START_WAIT
            ) else (
                echo.
                echo %RED%Ollama did not start successfully within the expected time.%RESET%
                echo Please start Ollama manually, then run this script again.
                pause
                exit /b 1
            )
        ) else (
            echo.
            echo %GREEN%Ollama started successfully!%RESET%
        )
    ) else (
        echo.
        echo %YELLOW%Please start Ollama manually, then run this script again.%RESET%
        pause
        exit /b 1
    )
) else (
    echo %GREEN%Ollama is running.%RESET%
)
echo.

echo %CYAN%Step 5: Checking for required model (tinyllama)...%RESET%
echo.

:: Check if the required model is available
ollama list | findstr "tinyllama" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %YELLOW%The required model 'tinyllama' is not found.%RESET%
    echo.
    echo %CYAN%Pulling the model now... (This might take a while depending on your internet speed)%RESET%
    ollama pull tinyllama
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo %RED%Failed to pull the model.%RESET%
        echo Please check your internet connection and try again.
        pause
        exit /b 1
    ) else (
        echo.
        echo %GREEN%Model 'tinyllama' has been successfully downloaded!%RESET%
    )
) else (
    echo %GREEN%Model 'tinyllama' is already installed.%RESET%
)
echo.

echo %CYAN%Step 6: Building and starting the application containers...%RESET%
echo.

:: Stop any existing containers
echo Stopping any existing containers...
docker-compose down

:: Build and start containers
echo Building and starting containers (this may take a few minutes on first run)...
docker-compose up --build -d

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo %RED%Failed to start the application containers.%RESET%
    echo Please check the error messages above for details.
    pause
    exit /b 1
)

echo.
echo %GREEN%=====================================================%RESET%
echo %GREEN%       Application is now running successfully!      %RESET%
echo %GREEN%=====================================================%RESET%
echo.
echo Access the application at: %CYAN%http://localhost:3000%RESET%
echo.
echo The following services are available:
echo  - Frontend: %CYAN%http://localhost:3000%RESET%
echo  - Backend API: %CYAN%http://localhost:5000/api%RESET%
echo  - MongoDB: localhost:27017
echo  - Qdrant: localhost:6333
echo.
echo %YELLOW%To stop the application, run:%RESET% docker-compose down
echo.
echo Press any key to exit...
pause >nul 