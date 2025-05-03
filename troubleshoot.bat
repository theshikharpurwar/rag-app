@echo off
echo Starting troubleshooting script...
echo.

echo 1. Checking for Docker...
echo.
where docker
if %ERRORLEVEL% EQU 0 (
    echo Docker command found in PATH
) else (
    echo Docker command NOT found in PATH
)
echo.

echo 2. Checking Docker installation paths...
echo.
if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
    echo Docker Desktop found at C:\Program Files\Docker\Docker\Docker Desktop.exe
) else (
    echo Docker Desktop NOT found at C:\Program Files\Docker\Docker\Docker Desktop.exe
)

if exist "C:\Program Files\Docker\Docker\resources\bin\docker.exe" (
    echo Docker executable found at C:\Program Files\Docker\Docker\resources\bin\docker.exe
) else (
    echo Docker executable NOT found at C:\Program Files\Docker\Docker\resources\bin\docker.exe
)
echo.

echo 3. Checking for Ollama...
echo.
where ollama
if %ERRORLEVEL% EQU 0 (
    echo Ollama command found in PATH
) else (
    echo Ollama command NOT found in PATH
)
echo.

echo 4. Checking Ollama installation paths...
echo.
if exist "%LOCALAPPDATA%\Ollama\ollama.exe" (
    echo Ollama found at %LOCALAPPDATA%\Ollama\ollama.exe
) else (
    echo Ollama NOT found at %LOCALAPPDATA%\Ollama\ollama.exe
)

if exist "C:\Program Files\Ollama\ollama.exe" (
    echo Ollama found at C:\Program Files\Ollama\ollama.exe
) else (
    echo Ollama NOT found at C:\Program Files\Ollama\ollama.exe
)
echo.

echo 5. Testing Docker...
echo.
docker --version
echo.
docker ps
echo.

echo 6. Testing Ollama...
echo.
if exist "%LOCALAPPDATA%\Ollama\ollama.exe" (
    "%LOCALAPPDATA%\Ollama\ollama.exe" --version
) else (
    ollama --version
)
echo.

echo Troubleshooting completed.
echo Please provide this output to assist with debugging.
pause 