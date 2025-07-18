@echo off
echo Starting Ollama in CPU-only mode...

REM Set environment variables to force CPU usage
set OLLAMA_NUM_GPU=0
set OLLAMA_HOST=0.0.0.0:11434

REM Start Ollama server
ollama serve

pause
