@echo off
echo Testing Docker builds locally...

echo.
echo Building backend image...
docker build -t rag-app-backend:test -f backend/Dockerfile .

if %ERRORLEVEL% NEQ 0 (
  echo Backend build failed!
  exit /b %ERRORLEVEL%
)

echo.
echo Building frontend image...
docker build -t rag-app-frontend:test -f frontend/Dockerfile frontend

if %ERRORLEVEL% NEQ 0 (
  echo Frontend build failed!
  exit /b %ERRORLEVEL%
)

echo.
echo Both images built successfully!
echo.
echo To test with Docker Compose, run:
echo docker-compose -f docker-compose.test.yml up -d
echo.
echo To check running containers:
echo docker ps
echo.
echo To view logs:
echo docker logs rag-app-frontend-1
echo docker logs rag-app-backend-1 