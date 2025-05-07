#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================================${NC}"
echo -e "${CYAN}RAG Application Setup and Launcher${NC}"
echo -e "${CYAN}========================================================${NC}"
echo

# Change to the script's directory to ensure we find docker-compose.yml
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"
echo "Working directory set to: $(pwd)"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Docker daemon is not running.${NC}"
    echo -e "${YELLOW}Please start Docker before continuing.${NC}"
    exit 1
else
    echo -e "${GREEN}Docker daemon is running.${NC}"
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${RED}Ollama is not running.${NC}"
    echo -e "${YELLOW}Please start Ollama before continuing.${NC}"
    exit 1
else
    echo -e "${GREEN}Ollama is running.${NC}"
fi

# Check if gemma3 model is available in Ollama
if ! curl -s http://localhost:11434/api/tags | grep -q "gemma3"; then
    echo -e "${RED}The 'gemma3' model is not available in Ollama.${NC}"
    echo -e "${YELLOW}Please run 'ollama pull gemma3:1b' before continuing.${NC}"
    exit 1
else
    echo -e "${GREEN}The 'gemma3' model is available in Ollama.${NC}"
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}docker-compose.yml file not found in the current directory.${NC}"
    echo -e "${YELLOW}Expected at: $(pwd)/docker-compose.yml${NC}"
    exit 1
else
    echo -e "${GREEN}Found docker-compose.yml file.${NC}"
fi

# Stop any existing containers
echo "Stopping any existing containers..."
docker compose down

# Build and start containers
echo "Building and starting containers (this may take a few minutes on first run)..."
if ! docker compose up --build -d; then
    echo -e "${RED}Failed to start the application containers.${NC}"
    echo "Please check the error messages above for details."
    exit 1
fi

echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}       Application is now running successfully!      ${NC}"
echo -e "${GREEN}=====================================================${NC}"
echo
echo -e "Access the application at: ${CYAN}http://localhost:3000${NC}"
echo
echo "The following services are available:"
echo -e " - Frontend: ${CYAN}http://localhost:3000${NC}"
echo -e " - Backend API: ${CYAN}http://localhost:5000/api${NC}"
echo " - MongoDB: localhost:27017"
echo " - Qdrant: localhost:6333"
echo
echo -e "${YELLOW}To stop the application, run:${NC} docker compose down" 