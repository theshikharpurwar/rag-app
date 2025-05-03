#!/bin/bash

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================================${NC}"
echo -e "${BLUE}RAG Application Setup and Launcher${NC}"
echo -e "${BLUE}========================================================${NC}"
echo

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo -e "${CYAN}Step 1: Checking for Docker installation...${NC}"
echo

if ! command_exists docker; then
    echo -e "${RED}Docker is not installed or not in PATH.${NC}"
    echo
    echo -e "${YELLOW}Docker is required to run this application.${NC}"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    echo "After installation, run this script again."
    exit 1
else
    echo -e "${GREEN}Docker is installed.${NC}"
fi
echo

# Check if Docker is running
echo -e "${CYAN}Step 2: Checking if Docker daemon is running...${NC}"
echo

if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Docker daemon is not running.${NC}"
    echo
    echo -e "${YELLOW}Please start Docker before continuing.${NC}"
    echo "On Linux: sudo systemctl start docker"
    echo "On Mac: Start Docker Desktop from Applications"
    exit 1
else
    echo -e "${GREEN}Docker daemon is running.${NC}"
fi
echo

# Check if docker-compose is installed
echo -e "${CYAN}Step 3: Checking for Docker Compose...${NC}"
echo

if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
    echo -e "${RED}Docker Compose is not installed.${NC}"
    echo
    echo -e "${YELLOW}Docker Compose is required to run this application.${NC}"
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    echo "After installation, run this script again."
    exit 1
else
    echo -e "${GREEN}Docker Compose is installed.${NC}"
fi
echo

# Check if Ollama is installed
echo -e "${CYAN}Step 4: Checking for Ollama installation...${NC}"
echo

if ! command_exists ollama; then
    echo -e "${RED}Ollama is not installed or not in PATH.${NC}"
    echo
    echo -e "${YELLOW}Ollama is required to run this application.${NC}"
    echo "Please install Ollama from https://ollama.com/download"
    echo "After installation, run this script again."
    exit 1
else
    echo -e "${GREEN}Ollama is installed.${NC}"
fi
echo

# Check if Ollama service is running
echo -e "${CYAN}Step 5: Checking if Ollama is running...${NC}"
echo

# Try to connect to Ollama's API
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${RED}Ollama is not running.${NC}"
    echo
    echo -e "${YELLOW}Please start Ollama before continuing.${NC}"
    echo "If you just installed Ollama, you might need to start it manually."
    echo "After starting Ollama, run this script again."
    exit 1
else
    echo -e "${GREEN}Ollama is running.${NC}"
fi
echo

# Check if the required model is available
echo -e "${CYAN}Step 6: Checking for required model (tinyllama)...${NC}"
echo

if ! ollama list | grep -q "tinyllama"; then
    echo -e "${YELLOW}The required model 'tinyllama' is not found.${NC}"
    echo
    echo -e "${CYAN}Pulling the model now... (This might take a while)${NC}"
    if ! ollama pull tinyllama; then
        echo
        echo -e "${RED}Failed to pull the model.${NC}"
        echo "Please check your internet connection and try again."
        exit 1
    else
        echo
        echo -e "${GREEN}Model 'tinyllama' has been successfully downloaded!${NC}"
    fi
else
    echo -e "${GREEN}Model 'tinyllama' is already installed.${NC}"
fi
echo

# Start the application
echo -e "${CYAN}Step 7: Building and starting the application containers...${NC}"
echo

# Stop any existing containers
echo "Stopping any existing containers..."
docker-compose down

# Build and start containers
echo "Building and starting containers (this may take a few minutes on first run)..."
if ! docker-compose up --build -d; then
    echo
    echo -e "${RED}Failed to start the application containers.${NC}"
    echo "Please check the error messages above for details."
    exit 1
fi

echo
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
echo -e "${YELLOW}To stop the application, run:${NC} docker-compose down" 