# RAG Application

A Retrieval Augmented Generation (RAG) application that lets you ask questions about your PDF documents. This application uses:
- MongoDB for document storage
- Qdrant for vector search 
- Ollama for local LLM processing (running AI models on your own computer)

## Quick Start

### Windows Users:

Simply run the `start.bat` file by double-clicking on it. This script will:
1. Check if Docker Desktop is installed and running
2. Check if Ollama is installed and running
3. Install required models if needed
4. Start all the necessary services

### Linux/Mac Users:

1. Make the start script executable:
   ```bash
   chmod +x start.sh
   ```

2. Run the script:
   ```bash
   ./start.sh
   ```

**Both scripts will help you download and install any missing components.**

## Detailed Setup (Manual Method)

If you prefer to set things up manually, follow these steps:

### 1. Install Prerequisites

1. **Docker Desktop** (or Docker Engine on Linux)
   - Download and install from [Docker's website](https://www.docker.com/products/docker-desktop)
   - Make sure Docker is running

2. **Ollama**
   - Download and install from [Ollama's website](https://ollama.com/download)
   - Start Ollama after installation

### 2. Download Required AI Model

Run this command in your terminal or command prompt:

```bash
ollama pull tinyllama
```

This downloads a small but capable AI model called TinyLlama that will run on your computer.

### 3. Start the Application

From the project directory, run:

```bash
docker-compose up -d
```

Wait for all services to start (this might take several minutes on the first run).

## Using the Application

1. Open your web browser and go to: http://localhost:3000
2. Upload a PDF document using the interface
3. After the document is processed, you can ask questions about its content
4. The AI will provide answers based on the information in your document

## Understanding the Parts

- **Frontend**: The web interface you interact with (http://localhost:3000)
- **Backend**: Processes your documents and questions (http://localhost:5000)
- **MongoDB**: Stores document metadata
- **Qdrant**: Stores document vectors for semantic search
- **Ollama**: Runs the AI model on your local machine

## Stopping the Application

To stop all services, run:

```bash
docker-compose down
```

## Troubleshooting

### Docker Issues
- Make sure Docker Desktop is running
- Try restarting Docker Desktop if you encounter issues
- On Windows, ensure Docker Desktop is using WSL2 for better performance
- On Linux, ensure your user is in the docker group: `sudo usermod -aG docker $USER`

### Ollama Issues
- Verify Ollama is running
- If you get errors about the model not being found, try running `ollama pull tinyllama` again

### Application Issues
- If the application doesn't start correctly, try running `docker-compose down` followed by `docker-compose up -d`
- Check the logs with `docker-compose logs`

### Not Enough Memory
- The application requires at least 8GB of RAM to run properly
- Close other memory-intensive applications before starting

### Help and Support
If you encounter issues not covered here, please check the project's GitHub issues section or open a new issue.

## Architecture

- **Frontend**: React application served by Nginx
- **Backend**: Node.js/Express with Python integration
- **Vector Store**: Qdrant for embedding storage and similarity search
- **Database**: MongoDB for document metadata
- **LLM**: Ollama running on your host machine (outside Docker)

### API Proxy

The frontend Nginx configuration includes a proxy that forwards all `/api` requests to the backend service. This allows the frontend to make relative API calls without hardcoding the backend URL.

## Persistent Storage

The following data is persisted through Docker volumes:
- `mongo_data`: MongoDB database files
- `qdrant_data`: Qdrant vector database
- `uploads_data`: Uploaded PDF files and extracted images

## Development

To make changes to the application:

1. Stop the Docker containers: `docker-compose down`
2. Make your changes
3. Rebuild the containers: `docker-compose up --build -d` 