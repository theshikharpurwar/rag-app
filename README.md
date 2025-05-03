# RAG Application

A Retrieval Augmented Generation (RAG) application that lets you ask questions about your PDF documents. This application uses:
- MongoDB for document storage
- Qdrant for vector search 
- Ollama for local LLM processing (running AI models on your own computer)

## Quick Start

### Windows Users:

1. **Ensure Prerequisites are Met:**
   - Install [Docker Desktop](https://www.docker.com/products/docker-desktop) and make sure it is **running**.
   - Install [Ollama](https://ollama.com/download) and make sure it is **running**.
   - Download the required model: `ollama pull tinyllama`

2. **Run the Start Script:**
   - Double-click on `start.bat`.
   - This script uses hardcoded paths based on your system (`C:\Program Files\Docker...` and `C:\Users\Shikhar Purwar...`). If Docker or Ollama are installed elsewhere, you may need to edit the script.

### Linux/Mac Users:

1. **Ensure Prerequisites are Met:**
   - Install Docker (Docker Desktop for Mac, Docker Engine for Linux).
   - Install Docker Compose (usually included with Docker Desktop, may need separate install on Linux).
   - Make sure the Docker service is **running**.
   - Install [Ollama](https://ollama.com/download).
   - Make sure Ollama is **running**.
   - Download the required model: `ollama pull tinyllama`

2. **Run the Start Script:**
   - Open your terminal.
   - Navigate to the project directory.
   - Make the script executable: `chmod +x start.sh`
   - Run the script: `./start.sh`

**The scripts will then build and start the application containers.**

## Production Deployment with GitHub Container Registry

For production deployment using pre-built containers from GitHub Container Registry (GHCR):

1. **Log in to GitHub Container Registry**:
   ```bash
   echo $GITHUB_TOKEN | docker login ghcr.io -u YOURUSERNAME --password-stdin
   ```

2. **Deploy using Docker Compose**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

For detailed deployment instructions, see the [DEPLOYMENT.md](DEPLOYMENT.md) guide.

> **Note:** The CI/CD workflow is configured to build and publish images when pushing to the `dev-ops` branch.

## Detailed Setup (Manual Method)

If you prefer to set things up manually *after* installing prerequisites:

1. **Install Prerequisites** (Docker, Ollama - see above)
2. **Download AI Model**: `ollama pull tinyllama`
3. **Start Application**: From the project directory, run `docker-compose up --build -d`

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
- Make sure Docker Desktop/Engine is running
- Try restarting Docker if you encounter issues
- On Windows, ensure Docker Desktop is using WSL2 for better performance
- On Linux, ensure your user is in the docker group: `sudo usermod -aG docker $USER` (then log out/in)

### Ollama Issues
- Verify Ollama is running
- If you get errors about the model not being found, try running `ollama pull tinyllama` again

### Application Issues
- If the application doesn't start correctly, try running `docker-compose down` followed by `docker-compose up --build -d`
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