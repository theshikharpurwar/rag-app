# Running the RAG Application with Docker

This guide provides step-by-step instructions for running the RAG application using Docker.

## Prerequisites

- [Docker](https://www.docker.com/get-started/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed
- [Ollama](https://ollama.com/) running locally (for LLM support)

## Option 1: Using Pre-built Container Images (Recommended)

The easiest way to run the application is using the pre-built images from GitHub Container Registry.

### Step 1: Download the Docker Compose file

```bash
# Replace OWNER/REPO with the actual repository name (e.g., username/rag-app)
# This uses the 'working' branch which is the current production branch
curl -O https://raw.githubusercontent.com/OWNER/REPO/working/docker-compose.ghcr.yml
```

Or create a new file named `docker-compose.ghcr.yml` with the following content (replace `OWNER/REPO` with the actual repository name):

```yaml
version: '3.8'

services:
  frontend:
    image: ghcr.io/OWNER/REPO-frontend:latest
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped

  backend:
    image: ghcr.io/OWNER/REPO-backend:latest
    ports:
      - "5000:5000"
    volumes:
      - uploads_data:/app/uploads
    environment:
      - MONGODB_URI=mongodb://mongo:27017/rag_db
      - OLLAMA_HOST_URL=http://host.docker.internal:11434
      - LLM_MODEL=gemma3:1b
      - NODE_ENV=production
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - mongo
      - qdrant
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"

  mongo:
    image: mongo:latest
    ports:
      - "27018:27017"
    volumes:
      - mongo_data:/data/db
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

volumes:
  qdrant_data:
  mongo_data:
  uploads_data:
```

### Step 2: Make sure Ollama is running

The RAG application requires Ollama to be running locally with the `gemma3:1b` model:

```bash
# Start Ollama if not already running
ollama serve

# In a separate terminal, pull the required model
ollama pull gemma3:1b
```

### Step 3: Start the application

```bash
docker compose -f docker-compose.ghcr.yml up -d
```

### Step 4: Access the application

- Frontend UI: http://localhost:3000
- Backend API: http://localhost:5000

## Option 2: Building from Source

If you prefer to build the Docker images yourself:

### Step 1: Clone the repository

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
```

### Step 2: Build and start the containers

```bash
docker compose up -d --build
```

## Configuration Options

You can modify the following environment variables in the Docker Compose file:

| Variable | Description | Default |
|----------|-------------|---------|
| MONGODB_URI | MongoDB connection string | mongodb://mongo:27017/rag_db |
| OLLAMA_HOST_URL | URL for Ollama service | http://host.docker.internal:11434 |
| LLM_MODEL | LLM model to use | gemma3:1b |
| QDRANT_HOST | Qdrant vector database host | qdrant |
| QDRANT_PORT | Qdrant vector database port | 6333 |

## Troubleshooting

### Checking Logs

```bash
# All services
docker compose -f docker-compose.ghcr.yml logs

# Specific service
docker compose -f docker-compose.ghcr.yml logs backend
```

### Common Issues

1. **Cannot connect to Ollama**
   - Make sure Ollama is running locally with `ollama serve`
   - Verify the model is installed with `ollama list`

2. **Frontend cannot connect to backend**
   - Check if backend container is running: `docker ps`
   - Verify backend logs: `docker compose logs backend`

3. **Data persistence issues**
   - The application uses Docker volumes for persistence
   - Check volume status: `docker volume ls` 