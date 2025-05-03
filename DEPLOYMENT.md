# Deployment Guide for RAG Application

This guide explains how to deploy the RAG application using Docker and GitHub Container Registry (GHCR).

## Prerequisites

- GitHub account with permissions to the repository
- Docker and Docker Compose installed on your deployment machine
- Access to GitHub Container Registry (GHCR)
- Ollama running on the host machine for LLM functionality

## CI/CD Pipeline

The application uses GitHub Actions for continuous integration and delivery:

- Images are automatically built and published to GHCR when pushing to the `dev-ops` branch
- Tagged versions (e.g., `v1.0.0`) also trigger the build and publish workflow
- Pull requests to the `dev-ops` branch will build but not publish images

## Managing the Dev-Ops Branch

To set up and use the dev-ops branch for CI/CD:

```bash
# Create and switch to the dev-ops branch
git checkout -b dev-ops

# After making changes, commit them
git add .
git commit -m "Your commit message"

# Push to the remote repository
git push -u origin dev-ops
```

To create a release version:

```bash
# Create a tag
git tag -a v1.0.0 -m "First production release"

# Push the tag
git push origin v1.0.0
```

## Deploying from GHCR (Production)

### 1. Log in to GitHub Container Registry

```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

Replace `USERNAME` with your GitHub username and ensure you have a `GITHUB_TOKEN` environment variable set with a token that has `read:packages` scope.

### 2. Create a .env file (optional)

Create a `.env` file with custom settings:

```
REGISTRY=ghcr.io
GITHUB_REPOSITORY=yourusername/rag-app
TAG=latest
FRONTEND_PORT=3000
BACKEND_PORT=5000
OLLAMA_HOST_URL=http://host.docker.internal:11434
LLM_MODEL=tinyllama
```

### 3. Pull and run with Docker Compose

```bash
# Pull and start the application
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

### 4. Setup Ollama (if not already running)

The application requires Ollama running on the host machine. To install:

1. Download Ollama from https://ollama.com/download
2. Install and start Ollama
3. Pull the required model:

```bash
ollama pull tinyllama
```

If you want to use a different model, specify it in the `.env` file or edit the `docker-compose.prod.yml` file.

## Local Development Deployment

For local development, use the standard docker-compose file:

```bash
docker-compose build
docker-compose up -d
```

## Accessing the Application

- Frontend: http://localhost:3000 (or your custom FRONTEND_PORT)
- Backend API: http://localhost:5000 (or your custom BACKEND_PORT)

## Troubleshooting

### Connection Issues

If the backend cannot connect to Qdrant or MongoDB:
- Check that all containers are running: `docker ps`
- Inspect the logs: `docker logs rag-app-backend-1`
- Ensure proper network connectivity: `docker network inspect rag-app_rag-network`

### PDF Upload Failures

If PDF uploads fail:
- Check if environment variables are being passed correctly
- Verify Qdrant is running and accessible
- Look for errors in the logs: `docker logs rag-app-backend-1`

## Updating the Application

To update to a new version:

```bash
# Pull the latest images
docker-compose -f docker-compose.prod.yml pull

# Restart the services
docker-compose -f docker-compose.prod.yml up -d
``` 