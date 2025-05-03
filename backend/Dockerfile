# FILE: backend/Dockerfile
# Combined Node.js backend and Python environment

# Use a Node base image that includes Python or makes it easy to install
# Node 18 includes Python 3.11 in some variants (e.g., bookworm)
FROM node:18-bookworm-slim

# Set working directory
WORKDIR /app

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-full \
    python3-pip \
    python3-venv \
    python3-dev \
    # Add any other system libs needed by Python packages
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONPATH="/app/python"

# Copy Python requirements first
COPY python/requirements.txt ./python_requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ./python_requirements.txt

# Copy Python scripts to maintain proper structure
COPY python /app/python

# Change to backend directory for Node.js operations
WORKDIR /app/backend

# Copy Node.js package files and install dependencies
COPY backend/package.json backend/package-lock.json* ./
RUN npm ci --only=production

# Copy the rest of the backend code
COPY backend/ .

# Create uploads directory structure
RUN mkdir -p ./uploads/images

# Expose the port the Node server listens on
EXPOSE 5000

# Set environment variables
ENV NODE_ENV=production
ENV PORT=5000

# Command to run the server
CMD ["node", "server.js"]