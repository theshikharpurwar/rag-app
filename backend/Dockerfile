# FILE: backend/Dockerfile
# Combined Node.js backend and Python environment

# Use a Node base image that includes Python or makes it easy to install
# Node 18 includes Python 3.11 in some variants (e.g., bookworm)
FROM node:18-bookworm-slim

# Set working directory
WORKDIR /app

# Install Python build dependencies (if needed for some pip packages), then cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    # Add any other system libs needed by Python packages (e.g., for PyMuPDF if needed)
    # build-essential libpoppler-cpp-dev pkg-config python3-tk etc.
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Install Python Dependencies ---
# Copy only requirements first to leverage Docker cache
COPY python/requirements.txt ./python_requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r python_requirements.txt

# Copy all python code (needed for running scripts)
# Ensure embeddings/, llm/, utils/ are included if they exist in your python dir
COPY python ./python/

# --- Install Node.js Dependencies ---
# Copy package files first
COPY package.json ./
COPY package-lock.json* ./
# Install production Node dependencies
RUN npm ci --only=production

# --- Copy Application Code ---
# Copy the rest of the backend code AFTER installing dependencies
COPY . .

# Create uploads directory structure (ensure correct path relative to WORKDIR /app)
# This will be inside the container at /app/uploads
RUN mkdir -p /app/uploads/images

# Expose the port the Node server listens on
EXPOSE 5000

# Command to run the backend server
# Ensure Python3 is in the PATH or use python3 explicitly in spawn
ENV PATH="/usr/local/bin:${PATH}" 
CMD [ "node", "server.js" ]