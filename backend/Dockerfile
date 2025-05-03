# FILE: backend/Dockerfile
# Combined Node.js backend and Python environment

# Stage 1: Build slim Python environment with only needed dependencies
FROM node:20-bookworm-slim

# Set working directory
WORKDIR /app

# Install only the minimal Python dependencies needed
# Avoid installing unnecessary build tools and dev packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    # Avoid installing python3-dev and build-essential
    # as we'll use pre-built wheels where possible
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONPATH="/app/python"

# Copy Python requirements first
COPY python/requirements.txt ./python_requirements.txt

# Install Python dependencies with specific options to avoid CUDA/unnecessary libraries
# Set environment variables to prevent PyTorch from installing CUDA
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Prevent PyTorch from installing CUDA
    TORCH_CUDA_ARCH_LIST="None" \
    USE_CUDA=0 \
    USE_CUDNN=0 \
    FORCE_CPU=1

# Install only CPU versions of PyTorch and other packages
RUN pip install --no-cache-dir --upgrade pip && \
    # Install specific CPU-only version of torch
    pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    # Install the rest of the requirements
    pip install --no-cache-dir -r ./python_requirements.txt

# Copy Python scripts to maintain proper structure
COPY python /app/python

# Change to backend directory for Node.js operations
WORKDIR /app/backend

# Copy Node.js package files and install dependencies
COPY backend/package.json ./
RUN npm install --production

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