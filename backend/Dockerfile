# FILE: backend/Dockerfile
# (Corrected for PEP 668 using venv - FINAL VERSION FOR SETUP A)

FROM node:18-bookworm-slim

WORKDIR /app

# Install Python 3 pip, Python 3 dev headers, and venv package
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-venv \
    # Add other system libs if required by python dependencies here (e.g., build-essential)
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Create and activate virtual environment ---
RUN python3 -m venv /opt/venv
# Add venv python/pip to path permanently for subsequent RUN/CMD/ENTRYPOINT
ENV PATH="/opt/venv/bin:$PATH"

# --- Install Python Dependencies into venv ---
# Copy requirements first to leverage Docker layer cache
COPY python/requirements.txt ./python_requirements.txt
# Upgrade pip and install requirements inside the venv (use 'python' as it's now from venv PATH)
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r python_requirements.txt

# Copy all python application code
COPY python ./python/

# --- Install Node.js Dependencies ---
COPY backend/package.json ./
COPY backend/package-lock.json* ./
# Using --legacy-peer-deps based on previous frontend logs, remove if not needed for backend
RUN npm ci --only=production --legacy-peer-deps

# --- Copy Application Code ---
COPY backend .

# Create uploads directory structure
RUN mkdir -p /app/uploads/images

EXPOSE 5000
# CMD will use python from the venv thanks to PATH modification above
# and node from the base image
CMD [ "node", "server.js" ]