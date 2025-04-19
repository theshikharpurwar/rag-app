# FILE: backend/Dockerfile
# Combined Node.js backend and Python environment for Setup A

FROM node:18-bookworm-slim

WORKDIR /app

# Install Python 3 pip and potentially build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    # Add other system libs if python deps need them (e.g., build-essential)
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python Dependencies
COPY python/requirements.txt ./python_requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r python_requirements.txt

# Copy all python code
COPY python ./python/

# Install Node.js Dependencies
COPY package.json ./
COPY package-lock.json* ./
RUN npm ci --only=production --legacy-peer-deps

# Copy Application Code
COPY . .

# Create uploads directory structure
RUN mkdir -p /app/uploads/images

EXPOSE 5000
ENV PATH="/usr/local/bin:${PATH}"
CMD [ "node", "server.js" ]