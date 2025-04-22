# FILE: backend/Dockerfile
# (Paths adjusted for root build context)

FROM node:18-bookworm-slim

WORKDIR /app

# Install Python & Dependencies (Paths relative to WORKDIR /app)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# *** FIX: Copy requirements from python dir relative to root context ***
COPY python/requirements.txt ./python_requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r python_requirements.txt

# *** FIX: Copy python code dir relative to root context ***
# This copies the entire content of ./python into /app/python within the container
COPY python ./python/

# Install Node.js Dependencies
# *** FIX: Copy package files from backend dir relative to root context ***
COPY backend/package.json ./
COPY backend/package-lock.json* ./
RUN npm ci --only=production --legacy-peer-deps

# Copy Application Code
# *** FIX: Copy backend code dir relative to root context ***
# This copies the content of ./backend into /app within the container
COPY backend .

# Create uploads directory (path relative to WORKDIR /app)
RUN mkdir -p /app/uploads/images

EXPOSE 5000
ENV PATH="/usr/local/bin:${PATH}"
CMD [ "node", "server.js" ] # server.js is now at /app/server.js