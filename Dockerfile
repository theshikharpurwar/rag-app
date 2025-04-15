# Stage 1: Build the Node.js backend
FROM node:16-alpine AS backend

WORKDIR /app/backend

# Copy package.json and install dependencies
COPY backend/package.json backend/package-lock.json* ./
RUN npm install --production

# Copy the backend application
COPY backend/ .

# Stage 2: Install Python and dependencies
FROM python:3.9-slim AS python

WORKDIR /app/python

# Copy Python scripts and requirements
COPY python/ .
RUN pip install -r requirements.txt

# Stage 3: Final image
FROM node:16-alpine

WORKDIR /app

# Copy backend from builder
COPY --from=backend /app/backend ./backend/

# Copy Python environment
COPY --from=python /app/python ./python/

# Install runtime dependencies for Python
RUN apk add --no-cache python3 py3-pip
RUN pip install qdrant-client sentence-transformers

# Expose the port
EXPOSE 5000

# Set environment variables
ENV NODE_ENV=production
ENV PORT=5000
ENV PYTHONPATH=/app/python

# Create uploads directory
RUN mkdir -p /app/backend/uploads

# Run the application
CMD ["node", "backend/server.js"]