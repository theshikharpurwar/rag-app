# FILE: python/Dockerfile (Conceptual - requires creating api_server.py)

FROM python:3.11-slim 

WORKDIR /app

# Install system dependencies if needed (e.g., for PyMuPDF)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install Python dependencies
# Consider using --no-cache-dir if image size is critical
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the Python application code
# Ensure embeddings/ and llm/ directories are copied
COPY . .

# Expose the port the Python API server will run on
EXPOSE 5001

# Command to run the Python API server (replace api_server.py and Flask command as needed)
# Ensure it binds to 0.0.0.0 to be accessible from other containers
CMD ["flask", "--app", "api_server:app", "run", "--host=0.0.0.0", "--port=5001"]

# --- NOTE ---
# You would need to create api_server.py with Flask routes that wrap the logic
# currently in compute_embeddings.py and local_llm.py.
# The Node.js backend would then make HTTP requests to this Python service
# instead of using spawn('python', ...).