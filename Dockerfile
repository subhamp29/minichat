FROM python:3.11-slim

# Install system dependencies required for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose the Streamlit port
EXPOSE 7860

# Healthcheck: wait for Streamlit to respond
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
  CMD curl -f http://localhost:${PORT:-7860}/_stcore/health || exit 1

# Run Streamlit via entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
