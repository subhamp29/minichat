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

# Expose the Streamlit port
EXPOSE 7860

# Run Streamlit (Railway sets $PORT automatically)
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0"]
