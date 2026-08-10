#!/bin/bash
set -e

# Use Railway's PORT if available, otherwise default to 7860
PORT_NUM=${PORT:-7860}

echo "Starting Streamlit on port $PORT_NUM"
exec streamlit run app.py --server.port=$PORT_NUM --server.address=0.0.0.0
