# Use a lightweight Python image with 3.10
FROM python:3.10-slim

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy project files
COPY . .

# Create and activate a virtual environment
RUN python -m venv /app/venv \
    && /app/venv/bin/pip install --upgrade pip \
    && /app/venv/bin/pip install torch==1.13.0 \
    && /app/venv/bin/pip install -e . \
    && /app/venv/bin/pip install hatch==1.13.0 \
    && /app/venv/bin/hatch build

# Expose Streamlit's default port
EXPOSE 8501

# Entry point for running Streamlit
CMD ["/app/venv/bin/streamlit", "run", "confluence_webapp/src/app.py", "--theme.base", "light", "--theme.primaryColor", "blue"]
