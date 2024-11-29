FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY models /app/models
COPY data /app/data
COPY confluence_webapp /app/confluence_webapp
COPY pyproject.toml /app/pyproject.toml

COPY requirements.txt /app/requirements.txt
RUN pip install  --no-cache-dir opencv-python==4.7.0.68 
RUN pip install --no-cache-dir opencv-python-headless==4.7.0.72

RUN pip install numpy==1.24.2
RUN pip install --upgrade pip \
    && pip install torch==1.13.0 \
    && pip install -e . \
    && pip install hatch==1.13.0

# Build the project (using hatch)
RUN hatch build
# Expose Streamlit's default port
EXPOSE 8501

# Entry point for running Streamlit
CMD ["streamlit", "run", "confluence_webapp/src/app.py", "--theme.base", "light", "--theme.primaryColor", "blue"]
