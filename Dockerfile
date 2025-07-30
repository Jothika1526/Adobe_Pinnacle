# Use the official python:3.10 image
FROM --platform=linux/amd64 python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies required for PyMuPDF, spaCy, and other scientific libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1-mesa-glx \
    gcc \
    gfortran \
    libatlas-base-dev \
    libopenblas-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- IMPORTANT FOR SPACY: Download and install the spaCy model ---
# For spacy 3.8.x, the compatible small model is en_core_web_sm-3.8.0
RUN python -m spacy download en_core_web_sm-3.8.0 --direct

# Verify the spaCy model installation
RUN python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded successfully!')"

# Copy your application files
COPY . /app/

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Command to run your application when the container starts
CMD ["python", "predict.py"]