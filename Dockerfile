# Dockerfile for Sound Classification ML Pipeline

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data/upload data/temp

# Expose ports
EXPOSE 5000 8501

# Default command (can be overridden in docker-compose)
CMD ["python", "api/app.py"]