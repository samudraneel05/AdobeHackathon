# Adobe Hackathon Challenge 1A: PDF Outline Extraction
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (vendor all packages)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY process_pdfs.py .
COPY sample_dataset/schema/ /app/schema/

# Create mount points
RUN mkdir -p /app/input /app/output

# Run the processing script
CMD ["python", "process_pdfs.py"] 