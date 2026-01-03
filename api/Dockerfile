FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (context is project root)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/

# Copy rainbowplus
COPY rainbowplus/ ./rainbowplus/

# Copy configs
COPY configs/ ./configs/

# Create data directory for SQLite
RUN mkdir -p data

# Expose port
EXPOSE 8000

# Set working directory to api folder
WORKDIR /app/api

# Run the application
CMD ["python", "server.py"]
