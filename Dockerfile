# ===============================
# Stage 1 - Base with dependencies
# ===============================
FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies for faiss/torch/pandas
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Stage 2 - Final runtime image
# ===============================
FROM python:3.10-slim

WORKDIR /app

# Copy installed dependencies from base
COPY --from=base /usr/local /usr/local

# Copy application code
COPY . .

# Expose port (Railway assigns dynamically, uses $PORT)
EXPOSE 8000

# Command for Railway (use $PORT env var)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
