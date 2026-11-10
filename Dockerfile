FROM python:3.11-slim

WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .
RUN pip install --no-cache-dir -e .

# Create results directory
RUN mkdir -p results

# Default: run the quick end-to-end demo
CMD ["python", "scripts/quick_results.py"]
