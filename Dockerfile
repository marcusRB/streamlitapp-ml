FROM python:3.11

# Install system dependencies ONCE

RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./

COPY data/ ./data/

RUN mkdir -p ./models \
    && mkdir -p ./logs \
    && mkdir -p ./output \
    && mkdir -p ./reports \
    && mkdir -p ./figures


# Copy entrypoint script
COPY run_app.sh .
RUN chmod +x run_app.sh

# Create non-root user
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8502

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT ["./run_app.sh"]
