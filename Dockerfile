FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY openenv.yaml .
COPY inference.py .
COPY env/ ./env/

# Expose port for HF Spaces (default 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860"]

