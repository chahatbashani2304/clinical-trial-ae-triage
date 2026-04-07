FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY openenv.yaml .
COPY inference.py .
COPY env/ ./env/
COPY server/ ./server/

# Expose port
EXPOSE 7860

# Healthcheck using Python (no curl needed)
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import requests; r=requests.get('http://localhost:7860/health', timeout=2); exit(0 if r.status_code==200 else 1)" || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860"]



