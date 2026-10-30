FROM python:3.12-slim

WORKDIR /app

# Install dependencies first, so Docker can cache this layer between code changes
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
# Cloud Run tells the container which port to listen on via $PORT
CMD streamlit run src/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true
