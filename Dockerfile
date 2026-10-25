FROM python:3.11-slim

# Install system deps (IMPORTANT for whisper + ffmpeg)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv (faster dependency install)
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY server.py .
COPY index.html .

# Install dependencies from pyproject
RUN uv pip install --system .

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

# Cloud Run uses PORT env variable
ENV PORT=8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]