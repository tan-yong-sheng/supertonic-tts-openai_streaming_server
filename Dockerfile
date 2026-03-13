# Dockerfile for SupertonicTTS OpenAI-Compatible Server
# Optimized for CPU inference (Supertonic runs efficiently on CPU)

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --shell /bin/bash supertonic
WORKDIR /app

COPY --chown=supertonic:supertonic app/ ./app/
COPY --chown=supertonic:supertonic static/ ./static/
COPY --chown=supertonic:supertonic templates/ ./templates/
COPY --chown=supertonic:supertonic voices/ ./voices/
COPY --chown=supertonic:supertonic tools/ ./tools/
COPY --chown=supertonic:supertonic server.py ./

RUN chown supertonic:supertonic /app && mkdir -p /app/logs && chown supertonic:supertonic /app/logs

RUN mkdir -p /home/supertonic/.cache/huggingface && \
    chown -R supertonic:supertonic /home/supertonic/.cache

USER supertonic

ENV SUPERTONIC_HOST=0.0.0.0 \
    SUPERTONIC_PORT=49112 \
    SUPERTONIC_VOICES_DIR=/app/voices \
    SUPERTONIC_LOG_DIR=/app/logs \
    SUPERTONIC_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

EXPOSE 49112

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:49112/health')" || exit 1

CMD ["python", "server.py"]
