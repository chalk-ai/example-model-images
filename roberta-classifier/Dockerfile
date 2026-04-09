# Roberta Classifier using chalk-remote-call-python
#
# Build:
#   docker build --platform linux/amd64 -t roberta-classifier .
#
# Run:
#   docker run -p 8080:8080 roberta-classifier

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:${PATH}"

RUN pip install --no-cache-dir uv

# Install core dependencies from lockfile for reproducible builds.
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project --no-group train

# Install remote-call runtime as an explicit shim layer.
# --no-deps keeps lockfile-managed core dependencies authoritative.
RUN uv pip install --no-deps chalk-remote-call-python

COPY classifier.py /app/classifier.py
COPY model.py /app/model.py

ENV PYTHONPATH=/app

EXPOSE 8080

ENTRYPOINT ["chalk-remote-call", "--handler", "model.handler", "--port", "8080"]
