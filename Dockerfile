FROM python:3.12-slim AS base

# System deps for sandbox (restricted backend uses subprocess + rlimits).
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package with api + tracing extras (editable so config/examples
# from the repo are available; for a published image, replace with a wheel).
COPY pyproject.toml README.md ./
COPY koboi/ koboi/
COPY configs/ configs/
COPY examples/ examples/
COPY skills/ skills/

RUN pip install --no-cache-dir -e ".[api,tracing]"

# Default runtime config — override via volume mount or KOBOI_CONFIG env.
ENV KOBOI_CONFIG=/app/configs/server_simple.yaml
ENV KOBOI_HOST=0.0.0.0
ENV KOBOI_PORT=8080

# Persistent data: SQLite DB + workspace + keys file.
ENV KOBOI_DATA_DIR=/data
RUN mkdir -p /data /data/workspace

# H8: least-privilege non-root user (koboi binds 8080 >1024, so no NET_BIND_SERVICE).
RUN groupadd -r koboi && useradd -r -g koboi -u 1000 koboi \
    && chown -R koboi:koboi /app /data
USER koboi

EXPOSE 8080

# Health check (liveness).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

CMD ["python", "-m", "koboi.cli", "serve", "/app/configs/e2e_full.yaml", "--host", "0.0.0.0", "--port", "8080"]
