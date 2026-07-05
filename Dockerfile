FROM python:3.12-slim AS base

# System deps for sandbox (restricted backend uses subprocess + rlimits; the
# [sandbox-seccomp] extra's `seccomp` package builds against libseccomp-dev and
# needs libseccomp2 at runtime for HARD network isolation).
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        git \
        libseccomp-dev \
        libseccomp2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package with api + tracing + sandbox-seccomp extras (editable so
# config/examples from the repo are available; for a published image, replace
# with a wheel). [sandbox-seccomp] enables the HARD egress-deny filter used by
# configs/server_deploy.yaml's network_isolation: seccomp.
COPY pyproject.toml README.md ./
COPY koboi/ koboi/
COPY configs/ configs/
COPY examples/ examples/
COPY skills/ skills/

RUN pip install --no-cache-dir -e ".[api,tracing,sandbox-seccomp]"

# Default runtime config — override via volume mount or KOBOI_CONFIG env.
ENV KOBOI_CONFIG=/app/configs/server_simple.yaml
ENV KOBOI_HOST=0.0.0.0
ENV KOBOI_PORT=8080
# Tier 2: a mounted/derived extensions dir made importable by koboi/_extensions_path.py.
ENV KOBOI_EXTENSIONS_DIR=/app/ext

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

# Shell form so KOBOI_CONFIG/HOST/PORT env vars are honored (override at `docker run`
# without retyping the CMD). Defaults preserve the prior behavior (e2e_full on :8080).
CMD python -m koboi.cli serve "${KOBOI_CONFIG:-/app/configs/server_simple.yaml}" --host "${KOBOI_HOST:-0.0.0.0}" --port "${KOBOI_PORT:-8080}"
