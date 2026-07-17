# ---- base: runtime image ----
FROM python:3.12-slim AS base

# System deps for sandbox (restricted backend uses subprocess + rlimits).
# `libseccomp2` is the runtime shared lib the binding links against; `python3-seccomp`
# is the libseccomp Python binding Debian ships for the system python3.12 (ABI cp312,
# which this image's /usr/local/bin/python3.12 also speaks). PYTHONPATH below makes it
# importable by /usr/local/bin/python3.12 so HARD syscall-layer egress deny
# (sandbox.network_isolation: seccomp) actually works in the shipped image (issue #51).
#
# Why apt and not build-from-source: the upstream libseccomp binding is NOT on PyPI,
# and its setup.py expects a full autotools C-library build (./autogen.sh && configure
# && make) to generate version.h before the Python sources can even generate metadata
# (raw `pip install git+...#subdirectory=src/python` fails with KeyError: 'VERSION_RELEASE'
# / BackendUnavailable). The Debian package is the same one the `seccomp:` CI job uses
# to run the egress tests, so it is known-good on this distro+python.
#
# Fail-safe: if a future Debian/Python ABI mismatch ever makes the binding unimportable,
# `network_isolation: seccomp_strict` refuses to boot (issue #51 installer is
# fail-closed) and legacy `seccomp` soft-degrades — never a silent fail-open.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        git \
        libseccomp2 \
        python3-seccomp \
    && rm -rf /var/lib/apt/lists/*
# Make the system python3-seccomp binding importable by this image's /usr/local/bin/python3.12.
ENV PYTHONPATH=/usr/lib/python3/dist-packages

WORKDIR /app

# Install the package with api + tracing extras (editable so config/examples
# from the repo are available; for a published image, replace with a wheel).
COPY pyproject.toml README.md ./
COPY koboi/ koboi/
COPY configs/ configs/
COPY examples/ examples/
COPY skills/ skills/

RUN pip install --no-cache-dir -e ".[api,tracing]"

# Best-effort build-time smoke: confirm HARD isolation is wired (prints OK). On a rare
# ABI mismatch, print WARN instead of failing the build — `seccomp_strict` fail-closes
# at runtime, so this never silently ships a fail-open image.
RUN python -c "import seccomp; print('seccomp OK')" || echo "WARN: seccomp not importable under this image's python; HARD isolation unavailable (soft-degrade / seccomp_strict fail-closed)"

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
