# ---- seccomp-builder: compile libseccomp Python bindings for /usr/local/bin/python3.12 ----
# The libseccomp bindings are NOT on PyPI, and apt's `python3-seccomp` targets
# debian's system python3 (3.11), NOT this image's `/usr/local/bin/python3.12`.
# So `import seccomp` would otherwise fail in the runtime image and the sandbox
# would silently fall back to SOFT network deny (token-scan only). To make HARD
# syscall-layer egress deny actually available we build the bindings from the
# upstream libseccomp source against the image's own Python (issue #51).
FROM python:3.12-slim AS seccomp-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        git \
        libseccomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "cython<3" setuptools wheel \
    && pip install --no-cache-dir --no-build-isolation git+https://github.com/seccomp/libseccomp@v2.5.5#subdirectory=src/python
# Build notes: pin Cython<3 (libseccomp 2.5.5's seccomp.pyx predates Cython 3 and
# fails to cythonize under it) and use --no-build-isolation so pip reuses the
# preinstalled Cython (the binding's build-system meta does not declare Cython,
# so PEP-517 isolation hides it → "ModuleNotFoundError: No module named 'Cython'").
# Preinstall setuptools+wheel too: python:3.12-slim ships neither, and
# --no-build-isolation won't fetch them → "BackendUnavailable: Cannot import
# 'setuptools.build_meta'".

# ---- base: runtime image ----
FROM python:3.12-slim AS base

# System deps for sandbox (restricted backend uses subprocess + rlimits).
# libseccomp2 is the runtime shared lib the bindings link against.
#
# NOTE on HARD network isolation (sandbox.network_isolation: seccomp): HARD
# syscall-layer egress deny now WORKS in this image -- the seccomp-builder stage
# above compiles the libseccomp Python bindings for /usr/local/bin/python3.12 and
# we COPY them in below, guarded by a build-time smoke check. (Issue #51: the
# installer is also fail-closed, so `network_isolation: seccomp_strict` will
# refuse to boot if the bindings are ever missing rather than silently soft-
# degrading.) HARD isolation still requires a Linux kernel (containers run Linux);
# on a non-Linux host set `seccomp_strict` to fail-closed or `seccomp` to soft-
# degrade. For full OS-level isolation (filesystem too), use the Docker backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        git \
        libseccomp2 \
    && rm -rf /var/lib/apt/lists/*

# Land the freshly-built seccomp bindings into the runtime Python's site-packages.
COPY --from=seccomp-builder /usr/local/lib/python3.12/site-packages/seccomp* /usr/local/lib/python3.12/site-packages/

WORKDIR /app

# Install the package with api + tracing extras (editable so config/examples
# from the repo are available; for a published image, replace with a wheel).
COPY pyproject.toml README.md ./
COPY koboi/ koboi/
COPY configs/ configs/
COPY examples/ examples/
COPY skills/ skills/

RUN pip install --no-cache-dir -e ".[api,tracing]"

# Build-time smoke check: FAIL the docker build if the seccomp bindings did not
# land (guards against a silent COPY glob miss or a builder-stage failure).
RUN python -c "import seccomp; print('seccomp OK')"

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
