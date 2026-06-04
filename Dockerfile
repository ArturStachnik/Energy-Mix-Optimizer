# Multi-stage build for the Energy Mix Optimizer API.
#
# Stage 1 builds the wheels in a fat base image; stage 2 installs only the
# wheels into a slim runtime image. The result is small, reproducible, and
# runs as a non-root user.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip build \
    && python -m build --wheel --outdir /wheels


FROM python:3.11-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMO_ARTIFACTS_DIR=/app/artifacts \
    EMO_API_HOST=0.0.0.0 \
    EMO_API_PORT=8000

RUN groupadd --system app && useradd --system --gid app --home /app --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels

RUN mkdir -p /app/artifacts/models && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; urllib.request.urlopen('http://localhost:8000/health', timeout=3).read(); sys.exit(0)" || exit 1

CMD ["uvicorn", "energy_mix_optimizer.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
