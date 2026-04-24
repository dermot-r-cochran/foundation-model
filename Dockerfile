# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Build dependencies only needed for native torch extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# Install CPU-only torch first so the image stays small; override at build time
# for GPU nodes by passing --build-arg TORCH_EXTRA="--index-url https://download.pytorch.org/whl/cu121"
ARG TORCH_EXTRA=""
RUN pip install --no-cache-dir torch ${TORCH_EXTRA} \
    && pip install --no-cache-dir -e ".[dev]"

COPY . .

# ── CPU target (local device / cloud VM without GPU) ──────────────────────────
FROM base AS cpu
ENV DEVICE=cpu
ENV DEMO_MODE=all
CMD ["python", "-m", "foundation_model"]

# ── GPU target (cloud VM with CUDA) ───────────────────────────────────────────
FROM base AS gpu
ENV DEVICE=cuda
ENV DEMO_MODE=all
CMD ["python", "-m", "foundation_model"]
