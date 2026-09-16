# distrainer node image: one image for the Ray head, the workers and the driver. Dependencies come
# from uv.lock (CPU torch, Ray 2.58) so the container matches the laptop environment exactly.
# Every layer is multi-architecture: `docker build` makes the Mac's own (arm64 under OrbStack),
# `docker buildx build --platform linux/amd64,linux/arm64 --push` one manifest for both (the
# uncloud driver's `build` does that for a registry image; OrbStack builds the amd64 half under
# Rosetta, the CPU torch wheel exists for x86_64 and aarch64). A GPU image (CUDA torch, an x86
# CUDA base image) is a separate Dockerfile, not a platform of this one.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAY_TRAIN_V2_ENABLED=1 \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    RAY_DISABLE_DOCKER_CPU_WARNING=1
WORKDIR /app
# procps: the scenario runner's abort path runs `pkill` inside the head (the slim image has none)
RUN apt-get update && apt-get install -y --no-install-recommends procps && rm -rf /var/lib/apt/lists/*
# dependency layer (cached until the lock changes), then the project itself
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
RUN chmod +x deploy/ray-head.sh deploy/ray-worker.sh
# Debian's /etc/profile resets PATH in login shells (`bash -lc`, which some KubeRay code paths and
# tools use): put the virtualenv back there so `ray` and `python` are found either way
RUN printf 'export PATH=/app/.venv/bin:$PATH\n' > /etc/profile.d/distrainer-venv.sh
