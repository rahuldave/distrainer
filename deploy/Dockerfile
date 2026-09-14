# distrainer node image: one image for the Ray head, the workers and the driver (arm64-native
# under OrbStack). Dependencies come from uv.lock (CPU torch, Ray 2.58) so the container matches
# the laptop environment exactly.
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
# dependency layer (cached until the lock changes), then the project itself
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
RUN chmod +x deploy/ray-head.sh deploy/ray-worker.sh
