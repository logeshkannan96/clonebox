FROM python:3.12.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 clonebox
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    CLONEBOX_HOST=0.0.0.0 \
    CLONEBOX_DATA=/data \
    HF_HOME=/cache/huggingface

# Dependencies first so code edits don't reinstall torch.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

COPY app.py core.py ./
COPY static ./static

RUN mkdir -p /data /cache && chown clonebox:clonebox /data /cache
USER clonebox

EXPOSE 8000
VOLUME ["/data", "/cache"]
CMD ["/opt/venv/bin/python", "app.py"]
