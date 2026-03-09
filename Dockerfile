FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN uv sync --frozen --no-dev

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/workspace/src"

ENTRYPOINT ["python", "-m", "leomed_bench.train"]
CMD ["--config", "configs/cifar10_1gpu.yaml"]
