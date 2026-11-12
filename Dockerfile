FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project
COPY src ./src
COPY configs ./configs
COPY data/splits ./data/splits
RUN uv sync --frozen
ENTRYPOINT ["uv", "run", "--no-sync", "pawmatch"]
CMD ["--help"]
