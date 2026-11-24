# Build:  docker build -t inspire-mcp .                         (lexical reranking, small image)
#         docker build --build-arg EXTRAS=semantic -t inspire-mcp:semantic .   (dense reranking, model baked in)
FROM python:3.12-slim
ARG EXTRAS=""
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/app/models INSPIRE_MCP_CACHE=/cache PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project ${EXTRAS:+--extra $EXTRAS}
COPY src ./src
RUN uv sync --frozen --no-dev ${EXTRAS:+--extra $EXTRAS}
# Download the models at build time so the container needs no access to Hugging Face at runtime.
RUN if [ -n "$EXTRAS" ]; then /app/.venv/bin/python -c "\
from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"; fi
RUN useradd --create-home --uid 10001 app && mkdir -p /cache && chown app /cache
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["/app/.venv/bin/inspire-mcp", "--transport", "streamable-http", "--host", "0.0.0.0"]
