# Packages the whole tini-agent source into an image that runs the dashboard
# by default. See docker-compose.yml for the full self-hosted stack (this +
# Qdrant + Ollama) and docs/document-ingestion.md for the RAG setup this
# image is meant to run.
#
#   docker compose up -d --build          # dashboard at http://localhost:9000
#   docker compose run --rm tini python -m tini ingest examples/denso_mock_docs --workspace denso
#   docker compose run --rm tini python -m tini            # chat in the terminal
#
# Runtime state (.tini/: memory db, traces, MCP config) is a volume in
# docker-compose.yml, not baked into the image — same "runtime state stays
# outside version control" rule CLAUDE.md states for a bare-metal install.
FROM python:3.11-slim

WORKDIR /app

# Install the package (core + qdrant + ocr, since this image's reason to
# exist is the self-hosted RAG stack, including scanned-PDF ingestion) before
# copying the rest of the source, so an ordinary code change doesn't
# invalidate the dependency layer.
COPY pyproject.toml README.md ./
COPY tini ./tini
COPY skills ./skills
RUN pip install --no-cache-dir '.[qdrant,ocr,minio]'

# Everything else a contributor might run from inside the container: scripts,
# evals, docs, examples (including the Denso mock corpus).
COPY scripts ./scripts
COPY evals ./evals
COPY docs ./docs
COPY examples ./examples
COPY Makefile ./

EXPOSE 9000
ENV TINI_HOME=/app/.tini
ENV TINI_DASHBOARD_HOST=0.0.0.0

CMD ["python", "-m", "tini.ops.dashboard"]
