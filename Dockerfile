# ─────────────────────────────────────────────────────────────
# Stage 1 — builder
#   Installs Python dependencies into an isolated prefix so
#   only the compiled packages (no pip/wheel cache) are copied
#   to the final image, keeping it as small as possible.
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# Install build tools needed by some packages (e.g. numpy C extensions)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into /install/packages — not the system site-packages
RUN pip install --no-cache-dir --prefix=/install/packages -r requirements.txt


# ─────────────────────────────────────────────────────────────
# Stage 2 — runtime
#   Copies only the installed packages from the builder and
#   the application code. No compiler, no pip, no cache files.
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create a non-root user and group for security
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install/packages /usr/local

# Copy application source
COPY app/ ./app/

# Copy data directory (pre-built embeddings + raw documents)
# Note: .env is intentionally NOT copied — pass secrets via env vars at runtime
COPY data/ ./data/

# Give the non-root user ownership of the app directory
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# --workers 1: single worker keeps the in-memory vector store consistent
# --no-access-log: reduces noise; use a reverse proxy for access logging in prod
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--no-access-log"]
