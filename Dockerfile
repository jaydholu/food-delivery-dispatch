# ============================================================
# Food Delivery Dispatch — OpenEnv Production Dockerfile
#
# Build:
#   docker build -t food_delivery_dispatch-env:latest .
#
# Run server:
#   docker run --rm -p 8000:8000 food_delivery_dispatch-env:latest
#
# Run with task override:
#   docker run --rm -p 8000:8000 -e FOOD_DELIVERY_TASK=hard food_delivery_dispatch-env:latest
# ============================================================

FROM python:3.12.6-slim

LABEL maintainer="Food Delivery RL Team"
LABEL description="OpenEnv Food Delivery Dispatch RL Environment"
LABEL version="1.0.0"

WORKDIR /app

# Minimal system build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy ALL application source files
COPY __init__.py      ./
COPY models.py        ./
COPY client.py        ./
COPY inference.py     ./
COPY openenv.yaml     ./
COPY pyproject.toml   ./
COPY server/          ./server/
COPY tasks/           ./tasks/
COPY baseline/        ./baseline/

# PYTHONPATH so imports resolve from /app
ENV PYTHONPATH="/app"
ENV FOOD_DELIVERY_TASK="medium"

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Default: start OpenEnv HTTP + WebSocket server on HF Spaces port
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
