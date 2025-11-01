# Base image with Python 3.12 slim
FROM python:3.14-slim AS app

# Set environment variables for Python and uv package manager
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.venv/bin:${PATH}"

# Base system packages for scientific Python wheels and archive handling
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Install uv once for dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Prime dependency cache before copying the full source tree
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Bring in the application sources (including pre-downloaded datasets under ./data)
COPY . .

# Install the project itself into the managed environment
RUN uv sync --frozen --no-dev

# Open port 8501 for Streamlit
EXPOSE 8501

# Launch the Streamlit web application
CMD ["streamlit", "run", "src/webapp.py", "--server.address=0.0.0.0", "--server.port=8501"]
