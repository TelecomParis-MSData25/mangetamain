# Use the official Python image with the required interpreter version
FROM python:3.12-slim AS base

# Configure Python for container usage and ensure uv creates copy-based venvs
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive

# Install system packages needed for building scientific Python wheels and fetching uv
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install the uv package manager (placed in /usr/local/bin for global usage)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Provide a `python` shim only if it does not already exist
RUN if ! command -v python >/dev/null 2>&1; then ln -s /usr/local/bin/python3 /usr/local/bin/python; fi

# Work inside the application directory
WORKDIR /app

# Copy dependency manifests first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Resolve and download dependencies using uv without yet copying the project source
RUN uv sync --frozen --no-dev --no-install-project

# Copy the remainder of the project including Streamlit entrypoint and configuration
COPY . .

# Install the project itself into the uv-managed virtual environment
RUN uv sync --frozen --no-dev

# Make the virtual environment executables available on PATH for subsequent commands
ENV PATH="/app/.venv/bin:${PATH}"

# Streamlit listens on port 8501 by default
EXPOSE 8501

# Launch the Streamlit web application via uv (guarantees the right environment is active)
CMD ["uv", "run", "streamlit", "run", "src/webapp.py", "--server.address=0.0.0.0", "--server.port=8501"]
