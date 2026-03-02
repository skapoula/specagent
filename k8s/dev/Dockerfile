# =============================================================================
# SpecAgent k3s Development Container
# =============================================================================
# Build and push to your registry:
#   docker build -t your-registry/specagent-dev:latest -f k8s/dev/Dockerfile .
#   docker push your-registry/specagent-dev:latest
#
# Or for local k3s with containerd:
#   docker build -t specagent-dev:latest -f k8s/dev/Dockerfile .
#   docker save specagent-dev:latest | sudo k3s ctr images import -
# =============================================================================

FROM codercom/code-server:latest

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    wget \
    jq \
    tree \
    vim \
    htop \
    ca-certificates \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel --break-system-packages

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Install heavy Python dependencies (cached layers)
RUN pip install --no-cache-dir --break-system-packages numpy>=1.26.0 lancedb>=0.14.0

# Install project dependencies
RUN pip install --no-cache-dir --break-system-packages \
    langgraph>=0.2.0 langchain>=0.3.0 langchain-community>=0.3.0 \
    langchain-huggingface>=0.1.0 fastapi>=0.115.0 uvicorn[standard]>=0.32.0 \
    httpx>=0.27.0 pydantic>=2.9.0 pydantic-settings>=2.6.0 pandas>=2.2.0 \
    huggingface-hub>=0.26.0 python-dotenv>=1.0.0 tenacity>=9.0.0 \
    structlog>=24.4.0 rich>=13.9.0 typer>=0.13.0

# Install dev dependencies
RUN pip install --no-cache-dir --break-system-packages \
    pytest>=8.3.0 pytest-cov>=6.0.0 pytest-asyncio>=0.24.0 pytest-httpx>=0.34.0 \
    ruff>=0.7.0 mypy>=1.13.0 bandit>=1.7.0 pre-commit>=4.0.0 \
    ragas>=0.2.0 arize-phoenix>=5.0.0 gradio>=5.6.0

# Install VS Code extensions
RUN code-server --install-extension ms-python.python \
    && code-server --install-extension charliermarsh.ruff \
    && code-server --install-extension redhat.vscode-yaml \
    && code-server --install-extension eamodio.gitlens

# Create workspace
RUN mkdir -p /workspace && chown coder:coder /workspace

# Switch back to coder user
USER coder
WORKDIR /workspace

# Set environment
ENV PYTHONPATH=/workspace/src
ENV PATH="/home/coder/.local/bin:$PATH"
