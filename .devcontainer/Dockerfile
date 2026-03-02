# =============================================================================
# SpecAgent Development Container
# =============================================================================
# This Dockerfile creates a development environment with:
# - Python 3.11 with all project dependencies
# - Claude Code CLI
# - Node.js for MCP servers
# - Common development tools
# =============================================================================

ARG PYTHON_VERSION=3.11

FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION}-bookworm

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    gcc \
    g++ \
    # Git and utilities
    git \
    curl \
    wget \
    jq \
    # For httpx/networking
    ca-certificates \
    # For development
    vim \
    less \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
# Claude Code requires Node.js (installed via devcontainer features)
# The CLI is installed via npm in post-create script since Node comes from features

# Upgrade pip and install Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pre-install heavy dependencies to speed up builds
# These are cached in the Docker layer
RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    lancedb>=0.14.0

# Set up workspace
WORKDIR /workspaces/specagent

# Copy project files for dependency installation
# Note: In devcontainer, the repo is mounted at /workspaces/specagent
# This COPY is only used during image build for pre-caching
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install project dependencies (will be reinstalled with mounted source)
RUN pip install --no-cache-dir -e ".[dev,eval,ui]" || true

# Clean up
RUN pip cache purge

# Set Python path
ENV PYTHONPATH=/workspaces/specagent/src

# Create directories for data
RUN mkdir -p /workspaces/specagent/data/raw \
    /workspaces/specagent/data/processed \
    /workspaces/specagent/data/index

# Switch to non-root user (provided by base image)
USER vscode

# Set shell to bash
SHELL ["/bin/bash", "-c"]
