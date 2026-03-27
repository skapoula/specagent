# Installation

## Requirements

- Python 3.11 or higher
- A Groq API key (free at [console.groq.com](https://console.groq.com)) — or a custom OpenAI-compatible LLM endpoint
- 2 GB disk space for the embedding model cache
- Internet access for initial model download and LLM API calls

## Install

### Option A: pip install from source (recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd specagent
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
3. Install the package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Set your Groq API key:
   ```bash
   export GROQ_API_KEY=your_groq_api_key_here
   ```
5. Download the embedding model to local cache:
   ```bash
   specagent download-model
   ```

### Option B: Docker Compose

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd specagent
   ```
2. Copy the environment template and fill in your API key:
   ```bash
   cp .env.example .env
   # Edit .env and set GROQ_API_KEY=your_groq_api_key_here
   ```
3. Start all services (API + Phoenix tracing UI):
   ```bash
   docker compose up
   ```
4. The API is available at `http://localhost:8000`.
5. The Phoenix tracing dashboard is available at `http://localhost:6006`.

## Verify It's Working

```bash
specagent version
```

Expected output:
```
specagent 0.3.0
```

Then run a health check against the API server (if using Option B or after `specagent serve`):
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{"status": "ok", "version": "0.3.0", "index_loaded": false}
```

## Uninstall

```bash
pip uninstall specagent
rm -rf data/lancedb   # Remove the local vector index
```
