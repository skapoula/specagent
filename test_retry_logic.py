#!/usr/bin/env python3
"""Quick test to verify retry logic works."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Enable debug logging to see retry attempts
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from specagent.llm.factory import create_llm

print("Creating LLM with retry logic...")
llm = create_llm()

print(f"LLM type: {type(llm).__name__}")
print(f"Endpoint: {llm.endpoint_url}")
print(f"Max retries: {llm.max_retries}")
print(f"Retry delay: {llm.retry_delay}s")
print()

print("Testing simple prompt (this may take time for cold start)...")
prompt = "Answer in one word: What is 2+2?"

try:
    response = llm.invoke(prompt)
    print(f"✓ Success! Response: {response}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)
