"""
Configuration for the LLM Survey Persona experiment.
"""
import os

# --- API ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"
TEMPERATURE = 1.0  # High temperature for response diversity

# --- Experiment ---
NUM_PERSONAS = 10 # 100
BATCH_SIZE = 5               # personas per API call
MAX_CONCURRENT_REQUESTS = 20
REQUEST_TIMEOUT = 30  # seconds

# --- Output ---
RESULTS_DIR = "results"
CACHE_DIR = "cache"
