"""
Configuration for the LLM Survey Persona experiment.
"""
import os

# --- API ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"
TEMPERATURE = 1.0  # default — overridden in sweep

# Temperature sweep evaluated on VAL questions after method selection.
# Grid chosen to span low/medium/high; winner applied to TEST.
TEMPERATURE_SWEEP = [0.5, 0.7, 1.0, 1.3]

# --- Experiment ---
NUM_PERSONAS = 20
BATCH_SIZE = 5               # personas per API call
MAX_CONCURRENT_REQUESTS = 20
REQUEST_TIMEOUT = 30  # seconds

# --- Panel focus ---
# Panel is a coherent slice: 28-year-olds with a degree.
# Vary everything else (gender / race / region / party / basket / income).
FOCUS_AGE = 28

# --- Output ---
RESULTS_DIR = "results"
CACHE_DIR = "cache"
