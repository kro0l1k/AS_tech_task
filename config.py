"""
Configuration for the LLM Survey Persona experiment.
"""
import os

# --- API ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"
TEMPERATURE = 1.0  # default — overridden in sweep

# Temperature sweep evaluated on VAL questions after method selection.
# Anthropic API clamps temperature to [0, 1]; grid must stay in-range.
TEMPERATURE_SWEEP = [0.3, 0.6, 0.8, 1.0]

# --- Experiment ---
NUM_PERSONAS = 100
BATCH_SIZE = 5               # personas per API call
MAX_CONCURRENT_REQUESTS = 8  # gentler concurrency to avoid 429/529 floods
REQUEST_TIMEOUT = 30         # seconds

# Any method / temperature with valid-response rate below this is excluded
# from ranking (prevents empty-distribution runs from looking like winners).
MIN_VALID_RATE = 0.80

# --- Panel focus ---
# Panel is a coherent slice: 28-year-olds with a degree.
# Vary everything else (gender / race / region / party / basket / income).
FOCUS_AGE = 28

# --- Output ---
RESULTS_DIR = "results"
CACHE_DIR = "cache"
