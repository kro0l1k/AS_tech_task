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
MAX_CONCURRENT_REQUESTS = 8  # used only for the legacy async fan-out path
REQUEST_TIMEOUT = 30         # seconds

# --- Parallelism ---
# When True, survey runs are dispatched via Anthropic's Message Batches API:
# one batch covers every (method × temperature × question × persona-chunk)
# combination in a phase, and the whole thing is processed server-side in
# parallel at 50% cost. Falls back to the async fan-out path when False.
USE_BATCH_API = True
BATCH_POLL_INTERVAL_SECONDS = 10   # starting poll cadence; backs off to 60s
BATCH_MAX_WAIT_SECONDS = 6 * 60 * 60   # hard ceiling per batch (6h of a 24h window)

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
