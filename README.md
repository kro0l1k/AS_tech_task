# LLM Survey Persona Experiment

Can 100 LLM-generated personas reproduce the opinion distribution of US adults? This project tests five different methods for constructing LLM personas, runs them through 7 real survey questions with known ground-truth distributions, and evaluates which method best captures actual public opinion.

## Methods

| # | Method | Hypothesis |
|---|--------|-----------|
| 1 | **Demographic Baseline** | Explicit demographic attributes activate the model's learned correlations with opinions |
| 2 | **Narrative Backstory** | Rich biographical context grounds personas in lived experience, reducing stereotype-driven defaults |
| 3 | **Value-Belief Anchoring** | Moral foundations predict opinions more directly than demographics — cut out the middleman |
| 4 | **Distribution-Aware Ensemble** | Telling the model about its role in a population sample leverages its aggregate knowledge |
| 5 | **Cognitive Deliberation** | Chain-of-thought reasoning produces more authentic, less reflexive responses |

## Survey Questions

7 questions from Gallup/Pew with known response distributions: gun laws, climate change, marijuana legalization, immigration levels, government healthcare, abortion, death penalty.

## Evaluation Metrics

- **Jensen-Shannon Divergence (JSD)**: Information-theoretic distribution similarity (0 = identical)
- **Total Variation Distance (TVD)**: Maximum probability mass that needs moving
- **Mean Absolute Error**: Average per-option error in percentage points
- **Plurality Accuracy**: Does the most-chosen option match ground truth's top answer?

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Test the full pipeline with simulated responses (no API key needed)
python main.py --dry-run

# Run the real experiment (requires Anthropic API key)
export ANTHROPIC_API_KEY='your-key-here'
python main.py

# Run a single method
python main.py --method narrative_backstory

# Analyze previously saved results
python main.py --analyze-only
```

## Project Structure

```
config.py          — API and experiment configuration
ground_truth.py    — Survey questions with real polling distributions
demographics.py    — Census-based demographic sampling
personas.py        — Five persona creation methods
survey.py          — Async survey runner with option randomization
metrics.py         — JSD, TVD, MAE, chi-squared evaluation
analysis.py        — Comparison tables and matplotlib charts
main.py            — CLI entry point
```

## Key Design Decisions

- **Option order randomization**: Options are shuffled per question-persona pair to mitigate LLM position bias
- **Shared demographic panel**: All methods use the same sampled demographics for fair comparison
- **Independent marginal sampling**: Demographics are sampled from marginal distributions (not joint), trading some realism for simplicity
- **Temperature = 1.0**: High temperature maximizes response diversity, which matters for distributional fidelity
- **Forced single-letter output**: Constrains responses for reliable parsing; CoT method allows reasoning then extracts final letter
