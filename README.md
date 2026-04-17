# LLM Survey Persona Experiment

Build 100 LLM personas of a narrow cohort — 28-year-olds with a college degree — and ask them real survey questions. Compare five persona-construction methods, sweep temperature, and measure how well the sampled distribution matches published polling ground truth.

Each response also carries three [0, 1] behavioral dials — willingness to **post**, willingness to **argue**, and **debate** frequency offline. These feed a downstream graph-based opinion-dynamics simulation where the numbers drive edge weights and node activity.

## What the pipeline does

1. Sample a panel of 100 personas (shared across every run).
2. **Phase 1** — run all 5 methods on the TRAIN questions.
3. **Phase 2** — rank methods by mean JSD vs. ground truth.
4. **Phase 3** — temperature sweep `{0.3, 0.6, 0.8, 1.0}` on VAL, best method only.
5. **Phase 4** — held-out TEST evaluation at the winning (method, T).
6. Print behavioral rollups and save an R³ scatter of (post, argue, debate) per TEST question.

Requests are submitted via the Anthropic Message Batches API — one batch per phase, results streamed back and demultiplexed by `custom_id`.

## Methods

| # | Name | Idea |
|---|------|------|
| 1 | `demographic_baseline` | Explicit demographic attributes |
| 2 | `narrative_backstory` | Biographical paragraph |
| 3 | `value_anchored` | Moral foundations + media diet, demographics dropped |
| 4 | `distribution_aware` | Tell the model it's one draw from a population |
| 5 | `cognitive_deliberation` | Chain-of-thought before the letter |

## Questions

8 items across guns, climate (2), abortion (2), immigration, AI jobs, AI-at-work. Split 6 / 1 / 1 (train / val / test). Single-select, A–J letter response, option order randomized per persona to blunt position bias.

## Metrics

- **JSD** — Jensen-Shannon divergence (0 = identical)
- **TVD** — total variation distance
- **MAE** — mean per-option error in percentage points
- **Plurality accuracy** — does the top predicted option match ground truth?

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY='...'
```

## Run

```bash
python main.py                      # full experiment via batch API
python main.py --dry-run            # simulated responses, no API calls
python main.py --method value_anchored   # single method on all splits
python main.py --analyze-only       # re-analyze saved results
```

Results land in `results/`. Cached batch outputs in `cache/`.

## Layout

```
config.py          API + experiment knobs
ground_truth.py    survey questions and published distributions
demographics.py    cohort sampling
personas.py        the five persona methods
survey.py          batch / async runner, parser
metrics.py         JSD, TVD, MAE, chi-squared, behavioral stats
analysis.py        tables, charts, R³ scatter
main.py            CLI
WRITEUP.md         full write-up: design, results, limitations
```

## Notes

- Ground-truth distributions are national. The cohort (28, degreed) skews left of those, so partisan questions land liberal of published means by design — treated as a cohort-vs-national mismatch, not a method miss.
- Behavioral dials are self-reported by the LLM. Treat them as relative intensities, not calibrated probabilities.
- Temperature is clamped to [0, 1] by the API; the sweep grid stays in range.
