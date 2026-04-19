# LLM Survey Persona Experiment

Build 100 LLM personas of a chosen target population and ask them real survey questions. Compare five persona-construction methods, sweep temperature, and measure how well the sampled distribution matches published polling ground truth.

Each response also carries three [0, 1] behavioral dials — willingness to **post**, willingness to **argue**, and **debate** frequency offline. These feed a downstream graph-based opinion-dynamics simulation where the numbers drive edge weights and node activity.

## Pipelines

Two run modes, chosen by flag:

**Default — single unified poll** (no `--model_selection`):
1. Sample a 100-persona panel of the chosen population.
2. Post-stratify to match target marginals (party / race / education / age) to ≤ 0pp drift via greedy swap over an oversampled pool.
3. Run **one method at one temperature** across ALL 13 questions in a single batch.
4. Print a poll-style report: per-question predicted vs. ground-truth distributions, flagged ✗ when predicted-top disagrees with ground-truth-top.

The default method and temperature are `value_anchored @ T=0.3`, identified as the winner in prior model-selection sweeps.

**Model selection — full 4-phase pipeline** (`--model_selection`):
1. Post-stratified panel (same as default).
2. **Phase 1** — run all 5 methods on TRAIN questions.
3. **Phase 2** — rank methods by mean JSD vs. ground truth.
4. **Phase 3** — temperature sweep `{0.3, 0.6, 0.8, 1.0}` on VAL, best method only.
5. **Phase 4** — held-out TEST evaluation at the winning (method, T).
6. Behavioral rollup + R³ scatter of (post, argue, debate) per test question.

Requests go through the Anthropic Message Batches API — one batch per phase, results streamed back and demultiplexed by `custom_id`.

## Populations

Selected by `--population`. Each is a `PopulationSpec` grounded in Census / ACS / Pew / BLS data:

| Name | Target | Notes |
|---|---|---|
| `college_educated` (default) | Bachelor's+ adults, ages 22–65, across USA | 58% Dem / 37% Rep / 5% Ind. Skews urban, higher income. |
| `seniors_south` | 65+ in FL / AL / LA / TX | 36% Dem / 55% Rep / 9% Ind. State weights from Census senior counts. |
| `general_us` | National adult sample matching poll frames | 45/43/12 party split. Matches the frame Gallup / Pew weight to. |

Sampling is **post-stratified** after the raw draw: we oversample 6× into a pool, then greedy-swap panel members until empirical marginals on party, race, education, and age bucket match target distributions exactly (typically 0.0pp drift at n=100). This is the biggest single lever for aligning aggregate answers with ground truth.

## Persona construction

Every persona carries:

- demographics (age, gender, race, education, income, region, community)
- state (when the population spec constrains it)
- industry / employment status (`employed_<industry>`, `retired_former_<industry>`, `student`, `unemployed`) — drawn from BLS CPS 2023 distributions conditional on education; status conditioned on age (no 21-year-old retirees, 62% of 65-year-olds retired)
- work tier (knowledge / services / physical)
- party, political basket (progressive / neo_liberal / neo_conservative / maga / alt_right), **voting history** (e.g. "voted Trump 2024, Trump 2020, Trump 2016") — ages-gated to only include cycles the persona was eligible for
- top-5 information diet — 3 from the basket pool + 2 general, sampled **weighted by audience size** (Nielsen prime-time, podcast rankings, paid subs, social reach). Mass outlets dominate; niche outlets appear rarely.

## Methods

| # | Name | Idea |
|---|------|------|
| 1 | `demographic_baseline` | Explicit demographic attributes |
| 2 | `narrative_backstory` | Biographical paragraph |
| 3 | `value_anchored` | Moral foundations + media diet, demographics dropped |
| 4 | `distribution_aware` | Tell the model it's one draw from a population |
| 5 | `cognitive_deliberation` | Chain-of-thought before the letter |

Every method prepends a **partisan-context block** (party + basket + voting history) at the top of the description so the partisan signal isn't buried mid-profile.

## Prompt design

The shared system prompt emphasises authenticity over decisiveness:

- Express the persona's **genuine** view. If they truly don't know, picking "don't know" / "not sure" is correct. If they're content with the status quo, picking "kept the same" is correct. Do not force extremes or commit when uncertain.
- Partisan identity is **context, not a dictator** — real people can hold unexpected positions on individual issues.
- Ignore the model's own priors (moderate, libertarian, pro-establishment) when they conflict with the persona.
- Before picking a letter, the model writes a `<thinking>3–5 sentences of genuine deliberation</thinking>` block — what the persona actually weighs, any uncertainty, which direction their gut leans.

Response format:

```
1. <thinking>…</thinking>
[LETTER] | post=X.XX argue=X.XX debate=X.XX
```

Question framing is jittered per batch (7 preambles) to blunt pattern-matching on canonical Pew/Gallup phrasing. Question text itself is never altered.

## Questions

13 items across guns, climate (2), abortion, immigration, AI jobs, AI-at-work, healthcare, democracy, Israel/Palestine, marijuana, vaccines, taxes. When model selection is on, split 10 / 2 / 1 (train / val / test). Single-select, A–J letter response, option order randomized per persona to blunt position bias.

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
python main.py                                  # default: value_anchored @ T=0.3, college_educated
python main.py --population seniors_south       # 65+ in FL/AL/LA/TX
python main.py --population general_us          # national adult frame
python main.py --model_selection                # full 4-phase pipeline
python main.py --dry-run                        # simulated responses, no API calls
```

### Flags

- `--population {college_educated | seniors_south | general_us}` — which underlying group the personas model. Default: `college_educated`.
- `--model_selection` — opt in to the full 4-phase pipeline (all methods on TRAIN → pick best → temp sweep on VAL → TEST at best config). Off by default; the default run uses `value_anchored` @ T=0.3 directly. Off saves ~90% of API calls.
- `--method NAME` — restrict the Phase-1 sweep to a single method. Requires `--model_selection`.
- `--dry-run` — simulate responses locally; no API key needed. Does not write a log file.
- `--seed INT` — controls panel sampling and persona construction. Default 42.

Results land in `results/`. Cached batch outputs in `cache/`. Every live run's full console transcript is archived to `logs/output_<population>_<timestamp>.txt` (dry runs skipped).

## Layout

```
config.py          API + experiment knobs
ground_truth.py    survey questions and published distributions
demographics.py    cohort sampling, post-stratification, industry + media weights
personas.py        the five persona methods + shared system prompt
survey.py          batch / async runner, <thinking>-aware parser
metrics.py         JSD, TVD, MAE, chi-squared, behavioral stats
analysis.py        tables, charts, R³ scatter
main.py            CLI
WRITEUP.md         full write-up: design, results, limitations
```

## Notes

- Ground-truth distributions are national. The `college_educated` and `seniors_south` populations deviate from national polls by design — treat those deltas as cohort-vs-national mismatches, not method misses. `general_us` is the apples-to-apples comparison.
- Behavioral dials are self-reported by the LLM. Treat them as relative intensities, not calibrated probabilities.
- Temperature is clamped to [0, 1] by the API; the sweep grid stays in range.
