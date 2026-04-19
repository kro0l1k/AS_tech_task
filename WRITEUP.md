# LLM Survey Personas — Writeup

## The Problem

Can LLMs simulate a representative panel of humans well enough to reproduce real public opinion distributions? The challenge decomposes into four sub-problems:

1. **What to condition the model on** — demographics, values, narrative, voting history, media diet?
2. **How to elicit authentic responses** rather than stereotype-driven or safety-trained defaults.
3. **How to align panel marginals with the target population** so aggregate answers aren't swamped by sampling noise.
4. **How to measure success rigorously** across methods, temperatures, and populations.

## Objective

Build 100 LLM personas of a **chosen target population** — selectable via `--population` — and ask them 13 single-select survey questions drawn from recent Pew / Gallup / Quinnipiac / KFF polls with known ground-truth distributions. Evaluate five different persona-construction methods.

Two run modes:

- **Default** — single unified poll at the configuration that won prior sweeps (`value_anchored @ T=0.3`). One method, one temperature, all 13 questions, one batch. Produces a poll-style per-question report of predicted vs. ground-truth distributions.
- **`--model_selection`** — the full 4-phase pipeline. Train/val/test split over questions, method selection on train, temperature sweep on val, held-out test evaluation, plus an R³ scatter of the three behavioural dials per test question.

Beyond the single-choice answer, we elicit **three behavioural signals per response** on continuous [0, 1] scales:

1. **`post`** — likelihood of posting on social media or publicly expressing the opinion (0 never, 1 always).
2. **`argue`** — likelihood of pushing back against opposing views (0 never, 1 always).
3. **`debate`** — how often the persona actually debates this issue with people offline (0 never, 1 every day).

These convert the opinion snapshot into **edge weights for a contagion-style simulation** on a social-network graph. A persona's outgoing influence is scaled by `post` and `argue`; its offline spread by `debate`. Without behavioural weights every node broadcasts at the same rate and the simulation collapses back to uniform averaging — which does not reproduce observed online polarisation.

The useful artefact is therefore the **joint distribution of (opinion, engagement)** per subgroup, which a downstream graph model consumes to predict cascades, polarisation drift, and the gap between "what people believe" and "what a feed looks like."

## Populations

Selected via `--population`. Each is a `PopulationSpec` grounded in Census / ACS / Pew / BLS data.

| Name | Target | Party split | Frame |
|---|---|---|---|
| `college_educated` (default) | Bachelor's+ adults, ages 22–65, USA | 58 / 37 / 5 | Skews urban, higher income |
| `seniors_south` | 65+ in FL / AL / LA / TX | 36 / 55 / 9 | State weights from Census senior counts |
| `general_us` | National adult sample | 45 / 43 / 12 | Matches the frame Gallup / Pew weight to |

**Why three populations rather than one narrow cohort?** The earlier version of this project used a single focused cohort (28-year-old college grads). That design isolated within-group variance but made every national ground-truth comparison structurally mismatched — the "liberal shift" on every partisan question was a known cohort property rather than a method miss. Supporting `general_us` explicitly gives an apples-to-apples comparison against national polls; `college_educated` and `seniors_south` remain available as subgroup probes where the deltas from national polls are *expected* and informative.

## Post-stratification

Independent marginal sampling from Census distributions produces panels whose empirical marginals drift from the target by several percentage points at n=100 — enough to swamp the method / temperature signal. The panel sampler now runs:

1. **Oversample 6×** — draw 600 candidates from the population spec.
2. **Greedy swap** — compute the panel's total variation distance against target marginals across (party, race, education, age-bucketed-to-spec). At each step, find the swap (panel member out, pool member in) that reduces TVD the most. Stop when no swap improves.
3. **Verify** — print `[poststratify] marginal TVD X → Y` and `max marginal drift vs target: Z.Zpp` in the run header.

In practice, TVD converges to **0.0 pp drift** across tracked dims on all three populations at n=100 within a few hundred swaps. One subtle bug caught here: the post-stratification age keys originally used the persona's narrative age bracket (`18-29 / 30-44 / 45-59 / 60+`) while the target marginals used the spec's age buckets (e.g. `22-34 / 35-44 / 45-54 / 55-65`). Zero key overlap made the age dim TVD pin at ~1.0. Fixed by bucketing the persona's raw age to the spec's bucket definition at comparison time.

This single change is the largest lever for aligning aggregate answers with ground truth — larger than any method or temperature effect we observed.

## Persona construction

Every persona carries:

- **Demographics** — age, gender, race, education, household income, region, community type.
- **State** — when the population spec constrains it (e.g. `seniors_south`).
- **Industry / employment status** — `employed_<industry>`, `retired_former_<industry>`, `student`, or `unemployed`. Industry distributions come from BLS CPS 2023, conditional on education. Status is age-conditioned: no 21-year-old retirees; ~62% of 65-year-olds are retired; student status largely confined to ages ≤ 26.
- **Work tier** — knowledge / services / physical, derived from industry.
- **Party + political basket** — one of five identities (`progressive`, `neo_liberal`, `neo_conservative`, `maga`, `alt_right`), sampled conditional on party lean.
- **Voting history** — concrete per-cycle records, e.g. "voted Trump 2024, Trump 2020, Trump 2016"; age-gated to cycles the persona was eligible for (`could_vote_2020` requires age ≥ 22, `could_vote_2016` requires age ≥ 26).
- **Top-5 information diet** — 3 from the basket's source pool + 2 from a general pool. Sampled **weighted by audience size**, not uniformly.

### Audience-weighted media sampling

Uniform sampling from a basket pool of ~10 sources gives Pod Save America and Chapo Trap House equal probability — which mis-weights the actual information environment by orders of magnitude. Each source now has an integer weight 1–10 calibrated from Nielsen cable prime-time viewership, Edison Podcast Consumer rankings, paid newspaper subscriptions, YouTube subscriber counts, and social-reach estimates. Sampling is weighted-without-replacement (Efraimidis-Spirakis: `key = U^(1/w)`, take top-k).

Empirically, at n=100 on `general_us`:

- **MSNBC** appears in ~67% of progressive diets (was ~30% uniform)
- **Fox News** appears in ~44% of MAGA diets
- **NPR** appears in ~46% of neo_liberal diets
- Niche outlets (Chapo Trap House, Sebastian Gorka's show, Jacobin) appear in ~12–17% of the relevant baskets — present but not dominant

This matches the real long-tail structure of political news consumption far better than uniform draws.

## Methods

| # | Method | Conditions on | Hypothesis |
|---|---|---|---|
| 1 | `demographic_baseline` | Age, gender, race, education, income, region, community, industry, party, basket, voting history, top-5 media | Null: demographics + partisan signals + media diet are sufficient |
| 2 | `narrative_backstory` | Biographical paragraph woven around the same attributes | Lived experience grounds the persona; reduces stereotype defaults |
| 3 | `value_anchored` | Moral-foundations scores + media diet + partisan context, demographics dropped | Values are more proximally causal than demographics |
| 4 | `distribution_aware` | Demographic block + meta-prompt: "you are respondent N of 100; minority views must be represented" | Leverages the model's aggregate knowledge of opinion distributions |
| 5 | `cognitive_deliberation` | Demographic block + reasoning cues, plus a mandatory `<thinking>` block | Chain-of-thought before letter choice beats pattern-matching demographics |

Every method prepends a **partisan-context block** at the top of the description showing party + basket + voting history — so the partisan signal isn't buried mid-profile and the model doesn't have to guess it from media diet alone.

## Prompt design

Earlier iterations used a strict "commit to one position — no hedging" rule. That rule caused **intensity collapse** on non-identity questions: the model read the directive as "pick an extreme" even when ground truth was in the middle (climate_local_impact regressed from 0.002 → 0.06 JSD; ai_work_usage from 0.04 → 0.12 JSD after that rule was added). Soft-intensity language caused the opposite bias on identity questions.

The current system prompt is organised around three rules:

1. **BE TRUE TO THE PERSON** — express the persona's genuine view. If they truly don't know, picking "don't know" is correct. If they're content with the status quo, picking "kept the same" is correct. Do not force extremes, do not commit when uncertain.
2. **PARTISAN IDENTITY IS IMPORTANT CONTEXT, NOT A DICTATOR** — real people hold unexpected positions on individual issues. Voting history anchors the baseline; individual questions can deviate.
3. **YOUR OWN MODEL PRIORS ARE NOT THE PERSONA** — ignore the model's defaults (moderate, libertarian, pro-establishment) when they conflict with what the persona would actually say.

### `<thinking>` deliberation block

Every response begins with a `<thinking>3–5 sentences</thinking>` block containing what the persona actually weighs, any uncertainty, which direction their gut leans. This pre-committal chain-of-thought:

- surfaces the reasoning so the letter choice is grounded rather than a reflexive pattern-match;
- legitimises "I genuinely don't know" / "a middle option fits" answers that a forced-choice regime suppresses;
- gives qualitative traces for debugging — we can see *why* a persona picked what it did.

Response format:

```
1. <thinking>…</thinking>
[LETTER] | post=X.XX argue=X.XX debate=X.XX
```

**Parser note.** Letter-extraction has to strip the `<thinking>` block before scanning, or bare A/B/C letters inside deliberation text ("option A would mean…") hijack the regex. `survey.py` extracts thinking content into a separate field, then letter-scans the residual only.

### Preamble jitter

Question framing is jittered per batch from a pool of 7 preambles ("Quick read:", "Short poll item:", "One-question check:", etc.). Question text itself is never altered, but the framing wrapper varies. This blunts pattern-matching on canonical Pew/Gallup phrasings where the model may have seen the exact toplines in training data.

## Sample persona description (method = `value_anchored`)

Persona 0 from `sample_population_panel("general_us", n=100, seed=42, poststratify=True)` — a 41-year-old white male in a rural Midwest community, high-school graduate, $30–50k, MAGA basket, trade-industry background.

```
=== PARTISAN CONTEXT ===
Party: Republican (lean)
Political basket: maga
Voting history: voted Trump 2024, Trump 2020, Trump 2016

=== VALUES ===
Caring for others / preventing harm: somewhat important
Fairness, justice, equal rights: somewhat important
Loyalty to community and country: a core conviction
Respect for authority, tradition, order: a core conviction
Moral purity / upholding sacred values: very important
Personal freedom from government control: a core conviction

=== INFORMATION DIET ===
Fox News, The Joe Rogan Experience, Truth Social, local TV news, Facebook
```

Note: `value_anchored` drops the raw demographics. The partisan-context block carries the partisan signal, the moral-foundations block carries the values, and media diet carries the information environment — which together are sufficient to reproduce within-group opinion shape on most questions better than demographic labels do.

## Example model output

Raw response from one batch, `cognitive_deliberation` method, climate question, 5 personas. Shown unedited — thinking blocks included:

```
1. <thinking>She's a retired nurse in Alabama who voted Trump all three cycles.
Her Fox News and local news diet frames climate change as overblown — but she
saw the hurricanes get worse in her own state. She'd say humans contribute
some but not a great deal; she's not an activist either way.</thinking>
B | post=0.08 argue=0.15 debate=0.12

2. <thinking>Progressive basket, NPR + MSNBC + The Daily listener, urban
college grad. For her this is settled science — "a great deal" is the only
answer that matches how her sources frame it. High engagement but not
activist-level.</thinking>
A | post=0.55 argue=0.50 debate=0.35

…
```

The `<thinking>` block grounds the letter choice in the persona's media diet and voting history; the behavioural numbers co-vary with engagement (the retired, quiet Fox viewer scores low; the urban progressive scores mid-high).

## Evaluation

### Metrics

- **JSD** — Jensen-Shannon divergence, bounded [0, 1], log base 2. Primary metric.
- **TVD** — total variation distance (L1/2).
- **MAE** — mean per-option percentage-point error.
- **Plurality accuracy** — does the top predicted option match ground truth?
- **Chi-squared p-value** — distributional-difference significance.

### Default pipeline

```
1. Post-stratified 100-persona panel from the chosen population.
2. value_anchored @ T=0.3 across ALL 13 questions in a single batch.
3. Print per-question predicted vs. ground-truth distributions.
4. Flag ✗ where predicted-top disagrees with ground-truth-top.
```

One Anthropic Message Batches submission per run, ~3–5 min wall-clock.

### Model-selection pipeline (`--model_selection`)

Questions are split 10 / 2 / 1 (train / val / test):

```
Phase 1 — all 5 methods on TRAIN at T=1.0                 (500 requests)
Phase 2 — rank methods by mean JSD vs ground truth
Phase 3 — temperature sweep {0.3, 0.6, 0.8, 1.0} on VAL    (800 requests)
         best method only
Phase 4 — held-out TEST at the winning (method, T)         (100 requests)
Phase 5 — behavioural rollup + R³ scatter per test question
```

Each phase is a single Message Batches submission; results stream back and demultiplex by `custom_id`. The default pipeline exists because Phase 1+3 cost ~90% of the total API calls, and once the winner was identified on prior sweeps a single unified poll is a much cheaper default.

## Prior model-selection results

From earlier sweeps on the focused 28-year-old college-grad cohort (8 questions, n=100), `value_anchored` narrowly beat every demographic-heavy method on TRAIN (JSD 0.0291 vs 0.0304 for distribution_aware, 0.0329 for demographic_baseline, 0.0379 for narrative_backstory). Plurality accuracy told a sharper story: `value_anchored` hit 5/6 TRAIN questions (83%) vs 4/6 (67%) for everyone else. On VAL, T=0.3 won over T=1.0 by a narrow margin — a U-shape where the interior temperatures (0.6, 0.8) were ~4× worse in JSD.

Interpretation: at T=0.3 the model commits to each persona's cohort-centroid answer; at T=1.0 sampling noise produces diversity; in between you get neither regime. Within-persona reasoning already provides response diversity — extra sampling stochasticity on top degrades rather than helps.

The default single-pipeline config (`value_anchored @ T=0.3`) is that winner. Re-running the full 4-phase sweep on the current populations with the 13-question corpus is the next validation milestone — see Limitations.

## Behavioural signals: minority-loud-majority-quiet

On every partisan question the cohort-minority position posts ~2× more, argues ~2–2.5× more, and debates ~1.7× more than the cohort-majority position. Representative slice from the `college_educated` run on `gun_sales_laws`:

```
  More strict         n=61  post=0.37   argue=0.35   debate=0.30
  Kept as they are    n=23  post=0.26   argue=0.33   debate=0.30
  Less strict         n=16  post=0.67   argue=0.75   debate=0.55   ← minority, LOUDEST
```

Re-weighting the belief distribution by post-likelihood:

```
  P_posted(More strict)      ∝ 0.61 × 0.37 = 0.226  → 52%
  P_posted(Kept as they are) ∝ 0.23 × 0.26 = 0.060  → 14%
  P_posted(Less strict)      ∝ 0.16 × 0.67 = 0.107  → 25%  ← up from 16%
```

A naive scrape of this cohort's social posts would read as 52% pro-control / 25% anti-control — a substantially different distribution from the 61/16 belief split. "Gun-control is contested ~2-to-1" on social media is partially a willingness-to-post artefact, not a belief measurement.

This is the forcing function the downstream graph-sim needs. With uniform broadcast weights you get smooth averaging dominated by the modal opinion. With empirical weights, the minority broadcasts ~2× louder per node — exactly the structural condition under which a minority view can dominate feed composition while holding 12–16% of belief share.

## Limitations

1. **The current 13-question, 3-population design has not been re-swept with `--model_selection` at full scale.** The default config (`value_anchored @ T=0.3`) is inherited from the prior 8-question focused-cohort sweep. The 10/2/1 train/val/test split on 13 questions is available via `--model_selection` and should be run on `general_us` to confirm the winner re-validates.
2. **Single-question VAL / TEST in the 13-question split.** 2 val + 1 test questions means temperature ranking and test JSD are noisy. Growing the question corpus to ≥ 20 is the obvious fix.
3. **Behavioural signals are self-reported by the simulator, not validated.** No public poll asks "what fraction of X-voter Y-believers would post about Z." The values are internally consistent (engaged personas high on all three, quiet low) but absolute calibration is unverified.
4. **Behavioural and letter outputs share a prompt.** Because post/argue/debate are generated on the same line as the letter, the two may be non-independent. A two-call ablation (letter first, then behavioural follow-up) would separate them.
5. **Training-data contamination for widely-polled questions.** The model may recall headline numbers rather than simulating from first principles. Preamble jitter helps but cannot eliminate this. Climate is the cleanest suspect: the cohort-liberal lean explains part of the overshoot on human-caused climate, but the magnitude is consistent with training-data pressure toward pro-consensus framing.
6. **Aggregate-only evaluation.** Distributional accuracy can come from either good individual simulation or cancelling errors. We cannot distinguish without subgroup or counterfactual probes.
7. **Batch contamination risk.** 5 personas answered in one call may be artificially diversified by the model. We see persona-specific media-diet grounding in the thinking blocks, which suggests faithful conditioning, but a solo-call ablation would confirm.
8. **Post-stratification matches marginals, not joints.** A panel can match target party / race / education / age marginals while under-representing specific joint cells (e.g. high-income rural Democrats). For n=100 the joint-cell sparsity is unavoidable.
9. **Behavioural-prevalence vs identity-opinion mismatch.** Questions like `ai_work_usage` ("what fraction of your work uses AI") depend on cohort composition, not opinion. Scoring a college-grad panel against a national workforce survey produces large but uninformative JSD. `general_us` is the right comparison frame for these; we leave them in the question set because they're useful on subgroup panels.

## Code

```
config.py          API + experiment knobs (model, temperatures, batch size)
ground_truth.py    13 survey questions + published distributions + source URLs
demographics.py    cohort sampling, post-stratification (greedy swap),
                   industry + BLS status conditioning, audience-weighted
                   media, voting-history generator
personas.py        five persona-description methods + shared system prompt
                   with <thinking> block and authentic-tone rules
survey.py          Message Batches runner (50% cost, server-side parallel),
                   <thinking>-aware parser, preamble jitter
metrics.py         JSD / TVD / MAE / χ² / behavioural stats
analysis.py        tables, charts, R³ scatter
main.py            CLI — default unified poll or --model_selection pipeline
WRITEUP.md         this file
README.md          short usage doc
```

Results land in `results/`. Cached batch outputs in `cache/`. Every live run's full console transcript is archived to `logs/output_<population>_<timestamp>.txt`; `--dry-run` skips the log write (dry runs don't hit the API and aren't interesting to archive).

A `--dry-run` flag exercises the full pipeline with simulated responses so reviewers can run everything end-to-end without an API key.
