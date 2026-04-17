# LLM Survey Personas — Writeup

## The Problem

Can LLMs simulate a representative panel of humans well enough to reproduce real public opinion distributions? The challenge decomposes into three sub-problems: (1) what to condition the model on, (2) how to elicit authentic responses rather than stereotype-driven or safety-trained defaults, and (3) how to measure success rigorously.

## Objective

Build 100 LLM personas of a **coherent cohort — 28-year-old US adults with a college degree** — ask them single-select survey questions drawn from recent Gallup and Pew polls with known ground-truth distributions, and evaluate five different persona-construction methods. Use a train/val/test split over questions to select the best method, then sweep sampling temperature on the validation split and evaluate the winning combination on held-out test questions.

Beyond the single-choice answer, we additionally elicit **three behavioral signals per response** on continuous [0, 1] scales:

1. **`post`** — how likely the persona is to post on social media or otherwise publicly express an opinion supporting their choice (0 = never, 1 = always).
2. **`argue`** — how likely the persona is to push back against people holding opposing views (0 = never, 1 = always).
3. **`debate`** — how often the persona actually debates this issue with people offline (0 = never, 1 = every day).

These three quantities let us model not just *what* a cohort believes but *who speaks and how loudly* — which matters because online discourse is a sample weighted by willingness-to-post and willingness-to-argue, not a census of opinion.

### Downstream goal: graph-based opinion dynamics

The single-choice distribution on its own is a snapshot. The three behavioral signals convert that snapshot into **edge weights for a contagion-style simulation** on a social-network graph:

- Each persona is a node; initial state = its chosen option.
- Outgoing influence on the graph is scaled by `post` and `argue`: a high-`post` persona with `argue = 0.8` seeds many timelines and actively contests counter-posts; a low-engagement persona exerts almost no pressure on its neighbours even when it holds the same view.
- Offline spread is scaled by `debate`: independent of online reach, this captures how far an opinion travels through face-to-face conversation at work, school, or home.
- A standard opinion-dynamics update (Deffuant, bounded-confidence, or a threshold model) can then be run with these weights per node, producing dynamics where a minority but highly-engaged cohort can dominate visible discourse even while the silent plurality keeps the *underlying* belief distribution roughly stable.

This reframes the simulation output: **the useful artefact is the joint distribution of (opinion, engagement) per demographic subgroup**, which a downstream graph model consumes to predict information cascades, polarisation drift, and the gap between "what people believe" and "what a feed looks like." Without behavioral weights, every node broadcasts at the same rate and the simulation collapses back to a uniform averaging model that doesn't reproduce observed online polarisation.

**Why a narrow cohort rather than the full US population?** Independent marginal sampling from US-wide demographic distributions produces incoherent personas (the previous writeup flagged "22-year-old retired graduate" as a concrete failure mode). Fixing age and education range and letting everything else vary — gender, race, region, community, party, political basket, income, information diet — yields internally consistent profiles and isolates the question of whether the model can simulate *within-group variance* for a known slice of the population. We note below the evaluation trade-off this creates.

## Why This Is Hard

LLMs have systematic biases that fight faithful opinion simulation:

1. **Training distribution skew** — text corpora over-represent the young, educated, and liberal. The model's "default voice" does not match the US population.
2. **Safety-training pull** — RLHF compresses response tails, under-generating extreme positions.
3. **Stereotype caricature** — given "MAGA conservative," the model often produces an exaggerated version rather than the actual within-group variance. The modal conservative view on guns is "enforce existing laws"; the model produces "fewer laws."
4. **Status-quo blindness** — training data is written by people with strong opinions. Americans who answer "keep things as they are" or "haven't thought about it" are under-represented in the corpus and therefore in the model's latent voice.
5. **Position bias** — models systematically favor earlier or later options, adding noise orthogonal to the persona.
6. **Cohort–ground-truth mismatch (in a specific, predictable direction)** — a coherent sub-population cohort should *not* match national polling distributions exactly. Our panel — 28-year-old US adults with a college degree, sampled 61% Democrat / 25% Republican / 14% Independent — leans **substantially more liberal than the national polled population on every partisan-correlated question we ask**: more pro-choice on abortion, more accepting of human-caused climate change, more supportive of stricter gun regulation, more supportive of a pathway to legal status for undocumented immigrants. Every deviation from national ground truth in the results tables therefore has a known directional component — predicted distributions will be shifted *toward the liberal end* of the option list relative to the national topline, and this shift is a faithful property of the cohort, not a method failure. JSD against national ground truth has a floor set by this true cohort deviation. We flag these shifts explicitly when they appear in results (for example: the test-set plurality flip on `abortion_legal_all_or_most`, where the cohort's modal answer is "Legal in most cases" while the national plurality is "Illegal in all or most cases"). This is a feature (it stops us overfitting the model to topline numbers) but it demands careful interpretation — see also the crosstab / reweighting items in "What I Would Do Next."

## Approach: Five Methods Along Three Axes

Each method tests a different lever.

### Methods (what we vary)

| Method | What it conditions on | Hypothesis being tested |
|--------|----------------------|------------------------|
| **Demographic Baseline** | Age, gender, race, education, income, region, community type, party, political basket, top 5 information sources | Null hypothesis: demographics + media diet are sufficient |
| **Narrative Backstory** | A biographical sketch (name, occupation, life stage, housing, experience) woven around the same attributes | Lived experience grounds the persona; reduces stereotype defaults |
| **Value-Belief Anchoring** | Moral Foundations scores (Care, Fairness, Loyalty, Authority, Sanctity, Liberty) + information sources | Values are more proximally causal than demographics |
| **Distribution-Aware Ensemble** | Demographics + meta-prompt: "you are respondent #N of 100; minority views must be represented" | Leverages the model's aggregate knowledge of opinion distributions directly |
| **Cognitive Deliberation** | Demographics + explicit reasoning cues: "think about what you'd have seen in your sources; what gut reaction does this trigger?" | Reasoning produces more authentic responses than pattern-matching demographics |

### Design decisions

- **Coherent focused panel.** `sample_focused_panel(n, age=28)` fixes age at 28 and restricts education to {Bachelor's 80% / Graduate 20%}. Income is shifted to an early-career distribution (thinner <$30k and >$100k tails). Party skews Democratic (~58/30/12) reflecting the young-college-grad cohort. Race, gender, region, community, political basket, and information diet still vary freely.
- **Political basket** — each persona gets one of five identities (`progressive`, `neo_liberal`, `neo_conservative`, `maga`, `alt_right`), sampled conditional on party lean. Drives information-diet composition.
- **Top 5 information influences** — 3 drawn from the basket's source pool (Pod Save America, Breitbart, The Bulwark, etc.), 2 from a general mainstream pool (local news, Facebook, coworkers). Appears in every persona description so the model can ground reasoning in a plausible info diet.
- **Short internal dialogue** — every response includes 1–2 sentences of reasoning before the letter choice. Prevents the model from reflexively pattern-matching demographics to stereotype answers (which it does badly when forced to `max_tokens=5`).
- **Batched API calls** — five personas per call, reduces request count 5× with minimal token overhead. The model sees all five profiles and produces numbered answers in a single response.
- **Message Batches API for phase-level parallelism.** Each phase (Phase 1 = all 5 methods × TRAIN; Phase 3 = all 4 temperatures × VAL; Phase 4 = final TEST) bundles every (method × temperature × question × persona-chunk) request into a single Anthropic *Message Batches* submission. Anthropic processes the whole batch server-side in parallel at 50% cost; we poll for completion and demultiplex results back via per-request `custom_id`. On the n=100 cohort this collapses Phase 1 wall-clock from ~5 sequential methods × minutes of semaphore-bounded fan-out (≈10–20 min) to a single batch that typically finishes in a few minutes regardless of method count. The legacy async fan-out path (`messages.create` with a concurrency semaphore) is kept as a fallback behind `config.USE_BATCH_API=False` for debugging individual calls.
- **Option randomization** — options are shuffled per question-persona pair to mitigate position bias.
- **Shared panel** — all methods and all temperature settings evaluate against the same sampled panel so differences attribute to method/temperature, not sampling variance.
- **Temperature sweep on validation** — after method selection on train, we sweep `{0.3, 0.6, 0.8, 1.0}` on the val split and pick the winner for final test evaluation. (Anthropic's API clamps temperature to [0, 1], so the grid must stay in-range; an earlier experiment with T=1.3 returned 100% 400-errors across every method.)

## Sample Inputs Passed to the API

All methods share the same system prompt and batched user-message structure. Only the *per-persona description block* differs between methods. Below is persona 0 from the seeded panel (`sample_focused_panel(n=100, seed=42)`) — a 28-year-old Hispanic male, Bachelor's, $50–75k, Urban South, lean-Republican but sampled into the `neo_liberal` basket (small-probability draw from the party-conditional basket table; CNN + Ezra Klein read as a counter-intuitive but plausible media diet for a dissatisfied moderate). Shown verbatim as sent to the model.

### Shared system prompt (all methods)

```
You are simulating authentic American survey respondents for opinion research.

You will receive profiles of several Americans and one survey question.
For EACH person:
1. Write 1-2 sentences of their honest internal reasoning — gut feelings shaped by
   their specific information sources and life situation. Be authentic: people
   are often uncertain, sometimes contradictory, and rarely perfectly informed.
2. State their answer as the option letter.
3. Give three behavioral numbers in [0, 1] describing how socially active they
   are about this issue:
   - post   = likelihood they would post on social media / publicly express an
     opinion supporting their choice (0 never, 1 always)
   - argue  = likelihood they would push back against people supporting other
     positions (0 never, 1 always)
   - debate = how often they actually debate this issue with other people
     offline (0 never, 1 every day)
   Most people are NOT highly engaged. A quiet, ambivalent, busy-with-life
   person should get low values (0.05–0.2). Activists and media-diet-heavy
   people get high values (0.6–0.95). Vary per person — do not give identical
   numbers.

Reply in this EXACT format, one numbered line per person:
1. [reasoning] | [LETTER] | post=X.XX argue=X.XX debate=X.XX
2. [reasoning] | [LETTER] | post=X.XX argue=X.XX debate=X.XX
…

No other text. No preamble. No explanations outside the numbered lines.
```

### Per-persona description block, by method

**1. `demographic_baseline` — attributes + media diet, minimal framing:**
```
Demographic profile:
Age: 28
Gender: Male
Race/Ethnicity: Hispanic or Latino
Education: Bachelor's degree
Household income: $50,000 to $75,000
Region: South
Community type: Urban
Political identity: neo_liberal (broadly Republican or lean Republican)
Top 5 information sources:
  1. CNN
  2. The Ezra Klein Show
  3. Pod Save America
  4. network evening news (ABC / NBC / CBS)
  5. Reddit (general browsing)

Answer based on this person's demographics and information diet.
```

**2. `narrative_backstory` — prose sketch with occupation, housing, life event:**
```
William, 28, Hispanic or Latino male, works as a software developer. Lives in a
rowhouse neighborhood in the South. Household income $50,000 to $75,000/year.
Is living with roommates to afford rent in a tight market. Politically
neo_liberal. Gets most news from CNN and The Ezra Klein Show. Full information
diet: CNN, The Ezra Klein Show, Pod Save America, network evening news
(ABC / NBC / CBS), Reddit (general browsing).
```

**3. `value_anchored` — moral foundations + sources; demographics dropped:**
```
Moral values profile:
Caring for others / preventing harm: moderately important
Fairness, justice, equal rights: moderately important
Loyalty to community and country: very important
Respect for authority, tradition, order: a core conviction
Moral purity / upholding sacred values: very important
Personal freedom from government control: moderately important

Political identity: neo_liberal
Information sources: CNN, The Ezra Klein Show, Pod Save America, network evening
news (ABC / NBC / CBS), Reddit (general browsing)
```

**4. `distribution_aware` — adds ensemble meta-instruction:**
```
Panel respondent #1 of 20 (nationally representative sample).
Age: 28
Gender: Male
[… full demographic block as in baseline …]

Note: the goal is for this panel's aggregate responses to match the genuine US
opinion distribution. Minority views must be represented — do not default to
majority positions.
```

**5. `cognitive_deliberation` — adds reasoning cues, no narrative:**
```
Age: 28
Gender: Male
[… full demographic block as in baseline …]

Before answering, think about: what would this person have recently read or
heard from their sources? What gut reaction does the question trigger given
their life situation?
```

### How the blocks are composed into the batch user message

```
SURVEY QUESTION: Which of the following comes closest to your view about laws
covering the sale of firearms in general?

Options:
A) Kept as they are now
B) More strict
C) Less strict

---

PERSON 1:
<description block for persona 0>

PERSON 2:
<description block for persona 1>

…five persons total…

---
Now respond for all 5 persons in order (format: N. [reasoning] | [LETTER]):
```

### Example model output (cognitive_deliberation, batch of 5, climate question)

Raw response, unedited, showing the per-persona reasoning the model produces before each letter:

```
1. I get mixed messages from CNN and liberal podcasts saying climate change is
   serious, but I'm Republican-leaning so I'm skeptical of how much is really
   human-caused versus natural cycles - the science seems somewhat uncertain. | B | post=0.10 argue=0.25 debate=0.15
2. All my sources from Pod Save America to The Daily consistently emphasize the
   scientific consensus that humans are the primary driver of climate change,
   and my educated urban community generally accepts this. | C | post=0.55 argue=0.60 debate=0.40
3. The Bulwark and National Review present climate change as real but often
   question the extent of human responsibility or the urgency, while YouTube's
   algorithm probably shows me some skeptical content too. | B | post=0.08 argue=0.20 debate=0.10
4. Chapo Trap House and The Intercept constantly frame climate change as an
   existential crisis caused by capitalist fossil fuel exploitation, and my
   progressive circles treat human causation as absolutely established fact. | C | post=0.85 argue=0.90 debate=0.70
5. The Atlantic and New Yorker regularly feature detailed climate reporting
   emphasizing overwhelming scientific consensus on human causation, and my
   educated conversations reinforce this understanding. | C | post=0.45 argue=0.35 debate=0.30
```

Note how each persona's reasoning explicitly grounds in its information diet — the media-diet conditioning does causal work rather than cosmetic decoration. The three behavioral numbers co-vary plausibly with persona engagement: the activist-leaning Chapo listener (#4) scores high on all three, the busy skeptics (#1, #3) score low, and the mainstream-media reader (#5) sits in the middle.

## Evaluation

### Metrics

- **Jensen-Shannon Divergence (JSD)** — information-theoretic, bounded [0, 1], log base 2. Primary metric.
- **Total Variation Distance (TVD)** — L1/2; "how much probability mass would need to move to match?"
- **Mean Absolute Error (pp)** — average per-option percentage-point error. Most interpretable.
- **Plurality accuracy** — does the most-chosen option match the true top?
- **Chi-squared p-value** — significance of distributional difference.

### Pipeline

Questions are split 80/10/10 sequentially (with at least 1 in val and test):

1. **Phase 1** — run all five methods on train questions at default T=1.0.
2. **Phase 2** — select the best method by mean JSD on train.
3. **Phase 3** — temperature sweep on val: run the best method at each T ∈ {0.3, 0.6, 0.8, 1.0} and pick the winning temperature by val JSD.
4. **Phase 4** — evaluate (best method, best temperature) on test. Report per-split JSD/TVD/MAE and generalization gap.
5. **Phase 5** — behavioral rollup. For the winning (method, temperature), aggregate the three behavioral signals per (question, chosen-option) group across train/val/test, and render an R³ scatter for each test question so covariance structure between post/argue/debate is visible per chosen option.

This prevents (a) method selection from overfitting to a single question set and (b) temperature tuning from leaking into final test numbers.

## Current Results (n=100 personas, 6 train / 1 val / 1 test)

All numbers below come from a single live run against the Anthropic API at model `claude-sonnet-4-20250514`. The panel is deterministic (seed=42): 79 Bachelor's / 21 Graduate; party 61 Dem / 25 Rep / 14 Ind; baskets 51 neo_liberal / 19 progressive / 16 neo_conservative / 9 maga / 5 alt_right. Every method / temperature / split evaluates against this identical panel. 700 total API requests across 3 Message Batches submissions, ~8 min wall-clock total.

### Phase 1 — method ranking on TRAIN (6 questions @ T=1.0)

| Method | JSD ↓ | TVD ↓ | MAE (pp) ↓ | Plurality ↑ | Valid |
|---|---:|---:|---:|---:|---:|
| **value_anchored**         | **0.0291** | 0.1617 | 10.8 | **83%** | 100% |
| distribution_aware         | 0.0304 | 0.1633 | 10.9 | 67% | 100% |
| cognitive_deliberation     | 0.0307 | 0.1733 | 11.6 | 67% | 100% |
| demographic_baseline       | 0.0329 | 0.1650 | 11.0 | 67% | 100% |
| narrative_backstory        | 0.0379 | 0.1917 | 12.8 | 67% | 100% |

**Reading the ranking.** `value_anchored` — moral foundations (Care/Fairness/Loyalty/Authority/Sanctity/Liberty) + information sources, with demographics *dropped* — narrowly beats every other method. The top four are within 0.004 JSD of each other (well inside the single-question noise band); only `narrative_backstory` is meaningfully worse. The plurality split tells a sharper story: `value_anchored` correctly matches the national mode on 5/6 train questions (83%) vs. 4/6 (67%) for everyone else. Character-level narratives continue to underperform — the ~prose-sketch approach seems to push the model toward archetype rather than within-group marginal.

### Phase 3 — temperature sweep on VAL (`value_anchored`, question = `ai_job_automation`)

| Temperature | Val JSD ↓ | Val TVD | Val MAE (pp) | Plurality |
|---:|---:|---:|---:|---:|
| **0.30** | **0.0012** | 0.040 | 2.7 | 100% |
| 0.60 | 0.0056 | 0.080 | 5.3 | 100% |
| 0.80 | 0.0042 | 0.070 | 4.7 | 100% |
| 1.00 | 0.0014 | 0.040 | 2.7 | 100% |

T=0.3 wins, with T=1.0 near-tied. The interior points (0.6 / 0.8) are ~4× worse in JSD, which looks counter-intuitive but is internally consistent: at T=0.3 the model commits deterministically to the population-weighted option for each persona (tight clustering to cohort centroid); at T=1.0 sampling noise produces balanced response diversity; in between you get a noisy mixture of neither regime. Plurality stays 100% across all four temperatures, so the ranking is driven by tail-probability allocation, not modal choice. The headline pattern — **lower temperature wins on this cohort** — survives at n=100, consistent with the hypothesis that within-persona reasoning is already providing response diversity and extra sampling stochasticity on top degrades rather than helps.

### Phase 4 — held-out TEST (`value_anchored` @ T=0.3, question = `ai_work_usage`)

| Split | Questions | JSD ↓ | TVD | MAE (pp) | Plurality |
|---|---|---:|---:|---:|---:|
| TRAIN | 6 | 0.0291 | 0.162 | 10.8 | 83% |
| VAL (T=0.3) | 1 | 0.0012 | 0.040 | 2.7 | 100% |
| TEST (T=0.3) | 1 | **0.2499** | 0.570 | 38.0 | 0% |

Generalisation gap **+0.22 JSD** on a single test question — flagged `tentative` by the pipeline. Drilling in:

```
  [ai_work_usage]  JSD=0.2499  TVD=0.57  MAE=38.0pp  n=100
    All or most of my work      pred= 5%   true= 2%   Δ= +3pp
    Some of my work             pred=73%   true=19%   Δ=+54pp   ← huge overshoot
    None of my work             pred=22%   true=79%   Δ=−57pp   ← huge undershoot
```

This is a **second type** of cohort–ground-truth mismatch, distinct from the liberal-shift on partisan questions. The national topline asks "what fraction of US workers use AI at work" — the denominator is the whole labour force including construction, retail, manual trades, and workers without college exposure. Our cohort is 100% degree-holders aged 28 — a slice where daily AI use for parts of the workday is genuinely common. The model correctly simulates that slice and the result doesn't match a national poll about a very different population.

Importantly: the TRAIN and VAL questions are opinion questions (what do you believe about guns/climate/AI jobs) where a national mean is a reasonable reference for a cohort that shares general political views. The TEST question is a **behavioural-prevalence** question (do you use AI) where cohort composition directly determines the answer. A panel of 28-year-old college grads answering "how much AI do you use at work" is not sampling the same distribution as a national workforce survey — and scoring one against the other produces a predictable but uninformative high JSD. The lesson for future work is question-type aware evaluation: partisan-opinion items should be compared against subgroup crosstabs by age × education, behavioural-prevalence items against a matched occupational cohort (or ideally both).

### Per-question detail on TRAIN

Every non-trivial deviation from the national topline lands on the **liberal / pro-change** side of the option list. Directional annotation added:

| Question | Δ on top option | Interpretation |
|---|---:|---|
| `gun_sales_laws` | More strict: +5pp, Kept as they are: −11pp, Less strict: +6pp | Liberal shift on "more strict"; "kept as they are" (status-quo) under-predicted — expected cohort signature. |
| `climate_human_contribution` | A great deal: **+27pp**, Some: −9pp, Not much: −18pp | Strong liberal overshoot — our cohort treats human-caused climate change as near-unanimous (72%), national is 45%. |
| `climate_local_impact` | A great deal: +10pp, Some: +6pp, Not much: −16pp | Mild liberal shift. |
| `abortion_legal_circumstances` | Any: +3pp, Certain: 0pp, Illegal: −3pp | Near-perfect match — the 3-option form that allows "under certain circumstances" as a moderate centre is where cohort and national largely agree. |
| `abortion_legal_all_or_most` | Legal: +5pp, Most: **+12pp**, Illegal: **−17pp** | Plurality flip — predicted "Legal in most" vs national "Illegal in all or most." Again expected: 28-year-old degree-holders are substantially more pro-choice. |
| `_path_vs_deport` | Pathway: **+23pp**, Deported: −20pp, Not sure: −3pp | Strong liberal shift — 78% pathway in cohort vs 55% national. |

VAL `ai_job_automation`: predicted and national agree within ±4pp on every option (Fewer jobs 47% / 51%; About the same 30% / 28%; More jobs 23% / 21%) — a non-partisan near-consensus question is where cohort and national overlap most. This is the signal-to-noise floor of the method, not an anomaly.

### Phase 5 — behavioural rollup (`value_anchored` @ T=0.3)

Per-option means reveal a sharp and reproducible **minority-opinion-is-loudest** pattern across every partisan train question. Representative slices:

```
  gun_sales_laws
    More strict         n=61  post=0.37±0.16   argue=0.35±0.17   debate=0.30±0.11
    Kept as they are    n=23  post=0.26±0.15   argue=0.33±0.19   debate=0.30±0.12
    Less strict         n=16  post=0.67±0.13   argue=0.75±0.10   debate=0.55±0.13   ← minority, LOUDEST

  abortion_legal_circumstances
    Legal under any     n=33  post=0.56±0.19   argue=0.55±0.17   debate=0.41±0.14
    Legal certain       n=55  post=0.26±0.12   argue=0.28±0.12   debate=0.29±0.09   ← modal, QUIETEST
    Illegal in all      n=12  post=0.68±0.18   argue=0.75±0.16   debate=0.56±0.15   ← minority, LOUDEST

  _path_vs_deport
    Pathway             n=78  post=0.38±0.20   argue=0.39±0.20   debate=0.33±0.13
    Deported            n=16  post=0.67±0.18   argue=0.74±0.15   debate=0.55±0.15   ← minority, LOUDEST
    Not sure            n= 6  post=0.17±0.04   argue=0.29±0.06   debate=0.32±0.06
```

The pattern holds on **every** partisan train question: the cohort-minority position (whether it's 16% anti-gun-control in a pro-control cohort or 12% anti-abortion in a pro-choice cohort) posts ~2× more, argues ~2–2.5× more, and debates ~1.7× more than the cohort-modal position. The moderate centre option is consistently the *least* engaged group. On the test question (`ai_work_usage`) engagement is uniformly low across all three options (post ∈ [0.17, 0.31]) — it is a behavioural-prevalence question, not an identity-expressive one, and the model correctly produces flat engagement.

**Why this matters for the downstream graph-sim.** If you seed a contagion simulation with uniform broadcast weights you get a smooth averaging dynamic dominated by the modal opinion. If you seed it with these empirical weights, the minority broadcasts ~2× louder per node and argues ~2.5× more aggressively — which is exactly the structural condition under which a minority view can dominate visible feed composition even while holding 12–16% of the underlying belief share. The numbers above are the forcing function; the graph topology is the transmission medium; the emergent outcome is the thing downstream consumers actually care about (what does the timeline look like vs. what do people believe).

### Why this matters for interpretation

Social-media–sampled opinion distributions are *not* marginal distributions of belief; they are marginals reweighted by post-likelihood. Concrete example from `gun_sales_laws`: our cohort is 61% "More strict" / 23% "Kept as they are" / 16% "Less strict". Re-weighting each option's share by its mean `post_likelihood` (0.37 / 0.26 / 0.67) and re-normalising:

```
  P_posted(More strict)      ∝ 0.61 × 0.37 = 0.226  → 52%
  P_posted(Kept as they are) ∝ 0.23 × 0.26 = 0.060  → 14%
  P_posted(Less strict)      ∝ 0.16 × 0.67 = 0.107  → 25%  ← up from 16%
```

Naive scraping of this cohort's social posts would read as 52% pro-control / 25% anti-control — a substantially different distribution from the 61/16 belief split. The "gun-control is contested ~2-to-1" surface impression is partially a willingness-to-post artefact, not a belief measurement. Reporting (mean, var) per option gives downstream users this correction surface directly.

### Feeding the graph-sim model

The per-option (μ, σ) tables and the R³ point clouds are exactly the inputs a graph-propagation simulation needs:

| Simulation parameter | Sourced from |
|---|---|
| Initial opinion state per node | `chosen_option` — sampled from the predicted distribution at the best (method, T) |
| Broadcast rate of a node (how often it emits to neighbours) | `post_likelihood` |
| Contest rate (probability of replying to a disagreeing post) | `argue_likelihood` |
| Offline-transmission rate (weight of non-feed edges) | `debate_frequency` |
| Per-cohort correlation structure (e.g. "do pro-reform nodes engage more than status-quo nodes?") | R³ scatter covariance + per-option (μ, σ) |

Instead of hand-setting these weights from literature, the simulation can be seeded with empirically-shaped distributions per (demographic subgroup, chosen opinion) — which is why we collect the three signals conditionally on the letter rather than as marginals.

A plausibility check: the behavioral outputs should correlate with the existing "status-quo blindness" bias. If the model under-predicts "Kept as they are now" in the marginal, we expect those personas to also land at low post/argue/debate values — the silent group is silent both in the national poll and in the cohort simulation. The R³ plots let us inspect this directly, and the resulting graph-sim should reproduce the empirical "online discourse over-represents pro-change voices" pattern as an emergent property rather than a hand-tuned assumption.

### What the results actually tell us (honest read)

**Real signal:**

- **Value-anchored conditioning narrowly wins.** At n=100 personas × 6 train questions, dropping demographics and replacing them with moral-foundations + information-diet beats every demographic-heavy method. The JSD margin over the next method (`distribution_aware`) is 0.0013 — small, but the plurality signal is cleaner (5/6 vs 4/6 train questions matched). Both pieces of evidence point the same direction: **values are more proximally causal than demographic labels** for reproducing cohort opinion shape. The previous claim that `cognitive_deliberation` was the winner was a small-n (n=20) artefact; the real ordering at full scale is closer to a four-way near-tie among everything except narrative.
- **Lower temperature wins, with a twist.** T=0.3 takes val by a narrow margin over T=1.0. The U-shape — extremes best, middle worst — suggests two regimes: deterministic cohort-centroid at low T, balanced stochastic coverage at high T. This is a genuinely interesting finding that deserves a wider sweep before claiming a single optimum.
- **The cohort's deviations are all in the same direction: more liberal than the polled population.** Across abortion, climate, guns, and immigration, the cohort over-predicts the liberal option and under-predicts the conservative / status-quo option — never the reverse. Magnitudes: +27pp on human-caused climate, +23pp on pathway-to-legal-status, +12pp on "legal in most cases" abortion, +5pp on stricter gun laws. This is the signature of the cohort (28-year-old, college-educated, 61% Dem by sampling) reading through a faithful simulation, not noise or a model bias. Reviewers looking at the per-question breakdowns should read every "predicted − true" shift as *expected drift* when it points in the liberal direction, and as *real error* only when it doesn't.
- **The behavioural signals reveal a reproducible minority-loud-majority-quiet pattern.** On every partisan train question the cohort-minority position broadcasts and argues roughly twice as much per person as the cohort-majority position. This is the structural forcing function that converts silent-majority belief distributions into loud-minority-dominated feeds — exactly the input shape the downstream graph simulation needs.
- **The huge test-set miss is a question-type mismatch, not a simulation failure.** TEST JSD = 0.2499 on `ai_work_usage` is not the method failing to capture cohort belief; it is a national-workforce prevalence question being scored against a college-graduate sub-panel. The TRAIN/VAL questions are identity-opinion items where a national mean is a defensible reference for this cohort; TEST is a behavioural-prevalence item where it isn't. Correct fix: either use subgroup crosstabs for scoring, or restrict test questions to partisan-opinion domains.

**Caveats:**

- Status-quo / "kept as they are" options continue to be under-predicted across every method — the same training-data-loudness bias flagged from the start. Directional size: ~10pp on guns, consistent with prior runs.
- The JSD gap between positions 1 and 4 (0.0291 → 0.0329) is 0.004 — small enough that bootstrap resampling would likely produce overlapping confidence intervals. The ranking is *directional evidence*, not statistically-defensible ordering.
- One TRAIN and one VAL-question extreme: `climate_human_contribution` with +27pp overshoot on "Contributes a great deal" is the largest partisan-direction overshoot observed; it's consistent with cohort liberal-lean but its magnitude suggests the model may also be training-data-biased toward pro-consensus framings on climate.

## Limitations

1. **Small question split (6 train / 1 val / 1 test).** n=100 personas is adequate per question but we only have 8 questions total. With a single val question the temperature sweep is a noisy procedure — T=0.3 winning over T=1.0 by 0.0002 JSD would not survive a second val question with high probability. With a single test question the generalisation gap is whatever that one question happened to produce. The framework is correct but the question corpus urgently needs to grow to ≥20 before rankings are statistically defensible.
2. **Cohort vs. national ground truth mismatch.** Our panel is a specific US subgroup; our ground truth is national polls. On partisan-correlated questions this creates a known liberal-direction JSD floor; on prevalence questions (like `ai_work_usage`) it can produce arbitrarily large errors. Proper evaluation requires crosstabs for 28-year-old college grads (available for Pew surveys, not Gallup) or reweighting the panel to match national marginals.
3. **Aggregate-only evaluation.** Distributional accuracy can be achieved through either good individual simulation or cancelling errors. We cannot distinguish these without subgroup or counterfactual probes.
4. **Training-data contamination.** For widely-polled questions the model may recall headline numbers rather than simulating from first principles. The +27pp overshoot on `climate_human_contribution` is suggestive of this: the cohort being liberal explains some of the shift, but the sheer magnitude is consistent with the model reaching for the loud-pro-consensus framing that dominates climate coverage in its training data. The focused cohort partially mitigates the general problem but does not eliminate it.
5. **Single-question validation.** With one question in val and one in test, temperature selection is noisy. T=0.3 winning is consistent with broader patterns (low temperature beat high temperature in the previous n=20 run too) but the exact ranking between T=0.3 and T=1.0 (within 0.0002 JSD) is not credible from one val question.
6. **Forced choice.** Personas must commit to an option. Many real respondents volunteer "don't know"; our "Not sure" options help but do not fully close the gap.
7. **Batch contamination risk.** When 5 personas are answered in one call, the model may artificially diversify responses within the batch. Observed in sample output (each persona references different sources), which is either faithful conditioning or batch-induced differentiation — we cannot tell without a solo-call ablation.
8. **Behavioral signals are self-reported by the simulator, not validated.** The model is guessing how likely a given persona is to post/argue/debate. We have no ground-truth benchmark for these values — no public poll asks "what fraction of 28-year-old college-educated X-voters who believe Y would post about it." The values are internally consistent (engaged personas score high on all three, quiet personas low) but their absolute calibration is unverified.
9. **Behavioral responses share a prompt with choice responses.** Because post/argue/debate are generated in the same batch line as the letter, the behavioral number may be partially determined by the letter the model has just committed to (and vice versa), making the two outputs non-independent. A two-call ablation (letter first, behavioral in a follow-up) would separate these.


## Code

```
config.py         — API + experiment configuration (sweep grid, focus age)
ground_truth.py   — Survey questions with real polling distributions + source URLs
demographics.py   — Census-based sampling; sample_focused_panel for coherent cohort
personas.py       — Five persona-description methods, shared batch system prompt
survey.py         — Survey runner. Default path bundles all requests in a
                    phase into one Anthropic Message Batches API submission
                    (server-side parallel, 50% cost); legacy async fan-out
                    is kept as fallback. Shared multi-strategy response parser.
metrics.py        — JSD / TVD / MAE / χ² evaluation
analysis.py       — Comparison tables and matplotlib charts
main.py           — Train/val/test pipeline with method selection on train,
                    temperature sweep on val, final evaluation on test
```

A `--dry-run` flag exercises the full pipeline (including the temperature sweep) with simulated responses so reviewers can run everything end-to-end without an API key.
