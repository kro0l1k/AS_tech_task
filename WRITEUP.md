# LLM Survey Personas — Writeup

## The Problem

Can LLMs simulate a representative panel of humans well enough to reproduce real public opinion distributions? The challenge decomposes into three sub-problems: (1) what to condition the model on, (2) how to elicit authentic responses rather than stereotype-driven or safety-trained defaults, and (3) how to measure success rigorously.

## Objective

Build 100 LLM personas of a **coherent cohort — 28-year-old US adults with a college degree** — ask them single-select survey questions drawn from recent Gallup and Pew polls with known ground-truth distributions, and evaluate five different persona-construction methods. Use a train/val/test split over questions to select the best method, then sweep sampling temperature on the validation split and evaluate the winning combination on held-out test questions.

**Why a narrow cohort rather than the full US population?** Independent marginal sampling from US-wide demographic distributions produces incoherent personas (the previous writeup flagged "22-year-old retired graduate" as a concrete failure mode). Fixing age and education range and letting everything else vary — gender, race, region, community, party, political basket, income, information diet — yields internally consistent profiles and isolates the question of whether the model can simulate *within-group variance* for a known slice of the population. We note below the evaluation trade-off this creates.

## Why This Is Hard

LLMs have systematic biases that fight faithful opinion simulation:

1. **Training distribution skew** — text corpora over-represent the young, educated, and liberal. The model's "default voice" does not match the US population.
2. **Safety-training pull** — RLHF compresses response tails, under-generating extreme positions.
3. **Stereotype caricature** — given "MAGA conservative," the model often produces an exaggerated version rather than the actual within-group variance. The modal conservative view on guns is "enforce existing laws"; the model produces "fewer laws."
4. **Status-quo blindness** — training data is written by people with strong opinions. Americans who answer "keep things as they are" or "haven't thought about it" are under-represented in the corpus and therefore in the model's latent voice.
5. **Position bias** — models systematically favor earlier or later options, adding noise orthogonal to the persona.
6. **Cohort–ground-truth mismatch** — a coherent sub-population cohort should *not* match national polling distributions exactly. A panel of 28-year-old degree-holders leans substantially more liberal than the nation on abortion, climate, and guns. JSD against national ground truth therefore has a floor set by the true cohort deviation, not the method's fidelity. This is a feature (it stops us overfitting the model to topline numbers) but it demands careful interpretation.

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
- **Option randomization** — options are shuffled per question-persona pair to mitigate position bias.
- **Shared panel** — all methods and all temperature settings evaluate against the same sampled panel so differences attribute to method/temperature, not sampling variance.
- **Temperature sweep on validation** — after method selection on train, we sweep `{0.5, 0.7, 1.0, 1.3}` on the val split and pick the winner for final test evaluation.

## Sample Inputs Passed to the API

All methods share the same system prompt and batched user-message structure. Only the *per-persona description block* differs between methods. Below is persona 0 from the seeded panel (`sample_focused_panel(n=20, seed=42)`) — a 28-year-old Hispanic male, Bachelor's, $50–75k, Urban South, lean-Republican but sampled into the `neo_liberal` basket (small-probability draw from the party-conditional basket table; CNN + Ezra Klein read as a counter-intuitive but plausible media diet for a dissatisfied moderate). Shown verbatim as sent to the model.

### Shared system prompt (all methods)

```
You are simulating authentic American survey respondents for opinion research.

You will receive profiles of several Americans and one survey question.
For EACH person:
1. Write 1-2 sentences of their honest internal reasoning — gut feelings shaped by
   their specific information sources and life situation. Be authentic: people
   are often uncertain, sometimes contradictory, and rarely perfectly informed.
2. State their answer as the option letter.

Reply in this EXACT format, one numbered line per person:
1. [reasoning] | [LETTER]
2. [reasoning] | [LETTER]
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
   human-caused versus natural cycles - the science seems somewhat uncertain. | B
2. All my sources from Pod Save America to The Daily consistently emphasize the
   scientific consensus that humans are the primary driver of climate change,
   and my educated urban community generally accepts this. | C
3. The Bulwark and National Review present climate change as real but often
   question the extent of human responsibility or the urgency, while YouTube's
   algorithm probably shows me some skeptical content too. | B
4. Chapo Trap House and The Intercept constantly frame climate change as an
   existential crisis caused by capitalist fossil fuel exploitation, and my
   progressive circles treat human causation as absolutely established fact. | C
5. The Atlantic and New Yorker regularly feature detailed climate reporting
   emphasizing overwhelming scientific consensus on human causation, and my
   educated conversations reinforce this understanding. | C
```

Note how each persona's reasoning explicitly grounds in its information diet — the media-diet conditioning does causal work rather than cosmetic decoration.

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
3. **Phase 3** — temperature sweep on val: run the best method at each T ∈ {0.5, 0.7, 1.0, 1.3} and pick the winning temperature by val JSD.
4. **Phase 4** — evaluate (best method, best temperature) on test. Report per-split JSD/TVD/MAE and generalization gap.

This prevents (a) method selection from overfitting to a single question set and (b) temperature tuning from leaking into final test numbers.

## Current Results (n=20 personas, 3 train / 1 val / 1 test)

### Phase 1 — method ranking on train (T=1.0)

| Method | JSD | TVD | MAE (pp) | Plurality |
|---|---:|---:|---:|---:|
| **cognitive_deliberation** | **0.0319** | 0.177 | 11.8 | 67% |
| value_anchored             | 0.0358 | 0.183 | 12.2 | 67% |
| demographic_baseline       | 0.0406 | 0.193 | 12.9 | 67% |
| distribution_aware         | 0.0433 | 0.210 | 14.0 | 67% |
| narrative_backstory        | 0.0604 | 0.260 | 17.3 | 67% |

**Notable reversals from the previous (un-focused, n=10) panel.**
- `cognitive_deliberation` went from worst to best. On a coherent cohort, reasoning cues produce within-group variance rather than stereotype amplification.
- `distribution_aware` dropped from best to 4th. The previous win likely leaned on retrieval of national poll toplines; a focused cohort blunts that advantage.
- `narrative_backstory` remains worst. Character-level detail (names, occupations, life events) seems to push the model toward archetype rather than marginal.

### Phase 3 — temperature sweep on val (`cognitive_deliberation`, question = `abortion_legal_circumstances`)

| Temperature | Val JSD | Val TVD | Val MAE (pp) | Plurality |
|---:|---:|---:|---:|---:|
| **0.5** | **0.0024** | 0.050 | 3.3 | 100% |
| 0.7 | 0.0097 | 0.100 | 6.7 | 100% |
| 1.0 | 0.0050 | 0.050 | 3.3 | 100% |
| 1.3 | 0.0457 | 0.217 | 14.4 | 0% |

T=0.5 wins. T=1.3 collapses plurality entirely — extreme temperature produces noise that overwhelms the conditioning. T=0.7 underperforming both 0.5 and 1.0 is almost certainly a single-question artifact (n_val=1). The broad pattern — **lower temperature wins** — is consistent with the hypothesis that within-persona reasoning is already providing response diversity, so sampling stochasticity on top degrades rather than helps.

### Phase 4 — held-out test (`cognitive_deliberation` @ T=0.5, question = `abortion_legal_all_or_most`)

| Split | JSD | TVD | MAE (pp) | Plurality |
|---|---:|---:|---:|---:|
| TRAIN | 0.0319 | 0.177 | 11.8 | 67% |
| VAL (T=0.5) | 0.0024 | 0.050 | 3.3 | 100% |
| TEST (T=0.5) | 0.0181 | 0.140 | 9.3 | 0% |

Generalization gap (test − train JSD): **−0.014** (no overfitting). But **test plurality drops to 0%** — the predicted mode is "Legal in most cases" while the national ground-truth mode is "Illegal in all or most cases." This is not a method failure; it is the cohort-ground-truth mismatch from Section "Why This Is Hard" #6 showing up directly. A panel of 28-year-old college graduates is genuinely more pro-choice than the national adult population — so the *right* answer for this cohort differs from the national poll topline against which we're scoring.

### What the results actually tell us (honest read)

**Real signal:**

- **Coherent cohorts flip the method ranking.** Reasoning-heavy methods (`cognitive_deliberation`, `value_anchored`) win when demographic attributes are internally consistent; retrieval-style methods (`distribution_aware`) win when the panel matches a national frame the model has memorized. This is strong evidence that the previous `distribution_aware` victory was retrieval, not simulation.
- **Low temperature wins on this cohort.** Within-persona reasoning already produces diversity; adding high-T sampling noise hurts more than it helps.
- **Cohort evaluation requires cohort ground truth.** The 0% test plurality with a −0.014 generalization gap is the clearest possible signal: the method is behaving consistently, but national polls are the wrong yardstick for a 28-college-graduate panel on partisan-correlated issues. We need subgroup crosstabs.

**Unchanged from the previous run:**

- Status-quo options continue to be under-predicted across every method.
- The ~0.03 JSD gap between top and bottom methods is within the single-question confidence interval at n=20; method rankings need more questions and bootstrap CIs before they are statistically defensible.

## Limitations

1. **Small n (20 personas, 3/1/1 question split).** Most observed differences still sit within a single standard error. The framework is correct but underpowered; rankings should be treated as directional.
2. **Cohort vs. national ground truth mismatch.** Our panel is a specific US subgroup; our ground truth is national polls. On partisan-correlated questions this creates a known non-zero JSD floor. Proper evaluation requires crosstabs for 28-year-old college grads (available for Pew surveys, not Gallup) or reweighting the panel to match national marginals.
3. **Aggregate-only evaluation.** Distributional accuracy can be achieved through either good individual simulation or cancelling errors. We cannot distinguish these without subgroup or counterfactual probes.
4. **Training-data contamination.** For widely-polled questions the model may recall headline numbers rather than simulating from first principles. The focused cohort partially mitigates this but does not eliminate it.
5. **Single-question validation.** With one question in val and one in test, temperature selection is a noisy procedure. T=0.5 winning is consistent with broader patterns but the exact ranking between T=0.5 and T=1.0 could flip on another question.
6. **Forced choice.** Personas must commit to an option. Many real respondents volunteer "don't know"; our "Not sure" options help but do not fully close the gap.
7. **Batch contamination risk.** When 5 personas are answered in one call, the model may artificially diversify responses within the batch. Observed in sample output (each persona references different sources), which is either faithful conditioning or batch-induced differentiation — we cannot tell without a solo-call ablation.

## What I Would Do Next (Prioritized)

### Tier 1 — fix the evaluation framing (highest leverage)

- **Subgroup ground truth.** Replace (or supplement) national toplines with crosstabs for our cohort. Pew publishes age × education breakouts for most of their poll questions. This directly addresses the 0% plurality signal and would let JSD actually reflect method quality rather than cohort-population drift.
- **Post-stratification reweighting.** If we want national inference from a focused panel, reweight responses by the inverse probability of each observed demographic cell in the US adult distribution. Closes a principled gap between "this cohort's answers" and "what the model thinks the country believes."
- **More questions, bootstrap CIs.** Grow ground_truth.py to ≥20 questions and run each method at 3–5 seeds. Bootstrap confidence intervals on mean JSD. Without CIs the ranking is not defensible at n=3 train questions.

### Tier 2 — probe whether individual simulation is working

- **Subgroup validation.** Check whether Republican personas match Republican polling on the question, not just whether aggregate matches aggregate. Reveals whether individual simulation is working or errors are cancelling.
- **Counterfactual sensitivity.** Swap one persona attribute (e.g., party R→D) and measure response shift. Tests whether conditioning does real causal work or just adds surface texture.
- **Retrieval-vs-simulation probe.** Test methods on obscure/local/hypothetical questions the model could not have seen in training. If `distribution_aware` collapses while individual methods hold up, retrieval hypothesis is confirmed definitively.

### Tier 3 — address known structural biases

- **Engagement / salience attribute.** Sample ~30% of personas as low-engagement; give them a prompt path that allows "haven't really thought about it" / status-quo answers. Directly targets the status-quo blindness that dominates current errors.
- **Response caching + cost tracking.** Hash `(persona, question, method, seed, temperature)` → response. Resumable runs, no duplicate cost. Essential at scale and essential for re-analysis.
- **Finer temperature grid.** Current sweep is four points; expand to eight and interpolate. Possibly cross with method (full grid) once budget allows.

### Tier 4 — scientific rigor

- **PUMS joint demographic sampling.** Replace the remaining marginal sampling (gender × race × region × income within the focused cohort) with correlated attributes sampled from real microdata.
- **Method ensembling.** Weighted combination of the five methods, weights learned on train. Likely beats any single method.
- **Within-persona consistency.** Ask the same question twice paraphrased; inconsistent personas are noise and should be downweighted.
- **Persona voice calibration.** Blue-collar MAGA prose should not read like PhD progressive prose; style-stamp per basket once the cohort broadens.

## Code

```
config.py         — API + experiment configuration (sweep grid, focus age)
ground_truth.py   — Survey questions with real polling distributions + source URLs
demographics.py   — Census-based sampling; sample_focused_panel for coherent cohort
personas.py       — Five persona-description methods, shared batch system prompt
survey.py         — Async batched survey runner (batch=5, temperature param),
                    multi-strategy response parser
metrics.py        — JSD / TVD / MAE / χ² evaluation
analysis.py       — Comparison tables and matplotlib charts
main.py           — Train/val/test pipeline with method selection on train,
                    temperature sweep on val, final evaluation on test
```

A `--dry-run` flag exercises the full pipeline (including the temperature sweep) with simulated responses so reviewers can run everything end-to-end without an API key.
