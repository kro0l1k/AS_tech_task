# LLM Survey Personas — Writeup

## The Problem

Can LLMs simulate a representative panel of US adults well enough to reproduce real public opinion distributions? The challenge decomposes into three sub-problems: (1) what to condition the model on, (2) how to elicit authentic responses rather than stereotype-driven or safety-trained defaults, and (3) how to measure success rigorously.

## Objective

Build 100 LLM personas representing US adults, ask them single-select survey questions drawn from recent Gallup and Pew polls with known ground-truth distributions, and evaluate five different persona-construction methods against real response distributions. Use a train/val/test split over questions to select and validate the best method.

## Why This Is Hard

LLMs have systematic biases that fight faithful opinion simulation:

1. **Training distribution skew** — text corpora over-represent the young, educated, and liberal. The model's "default voice" does not match the US population.
2. **Safety-training pull** — RLHF compresses response tails, under-generating extreme positions.
3. **Stereotype caricature** — given "MAGA conservative," the model often produces an exaggerated version rather than the actual within-group variance. The modal conservative view on guns is "enforce existing laws"; the model produces "fewer laws."
4. **Status-quo blindness** — training data is written by people with strong opinions. Americans who answer "keep things as they are" or "haven't thought about it" are under-represented in the corpus and therefore in the model's latent voice.
5. **Position bias** — models systematically favor earlier or later options, adding noise orthogonal to the persona.

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

- **Political basket** — each persona gets one of five identities (`progressive`, `neo_liberal`, `neo_conservative`, `maga`, `alt_right`), sampled conditional on party lean. Drives information-diet composition.
- **Top 5 information influences** — 3 drawn from the basket's source pool (Pod Save America, Breitbart, Infowars, etc.), 2 from a general mainstream pool (local news, Facebook, coworkers). Appears in every persona description so the model can ground reasoning in a plausible info diet.
- **Short internal dialogue** — every response includes 1–2 sentences of reasoning before the letter choice. Prevents the model from reflexively pattern-matching demographics to stereotype answers (which it does badly when forced to `max_tokens=5`).
- **Batched API calls** — five personas per call, reduces request count 5× with minimal token overhead. The model sees all five profiles and produces numbered answers.
- **Option randomization** — options are shuffled per question-persona pair to mitigate position bias.
- **Shared panel** — all methods evaluate against the same sampled demographic panel so differences attribute to method, not sampling variance.
- **Temperature 1.0** — high temperature maximizes within-method diversity, which matters for distributional (not individual-level) accuracy.

## Evaluation

### Metrics

- **Jensen-Shannon Divergence (JSD)** — information-theoretic, bounded [0, 1], log base 2. Primary metric.
- **Total Variation Distance (TVD)** — L1/2; "how much probability mass would need to move to match?"
- **Mean Absolute Error (pp)** — average per-option percentage-point error. Most interpretable.
- **Plurality accuracy** — does the most-chosen option match the true top?
- **Chi-squared p-value** — significance of distributional difference.

### Train / Val / Test Split

Questions are split 80/10/10 sequentially (at least 1 in val and test):

1. **Phase 1** — run all methods on train questions.
2. **Phase 2** — select the best method by mean JSD on train.
3. **Phase 3** — evaluate only the selected method on val and test.
4. **Phase 4** — compare train/val/test metrics to check generalization.

This prevents method selection from overfitting to a single question set and provides an honest held-out evaluation.

## Current Results (n=10 personas, 3 train / 1 val / 1 test)

```
Method                       JSD     TVD    MAE(pp)  Plurality
distribution_aware         0.0375  0.1567   10.4      33%
demographic_baseline       0.0568  0.2067   13.8      67%
narrative_backstory        0.1040  0.2133   14.2      67%
value_anchored             0.1047  0.2233   14.9      67%
cognitive_deliberation     0.1130  0.2433   16.2      67%
```

Best method: `distribution_aware`. Generalization gap (test − train JSD) is −0.027 — no overfitting detected. But with one question per split, these single-point estimates are statistically meaningless.

### What the results actually tell us (honest read)

**Real signal despite small n:**

- **Status-quo options are systematically under-predicted.** Across every method, every run, "kept as they are now" and "kept at present level" come out at a small fraction of their true rate. This is not noise — it is model bias. The training corpus lacks fluent voice for the satisfied or disengaged.
- **"Less strict" on gun laws is over-predicted by every method.** The model has learned the Twitter/talk-radio libertarian gun position, not the modal conservative voter position (which is "enforce existing laws").

**Probable signal, unconfirmed at n=10:**

- `distribution_aware` may be winning through retrieval, not simulation — it probably recalls poll toplines from training data. Would collapse on obscure questions the model has not seen; individual-based methods would hold up.
- `cognitive_deliberation` amplifies bias rather than reducing it. When asked to reason carefully, the model defaults to salient moral frames (e.g., "drugs are bad" reasoning over-generates prohibition responses).
- `narrative_backstory` and `value_anchored` produce nearly identical aggregate distributions. Richer conditioning may not be sampling different parts of the model's distribution.

**Noise, probably:** the ~0.07 JSD gap between methods. At n=10, 95% CI on a 33% proportion is ±30pp; method differences fall inside a single standard error. Cannot rank methods confidently until n ≥ 100.

## Limitations

1. **Small n** — most observed differences are sampling noise. The framework is correct but underpowered.
2. **Aggregate-only evaluation** — distributional accuracy can be achieved either through good individual simulation or through cancelling errors. We cannot tell which. `distribution_aware`'s strength is suspicious in this light.
3. **Training-data contamination** — for widely-polled questions, the model may recall the headline number rather than simulating responses from first principles.
4. **Independent marginal sampling** — demographic attributes are drawn from independent marginals, producing occasional unrealistic combos (22-year-old retired graduate). A joint sample from ACS microdata would fix this.
5. **Forced choice** — personas must commit to an option. In reality many respondents volunteer "don't know." Our "Not sure" options help but do not fully close the gap.
6. **Batch contamination risk** — when 5 personas are answered in one call, the model may artificially diversify responses within the batch. Unverified.

## What I Would Do Next (Prioritized)

Before scaling to 100 personas × 100 questions, I would invest in:

### Tier 1 — highest leverage, lowest effort

- **Post-stratification weighting.** Our panel will have basket imbalances; reweight by inverse propensity to match Census marginals. Closes 3–8pp of error with zero new API calls.
- **Multiple seeds with bootstrap confidence intervals.** Run each condition 3–5× with different seeds. Without CIs, method rankings are not statistically sound.
- **Response caching + cost tracking.** Hash `(persona, question, method, seed)` → response. Resumable runs, no duplicate cost. Essential at scale.

### Tier 2 — fixes known biases

- **Engagement / salience attribute.** Sample ~30% of personas as low-engagement; give them a prompt path that allows "haven't really thought about it" / status-quo answers. Directly targets the status-quo blindness that dominates current errors.
- **Subgroup validation.** Pew publishes crosstabs by party, age, education. Check whether Republican personas match Republican polling, not just whether aggregate matches aggregate. Reveals whether individual simulation is working or errors are cancelling.
- **Counterfactual sensitivity.** Swap one persona attribute (e.g., party R→D), measure response shift. Tests whether conditioning does real causal work or just adds texture.

### Tier 3 — scientific rigor

- **Retrieval-vs-simulation probe.** Test methods on obscure/local/hypothetical questions the model has not seen in training. If `distribution_aware` collapses while individual methods hold up, retrieval hypothesis confirmed.
- **Temperature sweep.** Test t ∈ {0.5, 0.7, 1.0, 1.3} as an orthogonal axis. Likely different optima per question type.
- **PUMS joint demographic sampling.** Replaces independent marginals with correlated attributes sampled from real microdata.
- **Age × basket correlation.** Currently basket only depends on party. Should also correlate with age (Gen Z more progressive / alt-right, Boomers more neo-conservative / neo-liberal).

### Tier 4 — quality polish

- **Persona voice calibration.** Blue-collar MAGA prose should not read like PhD progressive prose. Style stamping per basket.
- **Current-events priming.** Each basket "saw" 2–3 specific recent news items this week. Anchors personas in time.
- **Method ensembling.** Weighted combination of methods, weights learned on train split. Likely beats any single method.
- **Within-persona consistency.** Ask same question twice paraphrased; inconsistent personas are noise and should be downweighted.

## Code

The codebase is organized as focused Python modules — no notebooks, no frameworks:

```
config.py         — API and experiment configuration
ground_truth.py   — Survey questions with real polling distributions + source URLs
demographics.py   — Census-based sampling, political baskets, influence pools
personas.py       — Five persona-description methods, shared batch system prompt
survey.py         — Async batched survey runner, multi-strategy response parser
metrics.py        — JSD / TVD / MAE / χ² evaluation
analysis.py       — Comparison tables and matplotlib charts
main.py           — Train/val/test pipeline: run methods, select best, validate
```

A `--dry-run` flag exercises the full pipeline with simulated responses, letting reviewers run everything end-to-end without an API key.
