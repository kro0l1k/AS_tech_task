# LLM Survey Personas — Writeup

## The Problem

Can LLMs simulate a representative panel of humans well enough to reproduce real public opinion distributions? The challenge breaks into three sub-problems: (1) what information do you condition the model on, (2) how do you prompt for authentic responses rather than stereotypical or safety-trained defaults, and (3) how do you measure success.

## Objective

Build 100 LLM personas representing US adults, ask them 7 real survey questions with known ground-truth distributions from Gallup and Pew, and evaluate distributional fidelity. Test multiple persona-construction methods to understand what drives accuracy.

## Why This Is Hard

LLMs have systematic biases that work against faithful opinion simulation:

1. **Training distribution skew**: Models are trained on internet text, which over-represents certain demographics (younger, more educated, more liberal). This means the model's "default voice" doesn't match the US population.

2. **Safety-training pull**: RLHF/Constitutional AI training pushes models toward moderate, socially desirable responses. This compresses the tails of opinion distributions — the model under-generates extreme positions.

3. **Stereotype amplification**: When given demographics, models often map to a caricature rather than the genuine within-group variance. A prompt saying "Republican man from Texas" might produce an exaggerated conservative position that doesn't reflect the actual diversity of views within that group.

4. **Position bias**: LLMs systematically favor options listed earlier (primacy bias) or later (recency bias), introducing noise that has nothing to do with the persona.

## My Approach: Five Methods

I test five methods that vary along three axes — what you condition on (content), how you frame the task (individual vs. ensemble), and how the model processes the question (direct vs. reasoned).

### Method 1: Demographic Baseline
Give the model explicit demographic attributes (age, gender, race, education, income, region, community type, political leaning) sampled from Census/Gallup distributions. This is the simplest approach and serves as the null hypothesis. If demographics alone are sufficient, more complex methods aren't worth it.

### Method 2: Narrative Backstory
Same demographics, but wrapped in a biographical narrative — name, occupation, key life experience, community context. The hypothesis is that grounding the persona in specific lived experience constrains the model more tightly than abstract attributes. A "34-year-old nurse raising two kids in suburban Ohio" feels different from "Female, 34, Bachelor's, Midwest, $50-75k." The narrative reduces the model's degrees of freedom for defaulting to its training distribution.

### Method 3: Value-Belief Anchoring
Instead of demographics, condition on moral foundations (Care, Fairness, Loyalty, Authority, Sanctity, Liberty). Value profiles are sampled from distributions calibrated to match known liberal/conservative moral psychology research (Haidt et al.). The hypothesis: values are more proximally causal of opinions than demographics. Demographics predict opinions because they correlate with values — so cutting out the middleman might produce tighter distributions.

### Method 4: Distribution-Aware Ensemble
A meta-approach: tell the model it's respondent #N of 100, that the aggregate should match US population opinion, and that minority positions must be represented. This is philosophically different — it doesn't pretend the model IS an individual, it asks the model to CONTRIBUTE to a population distribution. This leverages the model's aggregate knowledge of public opinion directly.

### Method 5: Cognitive Deliberation
Same demographics as Method 1, but add a chain-of-thought step: "Before answering, think about what life experiences shape your view, what your community thinks, and what values matter most." The hypothesis is that reasoning produces more authentic responses by forcing the model to construct a rationale rather than pattern-matching demographics to a stereotypical answer. The counter-hypothesis: CoT might actually REDUCE diversity by giving the model room to rationalize toward its default position.

## Cross-Cutting Design Decisions

**Shared demographic panel**: All methods draw from the same set of 100 sampled demographic profiles (where applicable). This ensures differences in output are attributable to the persona-construction method, not sampling variation.

**Option randomization**: Survey options are shuffled for every question-persona pair. This controls for LLM position bias, which can dominate the signal if unchecked.

**Temperature = 1.0**: High temperature increases response diversity. For aggregate distributional accuracy, we want the full range of the model's conditional distribution, not just the mode.

## Evaluation

Three complementary metrics:

- **Jensen-Shannon Divergence (JSD)**: Information-theoretic measure bounded [0,1]. Zero means identical distributions. Captures the overall shape of distributional mismatch.
- **Total Variation Distance (TVD)**: The amount of probability mass you'd need to move to make the distributions match. More interpretable than JSD.
- **Mean Absolute Error (pp)**: Average per-option error in percentage points. Most intuitive — "each option is off by X points on average."
- **Plurality accuracy**: Binary — did the most popular answer match? A necessary but insufficient condition.

Chi-squared goodness-of-fit tests provide significance testing.

## What I'd Expect (Hypotheses Before Running)

1. **Demographic Baseline** will get plurality right most of the time but over-concentrate on majority positions (compressed tails).
2. **Narrative Backstory** will produce more variance but may not improve aggregate accuracy — richer prompts don't necessarily fix the model's systematic biases.
3. **Value-Belief Anchoring** will do well on socially contentious questions (abortion, gun control) where values are the primary driver, but may struggle on policy questions (immigration levels) where instrumental reasoning matters.
4. **Distribution-Aware Ensemble** will produce the best aggregate distributions because it directly leverages the model's population-level knowledge, but at the cost of individual-level authenticity.
5. **Cognitive Deliberation** could go either way — it's the method I'm least sure about. If the model's reasoning is genuinely persona-consistent, it helps. If reasoning just creates room for safety-training to pull responses toward the center, it hurts.

## Limitations and Future Directions

**This experiment doesn't answer**: Whether individual personas are "authentic" (only aggregate distributions are evaluated). You could have terrible individual simulation but good aggregate distributions if errors cancel out.

**What would make this better**:
- Joint demographic sampling from PUMS microdata instead of independent marginals
- Testing across multiple LLM families (GPT-4, Gemini, Llama) to see if method rankings are model-dependent
- Subgroup analysis: do persona distributions match ground truth WITHIN demographic slices (e.g., do Republican personas match Republican polling)?
- Post-stratification weighting as a correction layer on top of each method
- Iterative refinement: use Method 4's distributional awareness AS a calibration step for Method 1's individual personas
- Testing with actual validation surveys (not just known poll toplines) to avoid overfitting to widely-reported numbers the model has seen in training

**The deepest concern**: For questions where the ground-truth distribution is widely reported in news and training data, the model may simply recall the distribution rather than genuinely simulating individual opinions. Distribution-Aware Ensemble is especially susceptible to this — it may score well precisely because it's doing retrieval rather than simulation. Testing on obscure or custom survey questions would disentangle retrieval from simulation.

## Code

GitHub link: [to be filled in]

The codebase is organized as a set of focused Python modules — no notebooks, no frameworks, just clean functions and dataclasses. The `--dry-run` flag lets you run the full pipeline with simulated responses to verify everything works without API calls.
