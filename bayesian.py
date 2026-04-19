"""
Bayesian-update architecture: prior + LLM delta → posterior.

Why this is a fundamentally different architecture (not just another prompt):

    Standard methods (demographic_baseline / value_anchored / …) ask the
    LLM to do something LLMs are empirically bad at — pick a letter whose
    AGGREGATE frequency across 100 personas matches a real-world marginal
    distribution. Token likelihood ≠ population frequency. Training data
    is not a representative sample of American opinion. For polled
    questions the model has likely seen the topline in its training set
    and regenerates *remembered* numbers instead of simulating.

    This module decouples absolute frequency from within-cell adjustment.
    For each question we hand the LLM the PUBLISHED PARTY-CELL PRIOR
    (e.g. "88% of Democrats say 'more strict'") and ask it only for a
    log-odds ADJUSTMENT in [-2, 2] per option based on THIS specific
    persona's biography — voting history, media diet, values, industry,
    age, region. Absolute numbers are anchored to real data; the LLM only
    has to do relative reasoning, which is what it's actually good at.

    Advertised range to the LLM is ±1.5 log-odds (tightened from ±2 after
    the first live run; see bayesian.py DELTA_CLIP_DEFAULT comment).

Math:

    prior[opt]   ∈ simplex (from priors.py, party-conditioned)
    delta[opt]   ∈ [-1.5, 1.5]   (LLM output, clipped; tightened from ±2
                                  because the first live run showed the
                                  model was systematically too decisive)
    alpha        ∈ [0, 1]        (shrinkage — pulls the posterior back
                                  toward the prior; default 0.65)
    logit[opt]   = log(prior[opt]) + alpha * clip(delta[opt])
    posterior    = softmax(logit)

    If delta ≡ 0, softmax(log prior) == prior — a fully-deferred persona
    reproduces its party baseline. Non-zero deltas shift probability
    around the baseline without the LLM having to guess absolute
    frequencies. The shrinkage factor is a classic Bayesian correction
    for over-confident point estimates: the first live run showed the
    LLM inflated partisan-aligned personas past their baseline and
    squeezed compromise/middle options, producing a systematic
    middle-option under-allocation. α<1 discounts every delta equally
    and pulls every persona back toward its party baseline.

Aggregation: panel distribution = mean over personas of posterior.
Tail mass (e.g. a 5% chance this persona picks D) is preserved and
contributes to the aggregate, whereas argmax sampling would discard it.

Implementation: re-uses survey.py's Message Batches infrastructure by
emitting pre-planned _PlannedRequest objects with the Bayesian-specific
system prompt, user message (per-persona priors inlined), and parser.
Returns a `SurveyResults` with `posterior` populated on each response;
chosen_option is set to argmax(posterior) for compatibility with the
rest of the pipeline (metrics, behavioral rollup, plots).
"""
from __future__ import annotations

import asyncio
import math
import random
import re
import time
from dataclasses import dataclass

import anthropic

from config import (
    MODEL, BATCH_SIZE, USE_BATCH_API,
    BATCH_POLL_INTERVAL_SECONDS, BATCH_MAX_WAIT_SECONDS,
)
from demographics import DemographicProfile
from ground_truth import SurveyQuestion
from priors import prior_for
from survey import (
    SurveyResponse, SurveyResults, LETTERS, _slug, TOKENS_PER_PERSONA,
)


# ---------------------------------------------------------------------------
# System prompt — ask for log-odds, not a letter
# ---------------------------------------------------------------------------

BAYESIAN_SYSTEM_PROMPT = """\
You are producing calibrated opinion-distribution updates for a survey-research \
panel. You will see, for each person:
  1. A partisan baseline — the published % of people in their party cell who \
pick each option (this is REAL data from Pew/Gallup/Quinnipiac crosstabs).
  2. The specific person's biography — voting history, media diet, values, \
industry, demographics.

Your job is NOT to pick an option. Your job is to estimate how much THIS \
SPECIFIC PERSON deviates from their party baseline, based on the factors in \
their biography that would pull them off the party-average answer. You output \
a log-odds adjustment per option, and we combine it with the baseline to get \
this person's posterior distribution over options.

=== WHY LOG-ODDS ===
delta = 0     → this person is exactly average for their party on this issue.
delta = +0.5  → this person is ~1.6× more likely than the baseline to pick it.
delta = +1    → ~2.7× more likely than the baseline (clear personal lean).
delta = +1.5  → ~4.5× more likely (strong evidence in biography).
delta = -0.5  → ~1.6× less likely.
delta = -1    → ~2.7× less likely.
delta = -1.5  → ~4.5× less likely (strong personal pull away from it).
Valid range: [-1.5, 1.5]. Numbers can have one decimal, e.g. +0.8, -1.3.

=== DO NOT AMPLIFY WHAT THE BASELINE ALREADY CAPTURES ===
The baseline already reflects the full distribution of personalities inside \
each party cell — including strong partisans, moderates, and cross-pressured \
voters. It is NOT a "moderate person of this party" reading. If a persona's \
basket label is "MAGA Republican" or "progressive Democrat", that alone is \
NOT a reason to push them past their party baseline; the baseline already \
contains those people. Deviate ONLY when the biography contains evidence \
BEYOND partisan identity: voting history inconsistent with party, unusual \
media diet, direct industry / life stake in the specific issue, stated \
values that cut across party lines on this specific topic.

When a middle / compromise / "some of each" option exists, respect it. Many \
real people genuinely hold that view and the baseline already allocates mass \
to it. Do NOT reflexively push mass away from the middle toward the extremes \
just because the persona reads as ideologically coherent.

=== HOW TO REASON ===
• Look at baseline first — most people in this party pick X. What about THIS \
person's biography would deviate from that?
• Strong signals: voting history inconsistent with party (e.g. former R now \
voted D), unusual media diet (e.g. mostly NPR in a Republican basket), \
industry or life situation that makes one answer personally salient \
(healthcare worker on healthcare questions, parent on school questions).
• Weak signals: generic demographic category matches.
• Default: if nothing in the biography clearly pulls them off the baseline, \
output all zeros. Most people ARE their party baseline on most issues.
• Do NOT inflate adjustments to look decisive. Most |delta| values should \
be ≤ 0.5. Only use |delta| > 1 when the biography gives clear, specific \
evidence beyond partisan identity.
• Adjustments should sum to roughly zero (boosting one option typically \
means pulling from others).

=== FORMAT ===
For each person output a <thinking>...</thinking> block (2-4 sentences \
naming the specific biographical factors you weighed) then a single line \
with one `letter=delta` entry per option, separated by spaces, then a \
`| post=X.XX argue=X.XX debate=X.XX` tail.

Behavioral scores (floats in [0, 1]):
  post   = likelihood they'd post on social media supporting their leading view
  argue  = likelihood they'd push back against people with other views
  debate = how often they actually debate the issue offline (0 never, 1 daily)
Most people are NOT highly engaged — quiet, busy people get 0.05–0.20. \
Activists get 0.6–0.95. Vary per person.

Reply in this EXACT shape — one block per person, nothing else:

1. <thinking>2-4 sentences naming the factors for person 1</thinking>
A=+0.4 B=-0.3 C=-0.1 | post=0.XX argue=0.XX debate=0.XX

2. <thinking>...</thinking>
A=+0.0 B=+0.0 C=+0.0 | post=0.XX argue=0.XX debate=0.XX

No preamble, no commentary, no summary.\
"""


# ---------------------------------------------------------------------------
# Posterior math
# ---------------------------------------------------------------------------

def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


# Defaults tuned from the first live run's error analysis:
#   DELTA_CLIP=1.5 — cap individual deltas tighter than the ±2 prompt-advertised
#                   scale; |δ|=1.5 is still a 4.5× shift, plenty of headroom.
#   SHRINKAGE =0.65 — discount every delta by this factor before softmax, pulling
#                   the posterior toward the party baseline. Live run showed the
#                   LLM over-decisive by ~4-6pp on partisan / middle-option
#                   questions; shrinkage is the principled fix.
DELTA_CLIP_DEFAULT = 1.5
SHRINKAGE_DEFAULT  = 0.65


def compute_posterior(
    prior: dict[str, float],
    deltas: dict[str, float],
    options_order: list[str],
    delta_clip: float = DELTA_CLIP_DEFAULT,
    shrinkage: float = SHRINKAGE_DEFAULT,
) -> dict[str, float]:
    """
    Posterior = softmax(log prior + shrinkage · clip(delta)) over the
    option order.

    `deltas` may be missing keys (treated as 0). Values outside
    [-delta_clip, delta_clip] are clipped. `shrinkage` ∈ [0, 1] discounts
    every clipped delta uniformly — shrinkage=1 reproduces the original
    un-regularised posterior; shrinkage=0 collapses to the prior. The
    default 0.65 corrects the over-decisive behaviour observed in the
    first live run without squashing the signal entirely.

    `options_order` controls the canonical option order used to build
    the logit vector — the returned dict uses the same option strings
    as keys.
    """
    logits = []
    for opt in options_order:
        p = max(prior.get(opt, 1e-9), 1e-9)
        raw = deltas.get(opt, 0.0)
        clipped = max(-delta_clip, min(delta_clip, raw))
        logits.append(math.log(p) + shrinkage * clipped)
    probs = _softmax(logits)
    return {opt: probs[i] for i, opt in enumerate(options_order)}


# ---------------------------------------------------------------------------
# User-message builder
# ---------------------------------------------------------------------------

def _format_prior_line(prior: dict[str, float], options_order: list[str]) -> str:
    """Compact baseline line: 'A=85%  B=13%  C=2%'."""
    parts = [
        f"{LETTERS[i]}={prior.get(opt, 0)*100:.0f}%"
        for i, opt in enumerate(options_order)
    ]
    return "  ".join(parts)


def build_bayesian_user_msg(
    profiles_chunk: list[DemographicProfile],
    descriptions_chunk: list[str],
    question: SurveyQuestion,
    options_order: list[str],
    preambles: list[str],
    rng: random.Random,
) -> str:
    """
    Build the user message for one Bayesian batch.

    Each PERSON block includes the party-conditioned baseline inline,
    right above their biography — the LLM can see "baseline for this
    person's party is A=85% B=13% C=2%" and reason from there.
    """
    opts_text = "\n".join(
        f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options_order)
    )
    preamble = rng.choice(preambles).format(q=question.text)

    persons = []
    for i, (p, desc) in enumerate(zip(profiles_chunk, descriptions_chunk)):
        prior = prior_for(question.id, p.party)
        baseline = _format_prior_line(prior, options_order)
        persons.append(
            f"PERSON {i+1}:\n"
            f"Baseline for their party cell: {baseline}\n"
            f"{desc}"
        )
    persons_block = "\n\n".join(persons)

    n = len(profiles_chunk)
    return (
        f"{preamble}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"---\n\n"
        f"{persons_block}\n\n"
        f"---\n"
        f"Now output log-odds adjustments for all {n} persons in order. "
        f"Remember: delta=0 means 'exactly the baseline'. Most deltas should "
        f"be small. One block per person, exact format:\n"
        f"N. <thinking>...</thinking>\n"
        + "  ".join(f"{LETTERS[i]}=+X.X" for i in range(len(options_order)))
        + " | post=X.XX argue=X.XX debate=X.XX"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r'<think(?:ing)?>(.*?)</think(?:ing)?>', re.DOTALL | re.IGNORECASE)
_BEHAV_KEYS = ('post', 'argue', 'debate')


def _extract_behavioral(text: str) -> tuple[float | None, float | None, float | None]:
    """Same three-dial extractor used by survey.py; local copy to avoid coupling."""
    out = []
    for key in _BEHAV_KEYS:
        m = re.search(
            rf'{key}\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)',
            text, re.IGNORECASE,
        )
        if not m:
            out.append(None)
            continue
        try:
            v = float(m.group(1))
        except ValueError:
            out.append(None)
            continue
        out.append(max(0.0, min(1.0, v)))
    return tuple(out)  # type: ignore[return-value]


def _parse_bayesian_block(
    block: str,
    n_options: int,
) -> tuple[dict[str, float] | None, str | None, tuple]:
    """
    Parse one persona's block into (deltas_by_letter, thinking, behavioral).
    Returns (None, ...) if we can't find enough letter=delta pairs.
    """
    thinking = None
    tm = _THINK_RE.search(block)
    if tm:
        thinking = tm.group(1).strip()
        block = block[tm.end():].strip() or block

    # Extract "LETTER=<signed float>" pairs — tolerates +/-, spaces, decimals.
    valid_letters = set(LETTERS[:n_options])
    pair_re = re.compile(r'\b([A-J])\s*=\s*([+-]?\d*\.?\d+)')
    deltas: dict[str, float] = {}
    for m in pair_re.finditer(block):
        letter = m.group(1).upper()
        if letter not in valid_letters:
            continue
        try:
            deltas[letter] = float(m.group(2))
        except ValueError:
            continue

    # Must have at least one letter; for missing letters, fill 0.0.
    if not deltas:
        return (None, thinking, _extract_behavioral(block))
    for letter in valid_letters:
        deltas.setdefault(letter, 0.0)

    return (deltas, thinking, _extract_behavioral(block))


def parse_bayesian_response(
    raw: str,
    options_order: list[str],
    n_personas: int,
) -> list[tuple[dict[str, float] | None, str | None, tuple]]:
    """
    Split raw response into N persona blocks and parse each.
    Output: list of (posterior_dict_or_None, thinking_or_None, (post,argue,debate)).
    """
    n_options = len(options_order)
    cleaned = re.sub(r'\*{1,2}(\d+[.):])\*{1,2}', r'\1', raw)

    # Chunk by "N." / "N)" markers.
    blocks: dict[int, str] = {}
    current_idx = -1
    current_text = ""
    for line in cleaned.splitlines():
        m = re.match(r'^\s*(\d+)[.)]\s+(.*)', line)
        if m:
            if current_idx >= 0:
                blocks[current_idx] = current_text.strip()
            current_idx = int(m.group(1)) - 1
            current_text = m.group(2)
        elif current_idx >= 0 and line.strip():
            current_text += " " + line.strip()
    if current_idx >= 0:
        blocks[current_idx] = current_text.strip()

    results: list[tuple] = [(None, None, (None, None, None))] * n_personas
    for idx, block in blocks.items():
        if not (0 <= idx < n_personas):
            continue
        deltas_by_letter, thinking, behav = _parse_bayesian_block(block, n_options)
        if deltas_by_letter is None:
            results[idx] = (None, thinking, behav)
            continue
        # Translate letter -> option text via options_order.
        deltas_by_opt = {
            options_order[LETTERS.index(letter)]: d
            for letter, d in deltas_by_letter.items()
        }
        results[idx] = (deltas_by_opt, thinking, behav)
    return results


# ---------------------------------------------------------------------------
# Planning: one request per (question × persona-chunk)
# ---------------------------------------------------------------------------

# Defensive preamble pool — matches survey._QUESTION_PREAMBLES conceptually
# but scoped to this module so we don't import a private name.
_PREAMBLES = [
    "SURVEY QUESTION: {q}",
    "Consider this question: {q}",
    "Here's the question to answer:\n{q}",
    "Question: {q}",
    "The following item from the survey:\n\n{q}",
    "Item on the questionnaire:\n{q}",
    "Please answer this one:\n{q}",
]


@dataclass
class _BayReq:
    custom_id: str
    question_id: str
    persona_indices: list[int]
    options_order: list[str]        # shuffled options as shown to the model
    profiles_chunk: list[DemographicProfile]
    user_msg: str
    max_tokens: int
    temperature: float


def _plan(
    panel: list[DemographicProfile],
    descriptions: list[str],
    questions: list[SurveyQuestion],
    temperature: float,
    seed: int,
) -> list[_BayReq]:
    """Enumerate every (question × persona-chunk) request."""
    assert len(panel) == len(descriptions), "panel / descriptions must be parallel"
    plan: list[_BayReq] = []
    rng = random.Random(seed)

    for q in questions:
        for start in range(0, len(panel), BATCH_SIZE):
            profiles_chunk = panel[start: start + BATCH_SIZE]
            desc_chunk = descriptions[start: start + BATCH_SIZE]
            indices = list(range(start, start + len(profiles_chunk)))

            # Per-batch rng — deterministic given seed + question + start.
            batch_rng = random.Random(rng.randint(0, 2**32))
            options_order = list(q.options)
            batch_rng.shuffle(options_order)

            user_msg = build_bayesian_user_msg(
                profiles_chunk, desc_chunk, q, options_order,
                _PREAMBLES, batch_rng,
            )
            max_tokens = TOKENS_PER_PERSONA * len(profiles_chunk) + 50

            plan.append(_BayReq(
                custom_id=f"bay_q{_slug(q.id)}_b{start}",
                question_id=q.id,
                persona_indices=indices,
                options_order=options_order,
                profiles_chunk=profiles_chunk,
                user_msg=user_msg,
                max_tokens=max_tokens,
                temperature=temperature,
            ))
    return plan


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _build_persona_descriptions(panel: list[DemographicProfile]) -> list[str]:
    """
    Compact biography used inside the Bayesian user message.

    Explicitly NO partisan-header block — the partisan context is already
    conveyed by the baseline distribution shown to the model. We still
    surface voting history as the richest partisan signal. Values block
    is dropped to keep prompt tokens focused on biographical signal.
    """
    out = []
    for p in panel:
        body = p.to_text()
        history = p.voting_history or f"(no vote history; {p.party})"
        out.append(
            f"Voting history: {history}\n"
            f"{body}"
        )
    return out


def _results_from_parsed(
    parsed: list,
    req: _BayReq,
    raw: str,
    shrinkage: float,
    delta_clip: float,
) -> list[SurveyResponse]:
    """Convert parsed per-persona (deltas, thinking, behav) into SurveyResponses."""
    responses = []
    for i, (deltas_by_opt, thinking, behav) in enumerate(parsed):
        p_idx = req.persona_indices[i]
        if deltas_by_opt is None:
            responses.append(SurveyResponse(
                persona_index=p_idx,
                question_id=req.question_id,
                chosen_option=None,
                reasoning=thinking,
                raw_response=raw,
                options_order=req.options_order,
                post_likelihood=behav[0],
                argue_likelihood=behav[1],
                debate_frequency=behav[2],
                posterior=None,
            ))
            continue
        prior = prior_for(req.question_id, req.profiles_chunk[i].party)
        posterior = compute_posterior(
            prior, deltas_by_opt, req.options_order,
            delta_clip=delta_clip, shrinkage=shrinkage,
        )
        # chosen_option = argmax(posterior) — keeps downstream behavioural
        # rollups and plurality checks working unchanged.
        argmax_opt = max(posterior, key=posterior.get)
        responses.append(SurveyResponse(
            persona_index=p_idx,
            question_id=req.question_id,
            chosen_option=argmax_opt,
            reasoning=thinking,
            raw_response=raw,
            options_order=req.options_order,
            post_likelihood=behav[0],
            argue_likelihood=behav[1],
            debate_frequency=behav[2],
            posterior=posterior,
        ))
    return responses


def _error_responses(req: _BayReq, err: str) -> list[SurveyResponse]:
    return [
        SurveyResponse(
            persona_index=idx,
            question_id=req.question_id,
            chosen_option=None,
            reasoning=None,
            raw_response=f"ERROR: {err}",
            options_order=req.options_order,
        )
        for idx in req.persona_indices
    ]


async def run_bayesian_survey(
    panel: list[DemographicProfile],
    questions: list[SurveyQuestion],
    temperature: float = 0.3,
    seed: int = 42,
    shrinkage: float = SHRINKAGE_DEFAULT,
    delta_clip: float = DELTA_CLIP_DEFAULT,
    use_batch_api: bool | None = None,
) -> SurveyResults:
    """
    Run the Bayesian-update architecture across every question on the panel.

    Returns a single SurveyResults whose responses carry `posterior` on
    every successful parse. Downstream aggregation (via
    SurveyResults.get_distribution) uses the posterior-averaged path
    automatically.

    `shrinkage` and `delta_clip` are applied to the raw LLM deltas before
    softmax — see compute_posterior() for the math. The defaults were
    tuned to correct systematic LLM over-decisiveness observed on the
    first live run (middle-option squeeze, partisan over-amplification).
    """
    if use_batch_api is None:
        use_batch_api = USE_BATCH_API

    descriptions = _build_persona_descriptions(panel)
    plan = _plan(panel, descriptions, questions, temperature, seed)

    results = SurveyResults(method_name="bayesian_update", total_calls=0)
    t0 = time.time()

    if not use_batch_api:
        raise NotImplementedError("bayesian runner: only batch-API path implemented")

    client = anthropic.AsyncAnthropic()
    plan_by_id = {r.custom_id: r for r in plan}
    print(f"  Bayesian-update batch API: {len(plan)} request(s) "
          f"({len(questions)} questions × {(len(panel)+BATCH_SIZE-1)//BATCH_SIZE} chunks)")

    # Single batch submission (well under the 10k-request limit for our scale).
    payload = [
        {
            "custom_id": r.custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": r.max_tokens,
                "temperature": r.temperature,
                "system": BAYESIAN_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": r.user_msg}],
            },
        }
        for r in plan
    ]
    batch = await client.messages.batches.create(requests=payload)
    print(f"  submitted {len(payload)} requests → batch id {batch.id}")

    poll = float(BATCH_POLL_INTERVAL_SECONDS)
    waited = 0.0
    while True:
        batch = await client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        if batch.processing_status == "ended":
            print(f"  ended. succeeded={counts.succeeded} errored={counts.errored} "
                  f"expired={counts.expired} canceled={counts.canceled} "
                  f"({waited:.0f}s elapsed)")
            break
        if waited >= BATCH_MAX_WAIT_SECONDS:
            print(f"  TIMEOUT after {waited:.0f}s — giving up with partials")
            break
        if waited == 0 or int(waited) % 60 < poll:
            print(f"  processing… proc={counts.processing} "
                  f"ok={counts.succeeded} err={counts.errored}  ({waited:.0f}s)")
        await asyncio.sleep(poll)
        waited += poll
        poll = min(poll * 1.3, 60.0)

    if batch.processing_status == "ended":
        async for item in await client.messages.batches.results(batch.id):
            req = plan_by_id.get(item.custom_id)
            if req is None:
                continue
            result = item.result
            rtype = getattr(result, "type", None)
            if rtype == "succeeded":
                msg = result.message
                raw = msg.content[0].text if msg.content else ""
                parsed = parse_bayesian_response(
                    raw, req.options_order, len(req.persona_indices),
                )
                responses = _results_from_parsed(
                    parsed, req, raw,
                    shrinkage=shrinkage, delta_clip=delta_clip,
                )
            elif rtype == "errored":
                err = getattr(result, "error", "unknown")
                responses = _error_responses(req, f"batch errored: {err}")
            elif rtype == "expired":
                responses = _error_responses(req, "batch expired")
            elif rtype == "canceled":
                responses = _error_responses(req, "batch canceled")
            else:
                responses = _error_responses(req, f"unknown result type: {rtype}")

            results.total_calls += 1
            for r in responses:
                results.responses.append(r)
                if r.chosen_option is None:
                    results.failed_parses += 1

    # Fill any missing requests (shouldn't happen) with error responses.
    seen = {(r.question_id, r.persona_index) for r in results.responses}
    for req in plan:
        for idx in req.persona_indices:
            if (req.question_id, idx) not in seen:
                results.responses.append(SurveyResponse(
                    persona_index=idx,
                    question_id=req.question_id,
                    chosen_option=None,
                    reasoning=None,
                    raw_response="ERROR: no response returned by batch",
                    options_order=req.options_order,
                ))
                results.failed_parses += 1

    results.elapsed_seconds = time.time() - t0
    total = len(results.responses)
    print(f"  [bayesian_update] {results.failed_parses}/{total} parse failures "
          f"({results.elapsed_seconds:.1f}s)")
    return results


# ---------------------------------------------------------------------------
# Dry-run simulator — no API calls
# ---------------------------------------------------------------------------

def run_dry_bayesian_survey(
    panel: list[DemographicProfile],
    questions: list[SurveyQuestion],
    temperature: float = 0.3,
    seed: int = 42,
    shrinkage: float = SHRINKAGE_DEFAULT,
    delta_clip: float = DELTA_CLIP_DEFAULT,
) -> SurveyResults:
    """
    Simulate the Bayesian pipeline without hitting the API.

    The 'LLM' emits small gaussian deltas around zero; we still run the
    real posterior math and aggregation (including the current
    shrinkage/clip settings). Useful for verifying the plumbing
    end-to-end before spending live-API tokens.
    """
    rng = random.Random(seed)
    results = SurveyResults(method_name="bayesian_update", total_calls=0)
    t0 = time.time()
    for q in questions:
        for i, profile in enumerate(panel):
            prior = prior_for(q.id, profile.party)
            # Small, mostly-zero-mean deltas so the posterior stays close to
            # the prior (matches how the real LLM should behave).
            deltas = {
                opt: rng.gauss(0, 0.25) for opt in q.options
            }
            posterior = compute_posterior(
                prior, deltas, q.options,
                delta_clip=delta_clip, shrinkage=shrinkage,
            )
            chosen = max(posterior, key=posterior.get)
            results.responses.append(SurveyResponse(
                persona_index=i,
                question_id=q.id,
                chosen_option=chosen,
                reasoning="[dry run]",
                raw_response="[dry run]",
                options_order=list(q.options),
                post_likelihood=max(0.0, min(1.0, rng.gauss(0.3, 0.15))),
                argue_likelihood=max(0.0, min(1.0, rng.gauss(0.3, 0.15))),
                debate_frequency=max(0.0, min(1.0, rng.gauss(0.25, 0.12))),
                posterior=posterior,
            ))
            results.total_calls += 1
    results.elapsed_seconds = time.time() - t0
    print(f"  [bayesian_update] dry-run T={temperature}: "
          f"{results.total_calls} responses")
    return results
