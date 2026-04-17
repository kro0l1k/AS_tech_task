"""
Async survey runner — batched API edition.

Two execution paths share the same prompt-builder and parser:

1. **Message Batches API** (default, USE_BATCH_API=True) —
   `run_surveys(tasks)` bundles every request across every (method ×
   temperature × question × persona-chunk) into a single
   `client.messages.batches.create(...)` call. Anthropic processes the
   whole batch server-side in parallel at 50% cost. We poll for
   completion and demultiplex results back to per-survey SurveyResults
   via each request's custom_id.

2. **Async fan-out** (fallback when USE_BATCH_API=False) — the original
   semaphore-bounded `messages.create` path. Slower and more expensive,
   but useful for debugging individual calls or running without the
   batch endpoint.

Request shape (unchanged):
- BATCH_SIZE personas per request
- each persona writes 1-2 sentences of reasoning, a letter, and three
  behavioral numbers (post/argue/debate in [0, 1])
- response format:  `N. [reasoning] | [LETTER] | post=X.XX argue=X.XX debate=X.XX`
"""
import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from math import ceil
from typing import Iterable

import anthropic

from config import (
    MODEL, TEMPERATURE, MAX_CONCURRENT_REQUESTS, BATCH_SIZE,
    USE_BATCH_API, BATCH_POLL_INTERVAL_SECONDS, BATCH_MAX_WAIT_SECONDS,
)
from ground_truth import SurveyQuestion
from personas import BATCH_SYSTEM_PROMPT

LETTERS = "ABCDEFGHIJ"

# Tokens per persona in a batch:
#   ~80 reasoning + ~10 letter + ~30 for `post=X.XX argue=X.XX debate=X.XX`
TOKENS_PER_PERSONA = 160


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SurveyResponse:
    persona_index: int
    question_id: str
    chosen_option: str | None   # None if parsing failed
    reasoning: str | None       # the 1-2 sentence internal dialogue
    raw_response: str           # full raw text from this batch slot
    options_order: list[str]    # shuffled option order presented
    # Behavioral / social-media likelihoods (all in [0, 1]):
    post_likelihood:   float | None = None
    argue_likelihood:  float | None = None
    debate_frequency:  float | None = None


@dataclass
class SurveyResults:
    method_name: str
    responses: list[SurveyResponse] = field(default_factory=list)
    failed_parses: int = 0
    total_calls: int = 0        # number of API calls (batches)
    elapsed_seconds: float = 0.0

    def get_distribution(self, question: SurveyQuestion) -> dict[str, float]:
        counts = {opt: 0 for opt in question.options}
        valid = 0
        for r in self.responses:
            if r.question_id == question.id and r.chosen_option is not None:
                counts[r.chosen_option] += 1
                valid += 1
        if valid == 0:
            return {opt: 0.0 for opt in question.options}
        return {opt: counts[opt] / valid for opt in question.options}


@dataclass
class SurveyTask:
    """
    One survey spec: a method + descriptions + questions at one temperature.
    Multiple tasks can be bundled into a single Message Batches API submission
    by run_surveys() for maximum parallelism.
    """
    method_name: str
    descriptions: list[str]
    questions: list[SurveyQuestion]
    temperature: float = TEMPERATURE
    seed: int = 42
    system_prompt: str = BATCH_SYSTEM_PROMPT
    # Disambiguates identical method_names run at different temperatures.
    # Defaults to f"{method_name}@T{temperature}".
    run_key: str | None = None

    @property
    def key(self) -> str:
        return self.run_key or f"{self.method_name}@T{self.temperature}"


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _shuffle_options(question: SurveyQuestion, rng: random.Random) -> tuple[str, list[str]]:
    """Return (formatted_options_block, shuffled_options_list)."""
    shuffled = list(question.options)
    rng.shuffle(shuffled)
    lines = [f"{LETTERS[i]}) {opt}" for i, opt in enumerate(shuffled)]
    return "\n".join(lines), shuffled


def _build_batch_user_msg(
    descriptions: list[str],
    question: SurveyQuestion,
    ordered_opts: str,
) -> str:
    """Build the user message for a batch of personas."""
    n = len(descriptions)
    persons = "\n\n".join(
        f"PERSON {i+1}:\n{desc}" for i, desc in enumerate(descriptions)
    )
    return (
        f"SURVEY QUESTION: {question.text}\n\n"
        f"Options:\n{ordered_opts}\n\n"
        f"---\n\n"
        f"{persons}\n\n"
        f"---\n"
        f"Now respond for all {n} persons in order "
        f"(format: N. [reasoning] | [LETTER] | post=X.XX argue=X.XX debate=X.XX):"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_behavioral(text: str) -> tuple[float | None, float | None, float | None]:
    """
    Pull (post, argue, debate) floats in [0, 1] from a response tail.

    Accepts any of:
      post=0.7 argue=0.3 debate=0.2
      post: 0.7, argue: 0.3, debate: 0.2
      post 0.7 | argue 0.3 | debate 0.2
    Order-independent; missing keys come back as None.
    """
    def _find(key: str) -> float | None:
        m = re.search(
            rf'{key}\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)',
            text,
            re.IGNORECASE,
        )
        if not m:
            return None
        try:
            v = float(m.group(1))
        except ValueError:
            return None
        return max(0.0, min(1.0, v))

    return (_find('post'), _find('argue'), _find('debate'))


def _parse_batch_response(
    raw: str,
    ordered_opts: list[str],
    n_personas: int,
    n_options: int,
) -> list[tuple[str | None, str | None, float | None, float | None, float | None]]:
    """
    Parse a batch response into
      (chosen_option, reasoning, post, argue, debate)
    tuples — one per slot. Any field that can't be extracted is None.
    """
    valid = set(LETTERS[:n_options])
    EMPTY = (None, None, None, None, None)
    results: list[tuple] = [EMPTY] * n_personas

    # Strip bold/italic markdown that sometimes wraps numbers: **1.**
    cleaned = re.sub(r'\*{1,2}(\d+[.):])\*{1,2}', r'\1', raw)

    # --- Build per-slot block text ---
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

    def _finalize(idx: int, reasoning_text: str, letter: str, block: str):
        if not (0 <= idx < n_personas) or letter not in valid:
            return
        if results[idx][0] is not None:
            return
        post, argue, debate = _extract_behavioral(block)
        results[idx] = (
            ordered_opts[LETTERS.index(letter)],
            reasoning_text.strip(" |→"),
            post, argue, debate,
        )

    # Pass 1: strict pipe/arrow separator inside a block
    letter_re = re.compile(r'[|→]\s*([A-J])\b')
    for idx, block in blocks.items():
        m = letter_re.search(block)
        if m:
            _finalize(idx, block[:m.start()], m.group(1).upper(), block)

    # Pass 2: loose — no separator found, prefer a letter that precedes
    # post/argue/debate keywords; otherwise the last standalone letter.
    for idx, block in blocks.items():
        if results[idx][0] is not None:
            continue
        standalone = list(re.finditer(r'\b([A-J])\b', block))
        best = None
        for m in standalone:
            tail = block[m.end():]
            if re.search(r'\b(post|argue|debate)\b', tail, re.I):
                best = m
                break
        if best is None and standalone:
            best = standalone[-1]
        if best and best.group(1).upper() in valid:
            _finalize(idx, block[:best.start()], best.group(1).upper(), block)

    # Pass 3: nuclear — scan forward from "N." directly in raw
    if any(r[0] is None for r in results):
        for idx in range(n_personas):
            if results[idx][0] is not None:
                continue
            m = re.search(rf'\b{idx+1}[.)]\s+(.{{0,400}})', cleaned, re.DOTALL)
            if not m:
                continue
            chunk = m.group(1)
            lm = re.search(r'\b([A-J])\b', chunk)
            if lm and lm.group(1).upper() in valid:
                _finalize(idx, chunk[:lm.start()], lm.group(1).upper(), chunk)

    failed = [i+1 for i, r in enumerate(results) if r[0] is None]
    if failed:
        snippet = cleaned[:200].replace('\n', ' ')
        print(f"  PARSE WARN slots {failed} unresolved. raw[0:200]: {snippet!r}")

    return results


# ---------------------------------------------------------------------------
# Request-plan: shared between batch-API and async fan-out paths
# ---------------------------------------------------------------------------

@dataclass
class _PlannedRequest:
    """A single prepared API request and where its responses should go."""
    custom_id: str
    task_key: str
    method_name: str
    question_id: str
    temperature: float
    persona_indices: list[int]
    ordered_opts: list[str]
    n_options: int
    system_prompt: str
    user_msg: str
    max_tokens: int


def _plan_requests(tasks: list[SurveyTask]) -> list[_PlannedRequest]:
    """Enumerate every (task × question × persona-chunk) request up front."""
    plan: list[_PlannedRequest] = []
    for t_idx, task in enumerate(tasks):
        # Per-task rng so option shuffles are deterministic yet distinct.
        rng = random.Random(task.seed + t_idx * 7919)
        for q in task.questions:
            for batch_start in range(0, len(task.descriptions), BATCH_SIZE):
                chunk = task.descriptions[batch_start: batch_start + BATCH_SIZE]
                indices = list(range(batch_start, batch_start + len(chunk)))
                batch_rng = random.Random(rng.randint(0, 2**32))
                opts_block, ordered_opts = _shuffle_options(q, batch_rng)
                user_msg = _build_batch_user_msg(chunk, q, opts_block)
                max_tokens = TOKENS_PER_PERSONA * len(chunk) + 50
                # custom_id is bounded to 64 chars in the Batches API —
                # keep it compact but unique.
                custom_id = f"t{t_idx}_q{_slug(q.id)}_b{batch_start}"
                plan.append(_PlannedRequest(
                    custom_id=custom_id,
                    task_key=task.key,
                    method_name=task.method_name,
                    question_id=q.id,
                    temperature=task.temperature,
                    persona_indices=indices,
                    ordered_opts=ordered_opts,
                    n_options=len(q.options),
                    system_prompt=task.system_prompt,
                    user_msg=user_msg,
                    max_tokens=max_tokens,
                ))
    return plan


def _slug(s: str) -> str:
    """Compact a string for inclusion in a custom_id (≤64 chars)."""
    return re.sub(r'[^A-Za-z0-9]', '', s)[:20]


def _responses_from_parsed(
    parsed: list,
    req: _PlannedRequest,
    raw: str,
) -> list[SurveyResponse]:
    out = []
    for i, (chosen, reasoning, post, argue, debate) in enumerate(parsed):
        out.append(SurveyResponse(
            persona_index=req.persona_indices[i],
            question_id=req.question_id,
            chosen_option=chosen,
            reasoning=reasoning,
            raw_response=raw,
            options_order=req.ordered_opts,
            post_likelihood=post,
            argue_likelihood=argue,
            debate_frequency=debate,
        ))
    return out


def _error_responses(req: _PlannedRequest, err: str) -> list[SurveyResponse]:
    return [
        SurveyResponse(
            persona_index=idx,
            question_id=req.question_id,
            chosen_option=None,
            reasoning=None,
            raw_response=f"ERROR: {err}",
            options_order=req.ordered_opts,
        )
        for idx in req.persona_indices
    ]


# ---------------------------------------------------------------------------
# Message Batches API path
# ---------------------------------------------------------------------------

async def _run_via_batch_api(
    tasks: list[SurveyTask],
) -> dict[str, SurveyResults]:
    """
    Bundle every planned request from `tasks` into one Message Batches API
    submission, poll until completion, demultiplex results back per task.

    Up to 10_000 requests and 256MB per batch (Anthropic limits). We split
    into multiple batches here only if we exceed either limit.
    """
    client = anthropic.AsyncAnthropic()
    plan = _plan_requests(tasks)

    # Initialise output shells keyed by task.key (preserving method_name).
    out: dict[str, SurveyResults] = {
        t.key: SurveyResults(method_name=t.method_name, total_calls=0)
        for t in tasks
    }
    t0 = time.time()

    # ---- Chunk the plan if it exceeds single-batch limits ----
    MAX_REQUESTS_PER_BATCH = 10_000
    chunks: list[list[_PlannedRequest]] = [
        plan[i:i + MAX_REQUESTS_PER_BATCH]
        for i in range(0, len(plan), MAX_REQUESTS_PER_BATCH)
    ] or [[]]

    total_req = len(plan)
    print(f"  Batch API: {total_req} request(s) across {len(tasks)} task(s), "
          f"{len(chunks)} batch submission(s)")

    plan_by_id = {r.custom_id: r for r in plan}

    for chunk_idx, chunk in enumerate(chunks):
        if not chunk:
            continue

        # Build request list for this batch.
        requests_payload = [
            {
                "custom_id": r.custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": r.max_tokens,
                    "temperature": r.temperature,
                    "system": r.system_prompt,
                    "messages": [{"role": "user", "content": r.user_msg}],
                },
            }
            for r in chunk
        ]

        # Submit.
        batch = await client.messages.batches.create(requests=requests_payload)
        print(f"  [chunk {chunk_idx+1}/{len(chunks)}] submitted "
              f"{len(requests_payload)} requests → batch id {batch.id}")

        # Poll. Cadence starts at BATCH_POLL_INTERVAL_SECONDS, backs off to 60s.
        poll = float(BATCH_POLL_INTERVAL_SECONDS)
        waited = 0.0
        while True:
            batch = await client.messages.batches.retrieve(batch.id)
            counts = batch.request_counts
            if batch.processing_status == "ended":
                print(f"  [chunk {chunk_idx+1}] ended. "
                      f"succeeded={counts.succeeded} errored={counts.errored} "
                      f"expired={counts.expired} canceled={counts.canceled} "
                      f"({waited:.0f}s elapsed)")
                break
            if waited >= BATCH_MAX_WAIT_SECONDS:
                print(f"  [chunk {chunk_idx+1}] TIMEOUT after {waited:.0f}s — "
                      f"giving up with partial results")
                break
            if waited == 0 or int(waited) % 60 < poll:
                print(f"  [chunk {chunk_idx+1}] processing… "
                      f"proc={counts.processing} ok={counts.succeeded} "
                      f"err={counts.errored}  ({waited:.0f}s)")
            await asyncio.sleep(poll)
            waited += poll
            poll = min(poll * 1.3, 60.0)

        # Stream results. Each yielded item has .custom_id and .result (union).
        if batch.processing_status == "ended":
            async for item in await client.messages.batches.results(batch.id):
                req = plan_by_id.get(item.custom_id)
                if req is None:
                    print(f"  WARN: unknown custom_id {item.custom_id}")
                    continue
                result = item.result
                rtype = getattr(result, "type", None)

                if rtype == "succeeded":
                    msg = result.message
                    raw = msg.content[0].text if msg.content else ""
                    parsed = _parse_batch_response(
                        raw, req.ordered_opts,
                        len(req.persona_indices), req.n_options,
                    )
                    responses = _responses_from_parsed(parsed, req, raw)
                elif rtype == "errored":
                    err = getattr(result, "error", "unknown")
                    responses = _error_responses(req, f"batch errored: {err}")
                elif rtype == "expired":
                    responses = _error_responses(req, "batch expired")
                elif rtype == "canceled":
                    responses = _error_responses(req, "batch canceled")
                else:
                    responses = _error_responses(req, f"unknown result type: {rtype}")

                shell = out[req.task_key]
                shell.total_calls += 1
                for r in responses:
                    shell.responses.append(r)
                    if r.chosen_option is None:
                        shell.failed_parses += 1

    # Any requests that the API never returned (shouldn't happen, but belt-
    # and-braces): mark their personas failed.
    seen_ids = {
        f"{r.question_id}__{idx}"
        for shell in out.values()
        for r in shell.responses
        for idx in [r.persona_index]
    }
    for req in plan:
        for idx in req.persona_indices:
            key = f"{req.question_id}__{idx}"
            if key not in seen_ids:
                shell = out[req.task_key]
                shell.responses.append(SurveyResponse(
                    persona_index=idx,
                    question_id=req.question_id,
                    chosen_option=None,
                    reasoning=None,
                    raw_response="ERROR: no response returned by batch",
                    options_order=req.ordered_opts,
                ))
                shell.failed_parses += 1

    elapsed = time.time() - t0
    for shell in out.values():
        shell.elapsed_seconds = elapsed
        total = len(shell.responses)
        print(f"  [{shell.method_name}] {shell.failed_parses}/{total} "
              f"parse failures")

    print(f"  Batch API total wall-clock: {elapsed:.1f}s")
    return out


# ---------------------------------------------------------------------------
# Async fan-out path (legacy / fallback)
# ---------------------------------------------------------------------------

async def _one_fanout_call(
    client: anthropic.AsyncAnthropic,
    req: _PlannedRequest,
    semaphore: asyncio.Semaphore,
) -> list[SurveyResponse]:
    """Send one prepared request via messages.create with retries."""
    MAX_ATTEMPTS = 6
    last_error = "unknown"
    for attempt in range(MAX_ATTEMPTS):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    system=req.system_prompt,
                    messages=[{"role": "user", "content": req.user_msg}],
                )
            raw = response.content[0].text
            parsed = _parse_batch_response(
                raw, req.ordered_opts,
                len(req.persona_indices), req.n_options,
            )
            return _responses_from_parsed(parsed, req, raw)
        except anthropic.BadRequestError as e:
            last_error = f"BadRequest: {e}"
            print(f"  API 400 — not retrying: {e}")
            break
        except (anthropic.RateLimitError,
                anthropic.APIStatusError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError) as e:
            last_error = f"{type(e).__name__}: {e}"
            base = min(2 ** (attempt + 1), 60)
            jitter = random.random() * base
            await asyncio.sleep(base + jitter)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt >= 2:
                break
            await asyncio.sleep(2 ** attempt)
    return _error_responses(req, last_error)


async def _run_via_fanout(
    tasks: list[SurveyTask],
) -> dict[str, SurveyResults]:
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    plan = _plan_requests(tasks)

    out = {
        t.key: SurveyResults(method_name=t.method_name, total_calls=0)
        for t in tasks
    }
    t0 = time.time()

    print(f"  Async fan-out: {len(plan)} request(s), "
          f"concurrency={MAX_CONCURRENT_REQUESTS}")

    coros = [_one_fanout_call(client, r, semaphore) for r in plan]
    completed = 0
    # Pair each future with its originating request.
    futures = {asyncio.ensure_future(c): r for c, r in zip(coros, plan)}
    for fut in asyncio.as_completed(futures):
        responses = await fut
        req = futures[fut]
        shell = out[req.task_key]
        shell.total_calls += 1
        for r in responses:
            shell.responses.append(r)
            if r.chosen_option is None:
                shell.failed_parses += 1
        completed += 1
        if completed % 20 == 0:
            print(f"  fan-out {completed}/{len(plan)} requests done")

    elapsed = time.time() - t0
    for shell in out.values():
        shell.elapsed_seconds = elapsed
    print(f"  Fan-out total wall-clock: {elapsed:.1f}s")
    return out


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------

async def run_surveys(
    tasks: list[SurveyTask],
    use_batch_api: bool | None = None,
) -> dict[str, SurveyResults]:
    """
    Run multiple surveys. Returns a dict keyed by each task's `key`.

    When `use_batch_api` is True (default via config.USE_BATCH_API), every
    request across every task is bundled into a single Message Batches API
    submission, processed server-side in parallel at 50% cost. Otherwise
    falls back to the async fan-out path.
    """
    if not tasks:
        return {}
    if use_batch_api is None:
        use_batch_api = USE_BATCH_API
    if use_batch_api:
        return await _run_via_batch_api(tasks)
    return await _run_via_fanout(tasks)


async def run_survey(
    persona_descriptions: list[str],
    questions: list[SurveyQuestion],
    method_name: str,
    batch_system_prompt: str = BATCH_SYSTEM_PROMPT,
    seed: int = 42,
    temperature: float = TEMPERATURE,
) -> SurveyResults:
    """
    Single-survey convenience wrapper. Prefer run_surveys() for phase-level
    bundling — one batch spanning many methods/temperatures is cheaper and
    faster than repeated single-survey batches.
    """
    task = SurveyTask(
        method_name=method_name,
        descriptions=persona_descriptions,
        questions=questions,
        temperature=temperature,
        seed=seed,
        system_prompt=batch_system_prompt,
    )
    n_personas = len(persona_descriptions)
    total_batches = ceil(n_personas / BATCH_SIZE) * len(questions)
    print(
        f"  [{method_name}] {n_personas} personas × {len(questions)} questions "
        f"= {total_batches} batch calls (batch_size={BATCH_SIZE}, T={temperature})"
    )
    out = await run_surveys([task])
    result = out[task.key]
    print(
        f"  [{method_name}] Done. {result.failed_parses}/"
        f"{len(result.responses)} parse failures. "
        f"({result.elapsed_seconds:.1f}s)"
    )
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(results: SurveyResults, path: str):
    data = {
        "method_name": results.method_name,
        "total_calls": results.total_calls,
        "failed_parses": results.failed_parses,
        "elapsed_seconds": results.elapsed_seconds,
        "responses": [
            {
                "persona_index": r.persona_index,
                "question_id": r.question_id,
                "chosen_option": r.chosen_option,
                "reasoning": r.reasoning,
                "raw_response": r.raw_response,
                "options_order": r.options_order,
                "post_likelihood":  r.post_likelihood,
                "argue_likelihood": r.argue_likelihood,
                "debate_frequency": r.debate_frequency,
            }
            for r in results.responses
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_results(path: str) -> SurveyResults:
    with open(path) as f:
        data = json.load(f)
    results = SurveyResults(
        method_name=data["method_name"],
        total_calls=data["total_calls"],
        failed_parses=data["failed_parses"],
        elapsed_seconds=data["elapsed_seconds"],
    )
    for r in data["responses"]:
        results.responses.append(SurveyResponse(
            persona_index=r["persona_index"],
            question_id=r["question_id"],
            chosen_option=r["chosen_option"],
            reasoning=r.get("reasoning"),
            raw_response=r["raw_response"],
            options_order=r["options_order"],
            post_likelihood=r.get("post_likelihood"),
            argue_likelihood=r.get("argue_likelihood"),
            debate_frequency=r.get("debate_frequency"),
        ))
    return results
