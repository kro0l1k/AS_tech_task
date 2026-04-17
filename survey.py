"""
Async survey runner — batched edition.

Key changes from v1:
- API calls are BATCHED: 5 personas per call (configurable via BATCH_SIZE in config)
  → reduces call count by 5x with minimal token overhead
- Short internal dialogue: every persona writes 1-2 sentences of reasoning before
  committing to an answer (max_tokens scales with batch size)
- Response format: "N. [reasoning] | [LETTER]" — pipe separator for reliable parsing
- reasoning field saved per response for inspection / debugging
"""
import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from math import ceil

import anthropic

from config import MODEL, TEMPERATURE, MAX_CONCURRENT_REQUESTS, BATCH_SIZE
from ground_truth import SurveyQuestion
from personas import BATCH_SYSTEM_PROMPT

LETTERS = "ABCDEFGHIJ"

# Tokens per persona in a batch: ~80 reasoning + ~10 letter/formatting
TOKENS_PER_PERSONA = 120


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
        f"Now respond for all {n} persons in order (format: N. [reasoning] | [LETTER]):"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_batch_response(
    raw: str,
    ordered_opts: list[str],
    n_personas: int,
    n_options: int,
) -> list[tuple[str | None, str | None]]:
    """
    Parse a batch response into (chosen_option, reasoning) tuples.

    Tries five strategies in order, stopping per-slot once resolved:
      1. Strict  "N. text | LETTER"  (pipe separator)
      2. Arrow   "N. text → LETTER"  (model sometimes uses →)
      3. Multiline — join continuation lines then re-apply strict/arrow
      4. Loose   "N. ...text...LETTER"  (last valid letter in line)
      5. Nuclear — for each unresolved slot N, find "N." anywhere then
                   scan forward for first valid letter

    Logs a warning for any slot still None after all passes.
    """
    valid = set(LETTERS[:n_options])
    results: list[tuple[str | None, str | None]] = [(None, None)] * n_personas

    # Strip bold/italic markdown that sometimes wraps numbers: **1.**
    cleaned = re.sub(r'\*{1,2}(\d+[.):])\*{1,2}', r'\1', raw)

    def _resolve(idx: int, text: str, letter: str):
        if 0 <= idx < n_personas and letter in valid and results[idx][0] is None:
            results[idx] = (ordered_opts[LETTERS.index(letter)],
                            text.strip(" |→"))

    # --- Pass 1 & 2: pipe or arrow separator ---
    sep_re = re.compile(
        r'^\s*(\d+)[.)]\s+(.+?)\s*[|→]\s*([A-J])\s*$',
        re.MULTILINE,
    )
    for m in sep_re.finditer(cleaned):
        _resolve(int(m.group(1)) - 1, m.group(2), m.group(3).upper())

    # --- Pass 3: multiline reasoning — merge continuation lines ---
    if any(r[0] is None for r in results):
        # Collapse blocks: a numbered line + any following non-numbered lines
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

        for idx, text in blocks.items():
            if results[idx][0] is not None:
                continue
            # Try separator in merged text
            m = re.search(r'[|→]\s*([A-J])\s*$', text)
            if m and m.group(1).upper() in valid:
                _resolve(idx, text[:m.start()], m.group(1).upper())
            else:
                # Pass 4: last valid letter in merged text
                for ch in reversed(text):
                    if ch.upper() in valid:
                        _resolve(idx, text[:text.rfind(ch)], ch.upper())
                        break

    # --- Pass 5: nuclear — scan forward from "N." in raw ---
    if any(r[0] is None for r in results):
        for idx in range(n_personas):
            if results[idx][0] is not None:
                continue
            pattern = re.compile(
                rf'\b{idx+1}[.)]\s+(.{{0,300}})',
                re.DOTALL,
            )
            m = pattern.search(cleaned)
            if m:
                chunk = m.group(1)
                for ch in chunk:
                    if ch.upper() in valid:
                        _resolve(idx, chunk[:chunk.index(ch)], ch.upper())
                        break

    # Warn on remaining failures
    failed = [i+1 for i, r in enumerate(results) if r[0] is None]
    if failed:
        snippet = cleaned[:200].replace('\n', ' ')
        print(f"  PARSE WARN slots {failed} unresolved. raw[0:200]: {snippet!r}")

    return results


# ---------------------------------------------------------------------------
# Batch API call
# ---------------------------------------------------------------------------

async def _ask_batch(
    client: anthropic.AsyncAnthropic,
    system_prompt: str,
    descriptions: list[str],
    question: SurveyQuestion,
    persona_indices: list[int],
    rng: random.Random,
    semaphore: asyncio.Semaphore,
    temperature: float = TEMPERATURE,
) -> list[SurveyResponse]:
    """Send one batch (up to BATCH_SIZE personas) for one question."""
    n = len(descriptions)
    opts_block, ordered_opts = _shuffle_options(question, rng)
    user_msg = _build_batch_user_msg(descriptions, question, opts_block)
    max_tokens = TOKENS_PER_PERSONA * n + 50  # buffer

    # Up to 6 attempts. Exponential backoff with full jitter:
    #   base 2, 4, 8, 16, 32, 60s  +  uniform(0, base) jitter
    # Covers transient 429/529/overload across ~2 min per slot.
    MAX_ATTEMPTS = 6
    last_error: str = "unknown"

    for attempt in range(MAX_ATTEMPTS):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_msg}],
                )
            raw = response.content[0].text
            parsed = _parse_batch_response(raw, ordered_opts, n, len(question.options))

            out = []
            for i, (chosen, reasoning) in enumerate(parsed):
                out.append(SurveyResponse(
                    persona_index=persona_indices[i],
                    question_id=question.id,
                    chosen_option=chosen,
                    reasoning=reasoning,
                    raw_response=raw,
                    options_order=ordered_opts,
                ))
            return out

        except anthropic.BadRequestError as e:
            # 400 — non-retryable (e.g. temperature out of range).
            # Log once and stop retrying.
            last_error = f"BadRequest (non-retryable): {e}"
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
            # Unexpected — shorter backoff, fewer attempts.
            if attempt >= 2:
                break
            await asyncio.sleep(2 ** attempt)

    # All retries exhausted or non-retryable: emit ERROR responses carrying
    # the actual last error (not the placeholder "max retries").
    return [
        SurveyResponse(
            persona_index=persona_indices[i],
            question_id=question.id,
            chosen_option=None,
            reasoning=None,
            raw_response=f"ERROR: {last_error}",
            options_order=ordered_opts,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_survey(
    persona_descriptions: list[str],
    questions: list[SurveyQuestion],
    method_name: str,
    batch_system_prompt: str = BATCH_SYSTEM_PROMPT,
    seed: int = 42,
    temperature: float = TEMPERATURE,
) -> SurveyResults:
    """Run a full survey using batched API calls (BATCH_SIZE personas per call)."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    rng = random.Random(seed)

    results = SurveyResults(method_name=method_name)
    t0 = time.time()

    # Build all batch tasks: (descriptions_chunk, question, indices)
    tasks = []
    for q in questions:
        for batch_start in range(0, len(persona_descriptions), BATCH_SIZE):
            chunk = persona_descriptions[batch_start: batch_start + BATCH_SIZE]
            indices = list(range(batch_start, batch_start + len(chunk)))
            batch_rng = random.Random(rng.randint(0, 2**32))
            tasks.append(
                _ask_batch(
                    client, batch_system_prompt, chunk, q, indices,
                    batch_rng, semaphore, temperature=temperature,
                )
            )

    n_personas = len(persona_descriptions)
    n_batches = ceil(n_personas / BATCH_SIZE)
    total_batches = n_batches * len(questions)
    results.total_calls = total_batches

    print(
        f"  [{method_name}] {n_personas} personas × {len(questions)} questions "
        f"= {total_batches} batch calls (batch_size={BATCH_SIZE}, T={temperature})"
    )

    completed = 0
    for coro in asyncio.as_completed(tasks):
        batch_responses = await coro
        for r in batch_responses:
            results.responses.append(r)
            if r.chosen_option is None:
                results.failed_parses += 1
        completed += 1
        if completed % 10 == 0:
            print(f"  [{method_name}] {completed}/{total_batches} batches done")

    results.elapsed_seconds = time.time() - t0
    total_responses = len(results.responses)
    print(
        f"  [{method_name}] Done. {results.failed_parses}/{total_responses} parse failures. "
        f"({results.elapsed_seconds:.1f}s)"
    )
    return results


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
        ))
    return results
