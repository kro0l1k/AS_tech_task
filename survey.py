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

    Expected format per line: "N. [reasoning] | [LETTER]"
    Falls back to looser patterns if strict parse fails.
    Returns list of (chosen_option, reasoning) length n_personas.
    """
    valid = set(LETTERS[:n_options])
    results: list[tuple[str | None, str | None]] = [(None, None)] * n_personas

    # Primary: strict "N. text | LETTER" pattern
    strict = re.compile(
        r'^\s*(\d+)[.)]\s+(.+?)\s*\|\s*([A-J])\s*$',
        re.MULTILINE,
    )
    for m in strict.finditer(raw):
        idx = int(m.group(1)) - 1
        reasoning = m.group(2).strip()
        letter = m.group(3).upper()
        if 0 <= idx < n_personas and letter in valid:
            results[idx] = (ordered_opts[LETTERS.index(letter)], reasoning)

    # Fallback: numbered lines, grab last standalone letter
    if any(r[0] is None for r in results):
        loose = re.compile(r'^\s*(\d+)[.)]\s+(.+)$', re.MULTILINE)
        for m in loose.finditer(raw):
            idx = int(m.group(1)) - 1
            text = m.group(2).strip()
            if not (0 <= idx < n_personas) or results[idx][0] is not None:
                continue
            # Find last valid letter in text
            for ch in reversed(text):
                if ch.upper() in valid:
                    letter = ch.upper()
                    results[idx] = (
                        ordered_opts[LETTERS.index(letter)],
                        text[:text.rfind(ch)].strip(" |"),
                    )
                    break

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
) -> list[SurveyResponse]:
    """Send one batch (up to BATCH_SIZE personas) for one question."""
    n = len(descriptions)
    opts_block, ordered_opts = _shuffle_options(question, rng)
    user_msg = _build_batch_user_msg(descriptions, question, opts_block)
    max_tokens = TOKENS_PER_PERSONA * n + 50  # buffer

    for attempt in range(3):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=max_tokens,
                    temperature=TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_msg}],
                )
            raw = response.content[0].text
            parsed = _parse_batch_response(raw, ordered_opts, n, len(question.options))

            out = []
            for i, (chosen, reasoning) in enumerate(parsed):
                # Store the relevant slice of raw response for this persona
                persona_raw = raw  # full batch response stored (for inspection)
                out.append(SurveyResponse(
                    persona_index=persona_indices[i],
                    question_id=question.id,
                    chosen_option=chosen,
                    reasoning=reasoning,
                    raw_response=persona_raw,
                    options_order=ordered_opts,
                ))
            return out

        except anthropic.RateLimitError:
            await asyncio.sleep(2 ** attempt * 5)
        except Exception as e:
            if attempt == 2:
                return [
                    SurveyResponse(
                        persona_index=persona_indices[i],
                        question_id=question.id,
                        chosen_option=None,
                        reasoning=None,
                        raw_response=f"ERROR: {e}",
                        options_order=ordered_opts,
                    )
                    for i in range(n)
                ]
            await asyncio.sleep(2 ** attempt)

    # Should not reach here, but safety net
    return [
        SurveyResponse(
            persona_index=persona_indices[i],
            question_id=question.id,
            chosen_option=None,
            reasoning=None,
            raw_response="ERROR: max retries",
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
                _ask_batch(client, batch_system_prompt, chunk, q, indices, batch_rng, semaphore)
            )

    n_personas = len(persona_descriptions)
    n_batches = ceil(n_personas / BATCH_SIZE)
    total_batches = n_batches * len(questions)
    results.total_calls = total_batches

    print(
        f"  [{method_name}] {n_personas} personas × {len(questions)} questions "
        f"= {total_batches} batch calls (batch_size={BATCH_SIZE})"
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
