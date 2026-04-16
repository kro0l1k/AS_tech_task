"""
Async survey runner that sends questions to LLM personas and collects responses.

Design decisions:
- Option order is RANDOMIZED per question-persona pair to mitigate LLM position
  bias (models tend to favor options listed first or last)
- Responses are constrained to a single letter to simplify parsing
- Method 5 (cognitive deliberation) allows longer output for reasoning, but
  final answer is still extracted as a letter
- Retries with exponential backoff for transient API errors
- Semaphore-based concurrency control to respect rate limits
"""
import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field

import anthropic

from config import MODEL, TEMPERATURE, MAX_CONCURRENT_REQUESTS, REQUEST_TIMEOUT
from ground_truth import SurveyQuestion


@dataclass
class SurveyResponse:
    persona_index: int
    question_id: str
    chosen_option: str | None  # None if parsing failed
    raw_response: str
    options_order: list[str]  # the shuffled order presented


@dataclass
class SurveyResults:
    method_name: str
    responses: list[SurveyResponse] = field(default_factory=list)
    failed_parses: int = 0
    total_calls: int = 0
    elapsed_seconds: float = 0.0

    def get_distribution(self, question: SurveyQuestion) -> dict[str, float]:
        """Compute response distribution for a single question."""
        counts = {opt: 0 for opt in question.options}
        valid = 0
        for r in self.responses:
            if r.question_id == question.id and r.chosen_option is not None:
                counts[r.chosen_option] += 1
                valid += 1
        if valid == 0:
            return {opt: 0.0 for opt in question.options}
        return {opt: counts[opt] / valid for opt in question.options}


LETTERS = "ABCDEFGHIJ"


def _format_question(
    question: SurveyQuestion, rng: random.Random
) -> tuple[str, list[str]]:
    """Format a question with randomized option order. Returns (text, ordered_options)."""
    shuffled = list(question.options)
    rng.shuffle(shuffled)

    lines = [question.text, ""]
    for i, opt in enumerate(shuffled):
        lines.append(f"{LETTERS[i]}) {opt}")
    lines.append("")
    lines.append("Your answer (one letter only):")

    return "\n".join(lines), shuffled


def _format_question_cot(
    question: SurveyQuestion, rng: random.Random
) -> tuple[str, list[str]]:
    """Format question for cognitive deliberation method (allows reasoning)."""
    shuffled = list(question.options)
    rng.shuffle(shuffled)

    lines = [question.text, ""]
    for i, opt in enumerate(shuffled):
        lines.append(f"{LETTERS[i]}) {opt}")
    lines.append("")
    lines.append(
        "Think through your perspective briefly, then on the LAST line of your "
        "response write ONLY the letter of your answer."
    )

    return "\n".join(lines), shuffled


def _parse_letter(raw: str, n_options: int) -> str | None:
    """Extract a valid option letter from LLM output."""
    raw = raw.strip()
    valid = set(LETTERS[:n_options])

    # Try: response is just a letter
    if len(raw) == 1 and raw.upper() in valid:
        return raw.upper()

    # Try: last line contains a letter (for CoT responses)
    last_line = raw.strip().split("\n")[-1].strip()
    for char in last_line:
        if char.upper() in valid:
            return char.upper()

    # Try: first letter in the response
    for char in raw:
        if char.upper() in valid:
            return char.upper()

    return None


async def _ask_one(
    client: anthropic.AsyncAnthropic,
    persona_prompt: str,
    question: SurveyQuestion,
    persona_idx: int,
    is_cot: bool,
    rng: random.Random,
    semaphore: asyncio.Semaphore,
) -> SurveyResponse:
    """Send one question to one persona, with retries."""
    if is_cot:
        q_text, ordered_opts = _format_question_cot(question, rng)
        max_tokens = 300  # allow reasoning
    else:
        q_text, ordered_opts = _format_question(question, rng)
        max_tokens = 5  # just a letter

    for attempt in range(3):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=max_tokens,
                    temperature=TEMPERATURE,
                    system=persona_prompt,
                    messages=[{"role": "user", "content": q_text}],
                )
            raw = response.content[0].text
            letter = _parse_letter(raw, len(question.options))
            chosen = ordered_opts[LETTERS.index(letter)] if letter else None

            return SurveyResponse(
                persona_index=persona_idx,
                question_id=question.id,
                chosen_option=chosen,
                raw_response=raw,
                options_order=ordered_opts,
            )
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            await asyncio.sleep(wait)
        except Exception as e:
            if attempt == 2:
                return SurveyResponse(
                    persona_index=persona_idx,
                    question_id=question.id,
                    chosen_option=None,
                    raw_response=f"ERROR: {e}",
                    options_order=ordered_opts,
                )
            await asyncio.sleep(2 ** attempt)

    return SurveyResponse(
        persona_index=persona_idx,
        question_id=question.id,
        chosen_option=None,
        raw_response="ERROR: max retries",
        options_order=ordered_opts if 'ordered_opts' in dir() else [],
    )


async def run_survey(
    persona_prompts: list[str],
    questions: list[SurveyQuestion],
    method_name: str,
    is_cot: bool = False,
    seed: int = 42,
) -> SurveyResults:
    """Run a full survey: all questions × all personas."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    rng = random.Random(seed)

    results = SurveyResults(method_name=method_name)
    t0 = time.time()

    # Build all tasks
    tasks = []
    for q in questions:
        for i, prompt in enumerate(persona_prompts):
            # Each question-persona pair gets its own rng for option shuffling
            pair_rng = random.Random(rng.randint(0, 2**32))
            tasks.append(
                _ask_one(client, prompt, q, i, is_cot, pair_rng, semaphore)
            )

    results.total_calls = len(tasks)
    print(f"  [{method_name}] Sending {len(tasks)} API calls...")

    # Run with progress reporting
    completed = 0
    for coro in asyncio.as_completed(tasks):
        resp = await coro
        results.responses.append(resp)
        if resp.chosen_option is None:
            results.failed_parses += 1
        completed += 1
        if completed % 100 == 0:
            print(f"  [{method_name}] {completed}/{len(tasks)} done")

    results.elapsed_seconds = time.time() - t0
    print(
        f"  [{method_name}] Complete. {results.failed_parses} parse failures "
        f"out of {results.total_calls} calls. ({results.elapsed_seconds:.1f}s)"
    )
    return results


def save_results(results: SurveyResults, path: str):
    """Save survey results to JSON."""
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
    """Load survey results from JSON."""
    with open(path) as f:
        data = json.load(f)
    results = SurveyResults(
        method_name=data["method_name"],
        total_calls=data["total_calls"],
        failed_parses=data["failed_parses"],
        elapsed_seconds=data["elapsed_seconds"],
    )
    for r in data["responses"]:
        results.responses.append(SurveyResponse(**r))
    return results
