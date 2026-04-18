#!/usr/bin/env python3
"""
LLM Survey Persona Experiment
==============================

Default run: sample a panel from the chosen population, answer every survey
question with method='value_anchored' @ T=0.3, then report split metrics and
behavioral stats. Phases 1-3 (method comparison + temp sweep) only fire when
--model_selection is passed.

Flags:
  --population {college_educated | seniors_south | general_us}
      Which underlying group the synthetic personas model. Distributions are
      from Census / ACS / Pew. Default: college_educated.
          college_educated → bachelor's+ adults across USA, ages 22-65
          seniors_south    → 65+ in FL / AL / LA / TX (state weights from
                             Census senior counts)
          general_us       → national adult sample (matches poll frames)

  --model_selection
      Opt into the full 4-phase pipeline:
          Phase 1 — run ALL methods on TRAIN questions
          Phase 2 — pick best method by mean JSD on TRAIN
          Phase 3 — sweep temperature on VAL for the best method
          Phase 4 — evaluate (best method, best T) on TEST
      Off by default because prior runs already identified
      value_anchored @ T=0.3 as the winner; skipping saves ~90% of API calls.

  --method NAME
      Restrict the Phase-1 sweep to a single method. Requires
      --model_selection (no effect otherwise).

  --dry-run
      Simulate responses locally — no API key needed. Useful for testing the
      pipeline end-to-end.

  --seed INT
      Controls panel sampling and all stochastic persona construction.
      Default: 42.

Question split: first 80% train, next 10% val, last 10% test (at least one
question guaranteed in val and test).

Usage:
    python main.py                                     # default pipeline, live API
    python main.py --population seniors_south          # different cohort
    python main.py --dry-run                           # no API key needed
    python main.py --model_selection                   # full 4-phase pipeline
    python main.py --model_selection --method value_anchored  # single-method sweep
"""
import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from collections import Counter

from config import (
    NUM_PERSONAS, RESULTS_DIR, TEMPERATURE, TEMPERATURE_SWEEP,
    FOCUS_AGE, MIN_VALID_RATE,
    DEFAULT_POPULATION, DEFAULT_METHOD, DEFAULT_TEMPERATURE,
)
from ground_truth import QUESTIONS, SurveyQuestion
from demographics import sample_population_panel, POPULATION_SPECS
from personas import METHODS
from survey import (
    run_survey, run_surveys, SurveyTask,
    save_results, load_results, SurveyResults, SurveyResponse,
)
from metrics import evaluate_method, MethodMetrics
from analysis import (
    print_full_report, save_report, plot_distributions, generate_comparison_table,
    print_behavioral_summary, plot_behavioral_r3,
)


# ---------------------------------------------------------------------------
# Question splitting
# ---------------------------------------------------------------------------

def split_questions(
    questions: list[SurveyQuestion],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[list, list, list]:
    """
    Sequential 80/10/10 split.
    Guarantees at least 1 question in val and test regardless of total count.
    Requires at least 3 questions total.
    """
    n = len(questions)
    if n < 3:
        raise ValueError(f"Need ≥3 questions for train/val/test split, got {n}")

    n_val  = max(1, math.floor(n * val_frac))
    n_test = max(1, n - math.floor(n * train_frac) - n_val)
    n_train = n - n_val - n_test

    train = questions[:n_train]
    val   = questions[n_train : n_train + n_val]
    test  = questions[n_train + n_val :]
    return train, val, test


# ---------------------------------------------------------------------------
# Dry-run helper
# ---------------------------------------------------------------------------

def run_dry_survey(
    descriptions: list[str],
    questions: list[SurveyQuestion],
    method_name: str,
    seed: int = 42,
    temperature: float = TEMPERATURE,
) -> SurveyResults:
    """
    Simulated responses. Temperature widens/narrows the noise band so sweeps
    produce a (toy) signal: moderate T ≈ best, extremes get noisier.
    """
    rng = random.Random(seed)
    # Noise scales with |T - 1.0|; flat floor so low T is quiet but not perfect.
    noise_sigma = 0.03 + 0.08 * abs(temperature - 1.0)
    results = SurveyResults(method_name=method_name)
    t0 = time.time()
    for q in questions:
        gt = [q.ground_truth[o] for o in q.options]
        noise = [rng.gauss(0, noise_sigma) for _ in gt]
        perturbed = [max(0.01, p + n) for p, n in zip(gt, noise)]
        s = sum(perturbed)
        perturbed = [p / s for p in perturbed]
        # Synthetic behavioral centers per option — most people low-engagement,
        # some options attract more activism. Dry-run only.
        opt_centers = {
            opt: (rng.uniform(0.1, 0.6), rng.uniform(0.05, 0.5), rng.uniform(0.05, 0.4))
            for opt in q.options
        }
        for i in range(len(descriptions)):
            chosen = rng.choices(q.options, weights=perturbed, k=1)[0]
            cp, ca, cd = opt_centers[chosen]
            # Add per-persona noise; clamp to [0, 1].
            def _j(center):
                return max(0.0, min(1.0, rng.gauss(center, 0.18)))
            results.responses.append(SurveyResponse(
                persona_index=i,
                question_id=q.id,
                chosen_option=chosen,
                reasoning="[dry run]",
                raw_response="[dry run]",
                options_order=q.options,
                post_likelihood=_j(cp),
                argue_likelihood=_j(ca),
                debate_frequency=_j(cd),
            ))
            results.total_calls += 1
    results.elapsed_seconds = time.time() - t0
    print(f"  [{method_name}] dry-run T={temperature}: {results.total_calls} responses")
    return results


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

async def _run_one_method(
    method_name: str,
    descriptions: list[str],
    questions: list[SurveyQuestion],
    dry_run: bool,
    seed: int,
    save_path: str,
    temperature: float = TEMPERATURE,
) -> SurveyResults:
    if dry_run:
        results = run_dry_survey(
            descriptions, questions, method_name,
            seed=seed, temperature=temperature,
        )
    else:
        results = await run_survey(
            descriptions, questions, method_name,
            seed=seed, temperature=temperature,
        )
    save_results(results, save_path)
    return results


async def _run_many(
    tasks: list[SurveyTask],
    dry_run: bool,
    save_paths: dict[str, str],
) -> dict[str, SurveyResults]:
    """
    Run multiple SurveyTask specs. Live mode bundles them all into a single
    Message Batches API submission. Dry-run loops locally with no API.
    Returns dict keyed by task.key.
    """
    if dry_run:
        out: dict[str, SurveyResults] = {}
        for task in tasks:
            r = run_dry_survey(
                task.descriptions, task.questions, task.method_name,
                seed=task.seed, temperature=task.temperature,
            )
            out[task.key] = r
            save_results(r, save_paths[task.key])
        return out

    out = await run_surveys(tasks)
    for key, r in out.items():
        save_results(r, save_paths[key])
    return out


def _print_panel_summary(panel, spec):
    """Print the composition of the sampled panel against the spec."""
    print(f"\nPanel: {len(panel)} personas")
    print(f"  population: {spec.name}  —  {spec.description}")

    ages = [p.age for p in panel]
    print(f"  age        : min={min(ages)} max={max(ages)} "
          f"mean={sum(ages)/len(ages):.1f}")

    edu_counts     = Counter(p.education for p in panel)
    party_counts   = Counter(p.party for p in panel)
    race_counts    = Counter(p.race for p in panel)
    region_counts  = Counter(p.region for p in panel)
    basket_counts  = Counter(p.political_basket for p in panel)
    print(f"  education  = {dict(edu_counts)}")
    print(f"  race       = {dict(race_counts)}")
    print(f"  region     = {dict(region_counts)}")
    print(f"  party      = {dict(party_counts)}")
    print(f"  basket     = {dict(basket_counts)}")

    # Show state breakdown only when the spec constrains it.
    if spec.state_dist:
        state_counts = Counter(p.state for p in panel if p.state)
        print(f"  state      = {dict(state_counts)}")


async def _run_default_pipeline(
    panel,
    train_qs: list[SurveyQuestion],
    val_qs: list[SurveyQuestion],
    test_qs: list[SurveyQuestion],
    dry_run: bool,
    seed: int,
    population: str,
):
    """
    Skip method selection + temperature sweep. Run the hard-coded default
    (DEFAULT_METHOD @ DEFAULT_TEMPERATURE) on train / val / test and report.
    """
    method_name = DEFAULT_METHOD
    T = DEFAULT_TEMPERATURE

    print("=" * 62)
    print(f"  DEFAULT RUN — method='{method_name}'  T={T}  population='{population}'")
    print("  (pass --model_selection to run the full 4-phase pipeline)")
    print("=" * 62)

    method = METHODS[method_name]
    desc_fn = method["desc_fn"]
    descriptions = (
        desc_fn(panel, seed=seed) if method.get("needs_seed") else desc_fn(panel)
    )
    preview = descriptions[0][:180].replace("\n", " | ")
    print(f"\n[{method_name}] {method['description']}")
    print(f"  persona[0]: '{preview}…'")

    # One SurveyTask per split → single batch submission.
    tasks: list[SurveyTask] = []
    paths: dict[str, str] = {}
    for split_name, qs in [("train", train_qs), ("val", val_qs), ("test", test_qs)]:
        if not qs:
            continue
        key = f"{split_name}_{method_name}_T{T}"
        tasks.append(SurveyTask(
            method_name=method_name,
            descriptions=descriptions,
            questions=qs,
            temperature=T,
            seed=seed,
            run_key=key,
        ))
        paths[key] = os.path.join(
            RESULTS_DIR, split_name, f"{method_name}_T{T}.json",
        )

    out = await _run_many(tasks, dry_run, paths)

    train_results = out[f"train_{method_name}_T{T}"]
    val_results   = out[f"val_{method_name}_T{T}"]
    test_results  = out[f"test_{method_name}_T{T}"]

    train_metrics = evaluate_method(train_results, train_qs)
    val_metrics   = evaluate_method(val_results,   val_qs)
    test_metrics  = evaluate_method(test_results,  test_qs)

    _print_split_comparison(
        method_name,
        train_metrics, val_metrics, test_metrics,
        train_qs, val_qs, test_qs,
        best_T=T,
    )

    # Minimal report: no train-across-methods table, no temp sweep table.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "population": population,
        "method": method_name,
        "temperature": T,
        "model_selection": False,
        "splits": {
            split: {
                "mean_jsd": mm.mean_jsd,
                "mean_tvd": mm.mean_tvd,
                "mean_mae_pp": mm.mean_mae_pp,
                "plurality_accuracy": mm.plurality_accuracy,
                "per_question": {
                    qm.question_id: {
                        "jsd": qm.jsd, "tvd": qm.tvd, "mae_pp": qm.mae_pp,
                        "predicted": qm.predicted_dist,
                        "ground_truth": qm.ground_truth_dist,
                    }
                    for qm in mm.per_question
                },
            }
            for split, mm in [
                ("train", train_metrics),
                ("val",   val_metrics),
                ("test",  test_metrics),
            ]
        },
    }
    report_path = os.path.join(RESULTS_DIR, "report_default.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {report_path}")

    # Behavioral stats + R^3 scatter, same as the full pipeline.
    print("\n" + "=" * 62)
    print(f"  PHASE 5 — behavioral stats  (method: '{method_name}')")
    print("=" * 62)
    print_behavioral_summary(train_results, train_qs, header="TRAIN — behavioral")
    print_behavioral_summary(val_results,   val_qs,   header="VAL   — behavioral")
    print_behavioral_summary(test_results,  test_qs,  header="TEST  — behavioral")

    plot_dir = os.path.join(RESULTS_DIR, "behavioral")
    for q in test_qs:
        path = os.path.join(plot_dir, f"r3_{q.id}.png")
        plot_behavioral_r3(
            test_results, q, path,
            title_suffix=f"[TEST · {method_name} · T={T}]",
        )

    return train_results, val_results, test_results


async def run_experiment(
    population: str = DEFAULT_POPULATION,
    do_model_selection: bool = False,
    methods_to_run: list[str] | None = None,
    dry_run: bool = False,
    seed: int = 42,
):
    print("=" * 62)
    print("  LLM SURVEY PERSONA EXPERIMENT  (train / val / test)")
    print("=" * 62)

    # --- Split questions ---
    train_qs, val_qs, test_qs = split_questions(QUESTIONS)
    print(f"\nQuestions  total={len(QUESTIONS)}"
          f"  train={len(train_qs)}  val={len(val_qs)}  test={len(test_qs)}")
    print("  TRAIN:", [q.id for q in train_qs])
    print("  VAL:  ", [q.id for q in val_qs])
    print("  TEST: ", [q.id for q in test_qs])

    # --- Sample panel from the chosen population spec.
    # Shared across all methods/splits/temps for fair comparison.
    spec = POPULATION_SPECS[population]
    panel = sample_population_panel(population, NUM_PERSONAS, seed=seed)
    _print_panel_summary(panel, spec)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE API'}")
    sel_str = (
        "ON"
        if do_model_selection
        else f"OFF — using '{DEFAULT_METHOD}' @ T={DEFAULT_TEMPERATURE}"
    )
    print(f"Model selection: {sel_str}\n")

    # Fast path: no model selection. Run the default (method, T) on all splits
    # directly and report. This skips Phase 1 (all methods), Phase 2 (ranking),
    # and Phase 3 (temperature sweep).
    if not do_model_selection:
        return await _run_default_pipeline(
            panel=panel,
            train_qs=train_qs, val_qs=val_qs, test_qs=test_qs,
            dry_run=dry_run, seed=seed,
            population=population,
        )

    methods = methods_to_run or list(METHODS.keys())

    # =========================================================
    # PHASE 1 — all methods on TRAIN
    # =========================================================
    print("=" * 62)
    print("  PHASE 1 — all methods on TRAIN questions")
    print("=" * 62)

    train_dir = os.path.join(RESULTS_DIR, "train")

    # Build one SurveyTask per method and submit them all as a single
    # Message Batches API call — server-side parallel across methods.
    train_tasks: list[SurveyTask] = []
    train_paths: dict[str, str] = {}
    descriptions_by_method: dict[str, list[str]] = {}
    for method_name in methods:
        method = METHODS[method_name]
        desc_fn = method["desc_fn"]
        descriptions = desc_fn(panel, seed=seed) if method.get("needs_seed") else desc_fn(panel)
        descriptions_by_method[method_name] = descriptions
        preview = descriptions[0][:180].replace("\n", " | ")
        print(f"\n[{method_name}] {method['description']}")
        print(f"  persona[0]: '{preview}…'")

        task = SurveyTask(
            method_name=method_name,
            descriptions=descriptions,
            questions=train_qs,
            temperature=TEMPERATURE,
            seed=seed,
            run_key=method_name,   # one survey per method on train
        )
        train_tasks.append(task)
        train_paths[task.key] = os.path.join(train_dir, f"{method_name}.json")

    train_out = await _run_many(train_tasks, dry_run, train_paths)
    train_results: dict[str, SurveyResults] = {
        name: train_out[name] for name in methods
    }

    # =========================================================
    # PHASE 2 — pick best method from train
    # =========================================================
    print("\n" + "=" * 62)
    print("  PHASE 2 — method selection on TRAIN")
    print("=" * 62)

    train_metrics: dict[str, MethodMetrics] = {
        name: evaluate_method(r, train_qs)
        for name, r in train_results.items()
    }

    print(f"\n{'Method':<28} {'JSD':>8} {'TVD':>8} {'MAE(pp)':>8} {'Plurality':>10} {'Valid':>7}")
    print("-" * 73)
    for name, mm in sorted(train_metrics.items(), key=lambda x: x[1].mean_jsd):
        flag = "" if mm.valid_rate >= MIN_VALID_RATE else "  ← EXCLUDED (low valid rate)"
        print(f"  {name:<26} {mm.mean_jsd:>8.4f} {mm.mean_tvd:>8.4f} "
              f"{mm.mean_mae_pp:>7.1f} {mm.plurality_accuracy:>9.0%} "
              f"{mm.valid_rate:>6.0%}{flag}")

    # Only consider methods with sufficient valid-response rate.
    eligible = {
        name: mm for name, mm in train_metrics.items()
        if mm.valid_rate >= MIN_VALID_RATE
    }
    if not eligible:
        # Fall back to all methods but warn loudly.
        print(f"\n⚠ NO method met valid_rate ≥ {MIN_VALID_RATE:.0%}. "
              f"Falling back to highest valid_rate among all methods.")
        best_method = max(train_metrics, key=lambda m: train_metrics[m].valid_rate)
    else:
        best_method = min(eligible, key=lambda m: eligible[m].mean_jsd)
    best_train_jsd = train_metrics[best_method].mean_jsd
    best_train_valid = train_metrics[best_method].valid_rate
    print(f"\n→ Best method: {best_method}  "
          f"(train JSD={best_train_jsd:.4f}, valid={best_train_valid:.0%})")

    # =========================================================
    # PHASE 3 — temperature sweep on VAL (best method only)
    # =========================================================
    print("\n" + "=" * 62)
    print(f"  PHASE 3 — temperature sweep on VAL  (method: '{best_method}')")
    print(f"  Sweep grid: {TEMPERATURE_SWEEP}")
    print("=" * 62)

    best_descriptions = descriptions_by_method[best_method]

    # Submit ALL temperatures in one Message Batches call — sweep runs in
    # parallel server-side instead of sequentially.
    sweep_tasks: list[SurveyTask] = []
    sweep_paths: dict[str, str] = {}
    for T in TEMPERATURE_SWEEP:
        key = f"{best_method}@T{T}"
        task = SurveyTask(
            method_name=best_method,
            descriptions=best_descriptions,
            questions=val_qs,
            temperature=T,
            seed=seed,
            run_key=key,
        )
        sweep_tasks.append(task)
        sweep_paths[key] = os.path.join(RESULTS_DIR, "val", f"{best_method}_T{T}.json")

    sweep_out = await _run_many(sweep_tasks, dry_run, sweep_paths)

    val_by_temp: dict[float, SurveyResults] = {}
    val_metrics_by_temp: dict[float, MethodMetrics] = {}
    for T in TEMPERATURE_SWEEP:
        r = sweep_out[f"{best_method}@T{T}"]
        val_by_temp[T] = r
        val_metrics_by_temp[T] = evaluate_method(r, val_qs)

    print(f"\n  {'Temp':>5} {'JSD':>8} {'TVD':>8} {'MAE(pp)':>8} {'Plurality':>10} {'Valid':>7}")
    print("  " + "-" * 51)
    for T in TEMPERATURE_SWEEP:
        mm = val_metrics_by_temp[T]
        flag = "" if mm.valid_rate >= MIN_VALID_RATE else "  ← EXCLUDED"
        print(f"  {T:>5.2f} {mm.mean_jsd:>8.4f} {mm.mean_tvd:>8.4f} "
              f"{mm.mean_mae_pp:>7.1f} {mm.plurality_accuracy:>9.0%} "
              f"{mm.valid_rate:>6.0%}{flag}")

    eligible_T = {
        T: mm for T, mm in val_metrics_by_temp.items()
        if mm.valid_rate >= MIN_VALID_RATE
    }
    if not eligible_T:
        print(f"\n  ⚠ NO temperature met valid_rate ≥ {MIN_VALID_RATE:.0%}. "
              f"Falling back to highest valid_rate.")
        best_T = max(val_metrics_by_temp, key=lambda t: val_metrics_by_temp[t].valid_rate)
    else:
        best_T = min(eligible_T, key=lambda t: eligible_T[t].mean_jsd)
    best_val_jsd = val_metrics_by_temp[best_T].mean_jsd
    best_val_valid = val_metrics_by_temp[best_T].valid_rate
    print(f"\n  → Best temperature: T={best_T}  "
          f"(val JSD={best_val_jsd:.4f}, valid={best_val_valid:.0%})")

    # =========================================================
    # PHASE 4 — final TEST eval at (best_method, best_T)
    # =========================================================
    print("\n" + "=" * 62)
    print(f"  PHASE 4 — TEST  (method='{best_method}', T={best_T})")
    print("=" * 62)

    test_path = os.path.join(RESULTS_DIR, "test", f"{best_method}_T{best_T}.json")
    test_results = await _run_one_method(
        best_method, best_descriptions, test_qs,
        dry_run, seed, test_path, temperature=best_T,
    )
    test_metrics = evaluate_method(test_results, test_qs)

    # Use the val run at best_T for the split comparison.
    val_results  = val_by_temp[best_T]
    val_metrics  = val_metrics_by_temp[best_T]

    _print_split_comparison(
        best_method,
        train_metrics[best_method],
        val_metrics,
        test_metrics,
        train_qs, val_qs, test_qs,
        best_T=best_T,
    )

    # Save full reports (includes sweep table)
    _save_all_reports(
        train_metrics, val_metrics, test_metrics,
        best_method, best_T, val_metrics_by_temp,
        train_qs, val_qs, test_qs,
    )

    # =========================================================
    # PHASE 5 — behavioral / social-media stats
    # =========================================================
    print("\n" + "=" * 62)
    print(f"  PHASE 5 — behavioral stats  (best method: '{best_method}')")
    print("=" * 62)

    # Per-option mean ± sd for each split.
    best_train_results = train_results[best_method]
    print_behavioral_summary(best_train_results, train_qs, header="TRAIN — behavioral")
    print_behavioral_summary(val_results,        val_qs,   header="VAL   — behavioral")
    print_behavioral_summary(test_results,       test_qs,  header="TEST  — behavioral")

    # R^3 scatter plot for each TEST question (held-out, unseen by selection).
    plot_dir = os.path.join(RESULTS_DIR, "behavioral")
    for q in test_qs:
        path = os.path.join(plot_dir, f"r3_{q.id}.png")
        plot_behavioral_r3(
            test_results, q, path,
            title_suffix=f"[TEST · {best_method} · T={best_T}]",
        )

    return train_results, val_results, test_results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_split_comparison(
    method_name: str,
    train_mm: MethodMetrics,
    val_mm: MethodMetrics,
    test_mm: MethodMetrics,
    train_qs, val_qs, test_qs,
    best_T: float | None = None,
):
    tag = f"'{method_name}'" + (f" @ T={best_T}" if best_T is not None else "")
    print(f"\n  Split comparison for {tag}")
    print(f"\n  {'Split':<8} {'Questions':<30} {'JSD':>8} {'TVD':>8} {'MAE(pp)':>8} {'Plurality':>10}")
    print("  " + "-" * 68)

    def _row(label, mm, qs):
        ids = ", ".join(q.id for q in qs)
        print(f"  {label:<8} {ids:<30} {mm.mean_jsd:>8.4f} {mm.mean_tvd:>8.4f} "
              f"{mm.mean_mae_pp:>7.1f} {mm.plurality_accuracy:>9.0%}")

    _row("TRAIN", train_mm, train_qs)
    _row("VAL",   val_mm,   val_qs)
    _row("TEST",  test_mm,  test_qs)

    # Generalisation gap: test JSD minus train JSD.
    # Suppress the overfit verdict when test has few valid responses — the gap
    # is dominated by parse failures, not method overfit.
    gap = test_mm.mean_jsd - train_mm.mean_jsd
    if test_mm.valid_rate < MIN_VALID_RATE:
        verdict = f"inconclusive (test valid_rate={test_mm.valid_rate:.0%})"
    elif len(test_qs) < 3:
        verdict = f"tentative (only {len(test_qs)} test question{'s' if len(test_qs)>1 else ''})"
    else:
        verdict = "overfit" if gap > 0.02 else "OK"
    print(f"\n  Generalisation gap (test JSD − train JSD): {gap:+.4f}  [{verdict}]")

    # Per-question detail — ALL splits, with held-out test visually highlighted.
    # TRAIN comes from the Phase-1 method-selection run (at T=1.0);
    # VAL / TEST come from the Phase-3/4 runs at best_T.
    q_lookup = {q.id: q for q in train_qs + val_qs + test_qs}

    # ANSI styling — suppressed if not a TTY.
    import sys
    _use_ansi = sys.stdout.isatty()
    BOLD     = "\033[1m"     if _use_ansi else ""
    RED      = "\033[91m"    if _use_ansi else ""
    YELLOW   = "\033[93m"    if _use_ansi else ""
    DIM      = "\033[2m"     if _use_ansi else ""
    RESET    = "\033[0m"     if _use_ansi else ""
    BG_TEST  = "\033[41;97m" if _use_ansi else ""   # white on red

    splits = [
        ("TRAIN", train_mm, train_qs, False),
        ("VAL",   val_mm,   val_qs,   False),
        ("TEST",  test_mm,  test_qs,  True),   # held out — highlight
    ]

    for split_label, mm, qs, is_heldout in splits:
        if is_heldout:
            banner = f" ★ {split_label} — HELD OUT FROM SELECTION ★ "
            line = "═" * (len(banner) + 4)
            print(f"\n  {BG_TEST}{line}{RESET}")
            print(f"  {BG_TEST}  {banner}  {RESET}")
            print(f"  {BG_TEST}{line}{RESET}")
        else:
            label_color = DIM
            print(f"\n  {label_color}─── {split_label} detail "
                  f"({len(qs)} question{'s' if len(qs)!=1 else ''}) "
                  f"─────────────────────────{RESET}")

        for qm in mm.per_question:
            q = q_lookup[qm.question_id]
            prefix = f"{RED}★ {RESET}" if is_heldout else "  "
            qid_disp = f"{BOLD}{RED}{qm.question_id}{RESET}" if is_heldout else qm.question_id
            print(f"{prefix}[{qid_disp}]  JSD={qm.jsd:.4f}  TVD={qm.tvd:.4f}  "
                  f"MAE={qm.mae_pp:.1f}pp  n={qm.n_valid}")
            if qm.n_valid == 0:
                print(f"{prefix}  {YELLOW}(no valid responses — skipping detail){RESET}")
                continue
            max_opt  = max(qm.predicted_dist, key=qm.predicted_dist.get)
            true_top = max(q.ground_truth,    key=q.ground_truth.get)
            match    = (max_opt == true_top)
            mark     = "✓" if match else "✗"
            mark_col = "" if match else RED
            print(f"{prefix}  Predicted top: '{max_opt}'  |  True top: '{true_top}'  "
                  f"{mark_col}{mark}{RESET}")
            for opt in q.options:
                pred = qm.predicted_dist.get(opt, 0)
                true = q.ground_truth.get(opt, 0)
                bar  = "█" * int(pred * 30)
                print(f"{prefix}  {opt:<40} pred={pred:5.1%}  true={true:5.1%}  "
                      f"Δ={pred-true:+5.1%}  {bar}")


def _save_all_reports(
    train_metrics: dict[str, MethodMetrics],
    val_mm: MethodMetrics,
    test_mm: MethodMetrics,
    best_method: str,
    best_T: float,
    val_metrics_by_temp: dict[float, MethodMetrics],
    train_qs, val_qs, test_qs,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "best_method": best_method,
        "best_temperature": best_T,
        "temperature_sweep_val": {
            str(T): {
                "mean_jsd": mm.mean_jsd,
                "mean_tvd": mm.mean_tvd,
                "mean_mae_pp": mm.mean_mae_pp,
                "plurality_accuracy": mm.plurality_accuracy,
            }
            for T, mm in val_metrics_by_temp.items()
        },
        "train": {
            name: {
                "mean_jsd": mm.mean_jsd,
                "mean_tvd": mm.mean_tvd,
                "mean_mae_pp": mm.mean_mae_pp,
                "plurality_accuracy": mm.plurality_accuracy,
            }
            for name, mm in train_metrics.items()
        },
        "val": {
            "mean_jsd": val_mm.mean_jsd,
            "mean_tvd": val_mm.mean_tvd,
            "mean_mae_pp": val_mm.mean_mae_pp,
            "plurality_accuracy": val_mm.plurality_accuracy,
            "per_question": {
                qm.question_id: {
                    "jsd": qm.jsd, "tvd": qm.tvd, "mae_pp": qm.mae_pp,
                    "predicted": qm.predicted_dist, "ground_truth": qm.ground_truth_dist,
                }
                for qm in val_mm.per_question
            },
        },
        "test": {
            "mean_jsd": test_mm.mean_jsd,
            "mean_tvd": test_mm.mean_tvd,
            "mean_mae_pp": test_mm.mean_mae_pp,
            "plurality_accuracy": test_mm.plurality_accuracy,
            "per_question": {
                qm.question_id: {
                    "jsd": qm.jsd, "tvd": qm.tvd, "mae_pp": qm.mae_pp,
                    "predicted": qm.predicted_dist, "ground_truth": qm.ground_truth_dist,
                }
                for qm in test_mm.per_question
            },
        },
    }
    path = os.path.join(RESULTS_DIR, "report_splits.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM Survey Persona Experiment")
    parser.add_argument(
        "--population",
        type=str,
        choices=list(POPULATION_SPECS.keys()),
        default=DEFAULT_POPULATION,
        help=(
            "Which underlying group the personas should model. "
            "college_educated = bachelor's+ adults across USA (default). "
            "seniors_south = 65+ in FL/AL/LA/TX. "
            "general_us = national adult sample matching poll frames."
        ),
    )
    parser.add_argument(
        "--model_selection",
        action="store_true",
        help=(
            "Run the full 4-phase pipeline: evaluate all methods on TRAIN, pick "
            "the best, sweep temperature on VAL, then eval on TEST. "
            f"Default (off) uses method='{DEFAULT_METHOD}' @ T={DEFAULT_TEMPERATURE}."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Restrict model selection to a single method (requires --model_selection).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Use --dry-run or export the key.")
        sys.exit(1)

    if args.method and not args.model_selection:
        print("--method has no effect without --model_selection "
              f"(default path is fixed to '{DEFAULT_METHOD}' @ T={DEFAULT_TEMPERATURE}).")
        sys.exit(1)

    methods = [args.method] if args.method else None
    asyncio.run(run_experiment(
        population=args.population,
        do_model_selection=args.model_selection,
        methods_to_run=methods,
        dry_run=args.dry_run,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
