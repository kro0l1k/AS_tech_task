#!/usr/bin/env python3
"""
LLM Survey Persona Experiment
==============================

Workflow:
  Phase 1 — run ALL methods on TRAIN questions
  Phase 2 — pick best method by mean JSD on train
  Phase 3 — evaluate best method on VAL and TEST questions
  Phase 4 — compare train / val / test to check generalisation

Split: first 80% train, next 10% val, last 10% test
       (at least 1 question guaranteed in val and test)

Usage:
    python main.py                # live API run
    python main.py --dry-run      # simulated responses (no key needed)
    python main.py --analyze-only # re-analyse saved results
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

from config import NUM_PERSONAS, RESULTS_DIR
from ground_truth import QUESTIONS, SurveyQuestion
from demographics import sample_panel
from personas import METHODS
from survey import run_survey, save_results, load_results, SurveyResults, SurveyResponse
from metrics import evaluate_method, MethodMetrics
from analysis import print_full_report, save_report, plot_distributions, generate_comparison_table


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
) -> SurveyResults:
    rng = random.Random(seed)
    results = SurveyResults(method_name=method_name)
    t0 = time.time()
    for q in questions:
        gt = [q.ground_truth[o] for o in q.options]
        noise = [rng.gauss(0, 0.05) for _ in gt]
        perturbed = [max(0.01, p + n) for p, n in zip(gt, noise)]
        s = sum(perturbed)
        perturbed = [p / s for p in perturbed]
        for i in range(len(descriptions)):
            results.responses.append(SurveyResponse(
                persona_index=i,
                question_id=q.id,
                chosen_option=rng.choices(q.options, weights=perturbed, k=1)[0],
                reasoning="[dry run]",
                raw_response="[dry run]",
                options_order=q.options,
            ))
            results.total_calls += 1
    results.elapsed_seconds = time.time() - t0
    print(f"  [{method_name}] dry-run: {results.total_calls} responses")
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
) -> SurveyResults:
    if dry_run:
        results = run_dry_survey(descriptions, questions, method_name, seed=seed)
    else:
        results = await run_survey(descriptions, questions, method_name, seed=seed)
    save_results(results, save_path)
    return results


async def run_experiment(
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

    # --- Sample panel (shared across all methods & splits for fair comparison) ---
    panel = sample_panel(NUM_PERSONAS, seed=seed)
    basket_counts = Counter(p.political_basket for p in panel)
    print(f"\nPanel: {len(panel)} personas  baskets={dict(basket_counts)}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE API'}\n")

    methods = methods_to_run or list(METHODS.keys())

    # =========================================================
    # PHASE 1 — all methods on TRAIN
    # =========================================================
    print("=" * 62)
    print("  PHASE 1 — all methods on TRAIN questions")
    print("=" * 62)

    train_dir = os.path.join(RESULTS_DIR, "train")
    train_results: dict[str, SurveyResults] = {}

    for method_name in methods:
        method = METHODS[method_name]
        print(f"\n[{method_name}] {method['description']}")
        desc_fn = method["desc_fn"]
        descriptions = desc_fn(panel, seed=seed) if method.get("needs_seed") else desc_fn(panel)
        preview = descriptions[0][:180].replace("\n", " | ")
        print(f"  persona[0]: '{preview}…'")

        path = os.path.join(train_dir, f"{method_name}.json")
        r = await _run_one_method(method_name, descriptions, train_qs, dry_run, seed, path)
        train_results[method_name] = r

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

    print(f"\n{'Method':<28} {'JSD':>8} {'TVD':>8} {'MAE(pp)':>8} {'Plurality':>10}")
    print("-" * 65)
    for name, mm in sorted(train_metrics.items(), key=lambda x: x[1].mean_jsd):
        print(f"  {name:<26} {mm.mean_jsd:>8.4f} {mm.mean_tvd:>8.4f} "
              f"{mm.mean_mae_pp:>7.1f} {mm.plurality_accuracy:>9.0%}")

    best_method = min(train_metrics, key=lambda m: train_metrics[m].mean_jsd)
    best_train_jsd = train_metrics[best_method].mean_jsd
    print(f"\n→ Best method: {best_method}  (train JSD={best_train_jsd:.4f})")

    # =========================================================
    # PHASE 3 — best method on VAL and TEST
    # =========================================================
    print("\n" + "=" * 62)
    print(f"  PHASE 3 — '{best_method}' on VAL and TEST")
    print("=" * 62)

    best_method_cfg = METHODS[best_method]
    desc_fn = best_method_cfg["desc_fn"]
    best_descriptions = (
        desc_fn(panel, seed=seed) if best_method_cfg.get("needs_seed") else desc_fn(panel)
    )

    val_path  = os.path.join(RESULTS_DIR, "val",  f"{best_method}.json")
    test_path = os.path.join(RESULTS_DIR, "test", f"{best_method}.json")

    print(f"\n  Running on VAL  ({[q.id for q in val_qs]})…")
    val_results = await _run_one_method(
        best_method, best_descriptions, val_qs, dry_run, seed, val_path
    )

    print(f"\n  Running on TEST ({[q.id for q in test_qs]})…")
    test_results = await _run_one_method(
        best_method, best_descriptions, test_qs, dry_run, seed, test_path
    )

    # =========================================================
    # PHASE 4 — compare train / val / test
    # =========================================================
    print("\n" + "=" * 62)
    print(f"  PHASE 4 — generalisation report: '{best_method}'")
    print("=" * 62)

    val_metrics  = evaluate_method(val_results,  val_qs)
    test_metrics = evaluate_method(test_results, test_qs)

    _print_split_comparison(
        best_method,
        train_metrics[best_method],
        val_metrics,
        test_metrics,
        train_qs, val_qs, test_qs,
    )

    # Save full reports
    _save_all_reports(
        train_metrics, val_metrics, test_metrics,
        best_method, train_qs, val_qs, test_qs,
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
):
    print(f"\n  {'Split':<8} {'Questions':<30} {'JSD':>8} {'TVD':>8} {'MAE(pp)':>8} {'Plurality':>10}")
    print("  " + "-" * 68)

    def _row(label, mm, qs):
        ids = ", ".join(q.id for q in qs)
        print(f"  {label:<8} {ids:<30} {mm.mean_jsd:>8.4f} {mm.mean_tvd:>8.4f} "
              f"{mm.mean_mae_pp:>7.1f} {mm.plurality_accuracy:>9.0%}")

    _row("TRAIN", train_mm, train_qs)
    _row("VAL",   val_mm,   val_qs)
    _row("TEST",  test_mm,  test_qs)

    # Generalisation gap: test JSD minus train JSD
    gap = test_mm.mean_jsd - train_mm.mean_jsd
    print(f"\n  Generalisation gap (test JSD − train JSD): {gap:+.4f} "
          f"({'overfit' if gap > 0.02 else 'OK'})")

    # Per-question detail for val and test
    q_lookup = {q.id: q for q in train_qs + val_qs + test_qs}
    for split_label, mm in [("VAL", val_mm), ("TEST", test_mm)]:
        print(f"\n  --- {split_label} detail ---")
        for qm in mm.per_question:
            q = q_lookup[qm.question_id]
            print(f"  [{qm.question_id}]  JSD={qm.jsd:.4f}  TVD={qm.tvd:.4f}  "
                  f"MAE={qm.mae_pp:.1f}pp  n={qm.n_valid}")
            max_opt = max(qm.predicted_dist, key=qm.predicted_dist.get)
            true_top = max(q.ground_truth, key=q.ground_truth.get)
            print(f"    Predicted top: '{max_opt}'  |  True top: '{true_top}'  "
                  f"{'✓' if max_opt == true_top else '✗'}")
            for opt in q.options:
                pred = qm.predicted_dist.get(opt, 0)
                true = q.ground_truth.get(opt, 0)
                bar  = "█" * int(pred * 30)
                print(f"    {opt:<40} pred={pred:5.1%}  true={true:5.1%}  Δ={pred-true:+5.1%}  {bar}")


def _save_all_reports(
    train_metrics: dict[str, MethodMetrics],
    val_mm: MethodMetrics,
    test_mm: MethodMetrics,
    best_method: str,
    train_qs, val_qs, test_qs,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "best_method": best_method,
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
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--method",       type=str,  default=None)
    parser.add_argument("--seed",         type=int,  default=42)
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Use --dry-run or export the key.")
        sys.exit(1)

    methods = [args.method] if args.method else None
    asyncio.run(run_experiment(
        methods_to_run=methods,
        dry_run=args.dry_run,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
