#!/usr/bin/env python3
"""
LLM Survey Persona Experiment
==============================

Main entry point. Runs 100 LLM personas through 7 survey questions using
5 different persona-creation methods, then evaluates each method's ability
to reproduce known US public opinion distributions.

Usage:
    # Run full experiment (requires ANTHROPIC_API_KEY)
    python main.py

    # Run with simulated random responses (no API key needed)
    python main.py --dry-run

    # Run a single method
    python main.py --method demographic_baseline

    # Analyze previously saved results
    python main.py --analyze-only
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

from config import NUM_PERSONAS, RESULTS_DIR
from ground_truth import QUESTIONS
from demographics import sample_panel
from personas import METHODS
from survey import run_survey, save_results, load_results, SurveyResults, SurveyResponse
from metrics import evaluate_method
from analysis import print_full_report, save_report, plot_distributions


def run_dry_survey(
    persona_prompts: list[str],
    questions: list,
    method_name: str,
    seed: int = 42,
) -> SurveyResults:
    """
    Simulated survey using random responses — for testing the pipeline
    without API calls. Responses are drawn from a slight perturbation of
    ground truth to simulate realistic-ish results.
    """
    rng = random.Random(seed)
    results = SurveyResults(method_name=method_name)
    t0 = time.time()

    for q in questions:
        # Perturb ground truth slightly for this "method"
        gt_probs = [q.ground_truth[o] for o in q.options]
        noise = [rng.gauss(0, 0.05) for _ in gt_probs]
        perturbed = [max(0.01, p + n) for p, n in zip(gt_probs, noise)]
        total = sum(perturbed)
        perturbed = [p / total for p in perturbed]

        for i in range(len(persona_prompts)):
            chosen = rng.choices(q.options, weights=perturbed, k=1)[0]
            results.responses.append(SurveyResponse(
                persona_index=i,
                question_id=q.id,
                chosen_option=chosen,
                reasoning="[dry run]",
                raw_response="[dry run]",
                options_order=q.options,
            ))
            results.total_calls += 1

    results.elapsed_seconds = time.time() - t0
    print(f"  [{method_name}] Dry run complete: {results.total_calls} simulated responses")
    return results


async def run_experiment(
    methods_to_run: list[str] | None = None,
    dry_run: bool = False,
    seed: int = 42,
):
    """Run the full experiment across all specified methods."""
    
    print("=" * 60)
    print("  LLM SURVEY PERSONA EXPERIMENT")
    print("=" * 60)
    print(f"  Personas: {NUM_PERSONAS}")
    print(f"  Questions: {len(QUESTIONS)}")
    print(f"  Methods: {methods_to_run or list(METHODS.keys())}")
    print(f"  Mode: {'DRY RUN (random responses)' if dry_run else 'LIVE (API calls)'}")
    print()

    # Sample demographic panel (shared across methods for fair comparison)
    print("Sampling demographic panel...")
    panel = sample_panel(NUM_PERSONAS, seed=seed)
    print(f"  Panel of {len(panel)} personas sampled.")

    # Quick demographic summary
    from collections import Counter
    party_counts = Counter(p.party for p in panel)
    print(f"  Party split: {dict(party_counts)}")
    print()

    # Run each method
    methods = methods_to_run or list(METHODS.keys())
    all_results = []

    for method_name in methods:
        method = METHODS[method_name]
        print(f"\n--- Method: {method_name} ---")
        print(f"    {method['description']}")

        # Build per-persona descriptions (batch mode)
        desc_fn = method["desc_fn"]
        if method.get("needs_seed"):
            descriptions = desc_fn(panel, seed=seed)
        else:
            descriptions = desc_fn(panel)

        # Show example description
        print(f"    Example persona description (persona #0):")
        preview = descriptions[0][:300].replace("\n", " | ")
        print(f"    '{preview}...'")
        print()

        # Run survey
        if dry_run:
            results = run_dry_survey(descriptions, QUESTIONS, method_name, seed=seed + hash(method_name))
        else:
            results = await run_survey(descriptions, QUESTIONS, method_name, seed=seed)

        # Save results
        result_path = os.path.join(RESULTS_DIR, f"{method_name}.json")
        save_results(results, result_path)
        print(f"    Results saved to {result_path}")

        all_results.append(results)

    return all_results


def analyze_results(results_dir: str = RESULTS_DIR):
    """Load and analyze previously saved results."""
    all_results = []
    for fname in os.listdir(results_dir):
        if fname.endswith(".json") and not fname.startswith("report"):
            path = os.path.join(results_dir, fname)
            try:
                results = load_results(path)
                all_results.append(results)
            except Exception as e:
                print(f"  Warning: could not load {path}: {e}")

    if not all_results:
        print("No results found to analyze.")
        return

    # Evaluate all methods
    all_metrics = [evaluate_method(r, QUESTIONS) for r in all_results]

    # Print report
    print_full_report(all_metrics)

    # Save report
    report_path = os.path.join(results_dir, "report.json")
    save_report(all_metrics, report_path)
    print(f"\nReport saved to {report_path}")

    # Generate plots
    plot_distributions(all_metrics, results_dir)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="LLM Survey Persona Experiment"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use random responses instead of API calls"
    )
    parser.add_argument(
        "--method", type=str, default=None,
        help="Run a single method (e.g., demographic_baseline)"
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Only analyze previously saved results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.analyze_only:
        analyze_results()
        return

    # Validate method name
    methods = None
    if args.method:
        if args.method not in METHODS:
            print(f"Unknown method: {args.method}")
            print(f"Available: {list(METHODS.keys())}")
            sys.exit(1)
        methods = [args.method]

    # Check API key for live runs
    if not args.dry_run:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY not set.")
            print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
            print("Or use --dry-run to test without API calls.")
            sys.exit(1)

    # Run experiment
    asyncio.run(run_experiment(
        methods_to_run=methods,
        dry_run=args.dry_run,
        seed=args.seed,
    ))

    # Analyze
    print("\n\n" + "=" * 60)
    print("  ANALYSIS")
    print("=" * 60)
    analyze_results()


if __name__ == "__main__":
    main()
