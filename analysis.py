"""
Analysis and visualization of survey experiment results.

Generates comparison tables, per-question breakdowns, and (optionally)
matplotlib charts. Designed to work both as an importable module and
from the CLI via main.py.
"""
import json
import os
from collections import defaultdict

from ground_truth import QUESTIONS, SurveyQuestion
from survey import SurveyResults
from metrics import evaluate_method, MethodMetrics, QuestionMetrics


def print_method_summary(mm: MethodMetrics):
    """Print a summary table for one method."""
    print(f"\n{'='*60}")
    print(f"  Method: {mm.method_name}")
    print(f"{'='*60}")
    print(f"  Mean JSD:            {mm.mean_jsd:.4f}  (lower is better, 0 = perfect)")
    print(f"  Mean TVD:            {mm.mean_tvd:.4f}  (lower is better, 0 = perfect)")
    print(f"  Mean MAE:            {mm.mean_mae_pp:.1f} pp")
    print(f"  Plurality accuracy:  {mm.plurality_accuracy:.0%}")
    print()


def print_question_detail(qm: QuestionMetrics, question: SurveyQuestion):
    """Print detailed comparison for one question."""
    print(f"  [{question.id}] {question.text[:60]}...")
    print(f"  JSD={qm.jsd:.4f}  TVD={qm.tvd:.4f}  MAE={qm.mae_pp:.1f}pp  "
          f"χ²={qm.chi2_stat:.1f} (p={qm.chi2_p:.3f})  n={qm.n_valid}")

    max_opt_len = max(len(o) for o in question.options)
    for opt in question.options:
        pred = qm.predicted_dist.get(opt, 0)
        true = qm.ground_truth_dist.get(opt, 0)
        diff = pred - true
        bar_pred = "█" * int(pred * 40)
        bar_true = "░" * int(true * 40)
        print(
            f"    {opt:<{max_opt_len}}  "
            f"pred={pred:5.1%}  true={true:5.1%}  "
            f"Δ={diff:+5.1%}  {bar_pred}"
        )
    print()


def print_full_report(all_metrics: list[MethodMetrics]):
    """Print a comprehensive comparison across all methods."""
    print("\n" + "=" * 70)
    print("  LLM SURVEY PERSONA EXPERIMENT — RESULTS")
    print("=" * 70)

    # Summary table
    print(f"\n{'Method':<25} {'JSD':>8} {'TVD':>8} {'MAE(pp)':>8} {'Plurality':>10}")
    print("-" * 62)
    for mm in sorted(all_metrics, key=lambda m: m.mean_jsd):
        print(
            f"{mm.method_name:<25} "
            f"{mm.mean_jsd:>8.4f} "
            f"{mm.mean_tvd:>8.4f} "
            f"{mm.mean_mae_pp:>7.1f} "
            f"{mm.plurality_accuracy:>9.0%}"
        )
    print()

    # Per-question breakdown for each method
    q_lookup = {q.id: q for q in QUESTIONS}
    for mm in all_metrics:
        print_method_summary(mm)
        for qm in mm.per_question:
            print_question_detail(qm, q_lookup[qm.question_id])


def generate_comparison_table(all_metrics: list[MethodMetrics]) -> str:
    """Generate a markdown comparison table."""
    lines = []
    lines.append("| Method | Mean JSD ↓ | Mean TVD ↓ | Mean MAE (pp) ↓ | Plurality Acc ↑ |")
    lines.append("|--------|-----------|-----------|----------------|-----------------|")
    for mm in sorted(all_metrics, key=lambda m: m.mean_jsd):
        lines.append(
            f"| {mm.method_name} | {mm.mean_jsd:.4f} | {mm.mean_tvd:.4f} | "
            f"{mm.mean_mae_pp:.1f} | {mm.plurality_accuracy:.0%} |"
        )
    return "\n".join(lines)


def save_report(all_metrics: list[MethodMetrics], path: str):
    """Save a JSON report of all metrics."""
    report = {}
    for mm in all_metrics:
        report[mm.method_name] = {
            "mean_jsd": mm.mean_jsd,
            "mean_tvd": mm.mean_tvd,
            "mean_mae_pp": mm.mean_mae_pp,
            "plurality_accuracy": mm.plurality_accuracy,
            "per_question": {
                qm.question_id: {
                    "jsd": qm.jsd,
                    "tvd": qm.tvd,
                    "mae_pp": qm.mae_pp,
                    "chi2_stat": qm.chi2_stat,
                    "chi2_p": qm.chi2_p,
                    "predicted": qm.predicted_dist,
                    "ground_truth": qm.ground_truth_dist,
                    "n_valid": qm.n_valid,
                }
                for qm in mm.per_question
            },
        }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


# --- Optional matplotlib visualization ---

def plot_distributions(all_metrics: list[MethodMetrics], output_dir: str = "results"):
    """Generate bar charts comparing predicted vs ground truth distributions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not installed — skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)
    q_lookup = {q.id: q for q in QUESTIONS}

    for q in QUESTIONS:
        n_methods = len(all_metrics)
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 5))
        fig.suptitle(f"{q.text[:80]}...", fontsize=11, y=1.02)

        options = q.options
        x = np.arange(len(options))
        bar_w = 0.6

        # Ground truth
        ax = axes[0]
        gt_vals = [q.ground_truth[o] for o in options]
        ax.bar(x, gt_vals, bar_w, color="gray", alpha=0.7)
        ax.set_title("Ground Truth", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(options, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion")

        # Each method
        for i, mm in enumerate(sorted(all_metrics, key=lambda m: m.mean_jsd)):
            ax = axes[i + 1]
            qm = next(qm for qm in mm.per_question if qm.question_id == q.id)
            pred_vals = [qm.predicted_dist.get(o, 0) for o in options]
            colors = []
            for pv, gv in zip(pred_vals, gt_vals):
                if abs(pv - gv) < 0.05:
                    colors.append("green")
                elif abs(pv - gv) < 0.10:
                    colors.append("orange")
                else:
                    colors.append("red")
            ax.bar(x, pred_vals, bar_w, color=colors, alpha=0.7)
            ax.set_title(f"{mm.method_name}\nJSD={qm.jsd:.3f}", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(options, rotation=45, ha="right", fontsize=7)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"dist_{q.id}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    print(f"  Plots saved to {output_dir}/")
