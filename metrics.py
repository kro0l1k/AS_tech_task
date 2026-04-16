"""
Statistical metrics for comparing LLM survey distributions to ground truth.

Three complementary metrics capture different aspects of distributional fidelity:

- Jensen-Shannon Divergence (JSD): information-theoretic measure of similarity
  between two distributions. Bounded [0, 1] when using log base 2. Zero means
  identical distributions.

- Total Variation Distance (TVD): maximum absolute difference in probability
  assigned to any event. Intuitive: "how much probability mass would you need
  to move to make the distributions identical?" Bounded [0, 1].

- Mean Absolute Error (MAE): average absolute difference across all options,
  in percentage points. Most interpretable for practitioners — "on average,
  each option is off by X points."

We also compute chi-squared goodness-of-fit p-values as a significance test.
"""
import numpy as np
from scipy import stats
from dataclasses import dataclass

from ground_truth import SurveyQuestion, QUESTIONS
from survey import SurveyResults


@dataclass
class QuestionMetrics:
    question_id: str
    jsd: float
    tvd: float
    mae_pp: float  # mean absolute error in percentage points
    chi2_stat: float
    chi2_p: float
    predicted_dist: dict[str, float]
    ground_truth_dist: dict[str, float]
    n_valid: int  # number of valid responses


@dataclass
class MethodMetrics:
    method_name: str
    per_question: list[QuestionMetrics]
    mean_jsd: float
    mean_tvd: float
    mean_mae_pp: float
    plurality_accuracy: float  # fraction of questions where top answer matches


def _to_arrays(predicted: dict[str, float], ground_truth: dict[str, float], options: list[str]):
    """Convert dicts to aligned numpy arrays, adding small epsilon to avoid log(0)."""
    eps = 1e-10
    p = np.array([predicted.get(o, 0.0) for o in options], dtype=float) + eps
    q = np.array([ground_truth.get(o, 0.0) for o in options], dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return p, q


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JSD using log base 2, bounded [0, 1]."""
    m = 0.5 * (p + q)
    return float(0.5 * stats.entropy(p, m, base=2) + 0.5 * stats.entropy(q, m, base=2))


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """TVD = 0.5 * L1 distance."""
    return float(0.5 * np.sum(np.abs(p - q)))


def mean_absolute_error_pp(p: np.ndarray, q: np.ndarray) -> float:
    """Mean absolute error in percentage points."""
    return float(np.mean(np.abs(p - q)) * 100)


def evaluate_question(
    results: SurveyResults, question: SurveyQuestion
) -> QuestionMetrics:
    """Evaluate a single question's results against ground truth."""
    pred_dist = results.get_distribution(question)
    n_valid = sum(
        1 for r in results.responses
        if r.question_id == question.id and r.chosen_option is not None
    )

    p, q = _to_arrays(pred_dist, question.ground_truth, question.options)

    # Chi-squared: need observed counts
    obs_counts = np.array([pred_dist.get(o, 0.0) for o in question.options]) * n_valid
    exp_counts = np.array([question.ground_truth[o] for o in question.options]) * n_valid

    # Avoid chi2 with zero expected counts
    mask = exp_counts > 0
    if mask.sum() > 1 and n_valid > 0:
        chi2, chi2_p = stats.chisquare(obs_counts[mask], exp_counts[mask])
    else:
        chi2, chi2_p = 0.0, 1.0

    return QuestionMetrics(
        question_id=question.id,
        jsd=jensen_shannon_divergence(p, q),
        tvd=total_variation_distance(p, q),
        mae_pp=mean_absolute_error_pp(p, q),
        chi2_stat=float(chi2),
        chi2_p=float(chi2_p),
        predicted_dist=pred_dist,
        ground_truth_dist=dict(question.ground_truth),
        n_valid=n_valid,
    )


def evaluate_method(
    results: SurveyResults,
    questions: list[SurveyQuestion] | None = None,
) -> MethodMetrics:
    """Evaluate a method across all questions."""
    if questions is None:
        questions = QUESTIONS

    per_q = [evaluate_question(results, q) for q in questions]

    # Plurality accuracy: does the most-chosen option match ground truth's top?
    plurality_correct = 0
    for qm, q in zip(per_q, questions):
        pred_top = max(qm.predicted_dist, key=qm.predicted_dist.get)
        true_top = max(q.ground_truth, key=q.ground_truth.get)
        if pred_top == true_top:
            plurality_correct += 1

    return MethodMetrics(
        method_name=results.method_name,
        per_question=per_q,
        mean_jsd=float(np.mean([qm.jsd for qm in per_q])),
        mean_tvd=float(np.mean([qm.tvd for qm in per_q])),
        mean_mae_pp=float(np.mean([qm.mae_pp for qm in per_q])),
        plurality_accuracy=plurality_correct / len(questions) if questions else 0.0,
    )
