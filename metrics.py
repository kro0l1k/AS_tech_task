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
    valid_rate: float = 1.0    # fraction of responses that parsed successfully


# ---------------------------------------------------------------------------
# Behavioral (social-media) metrics
# ---------------------------------------------------------------------------

@dataclass
class BehavioralStats:
    """Mean + variance for the 3 behavioral dimensions within a group."""
    n: int
    post_mean:   float
    post_var:    float
    argue_mean:  float
    argue_var:   float
    debate_mean: float
    debate_var:  float

    @classmethod
    def from_samples(cls, post: list[float], argue: list[float], debate: list[float]):
        n = len(post)
        if n == 0:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        post_a  = np.asarray(post,   dtype=float)
        argue_a = np.asarray(argue,  dtype=float)
        debate_a = np.asarray(debate, dtype=float)
        # Population variance (ddof=0) — we have the whole sample of personas.
        return cls(
            n=n,
            post_mean=float(post_a.mean()),
            post_var=float(post_a.var(ddof=0)),
            argue_mean=float(argue_a.mean()),
            argue_var=float(argue_a.var(ddof=0)),
            debate_mean=float(debate_a.mean()),
            debate_var=float(debate_a.var(ddof=0)),
        )


def behavioral_by_question_option(
    results: SurveyResults,
    questions: list[SurveyQuestion],
) -> dict[str, dict[str, BehavioralStats]]:
    """
    Group responses by (question_id, chosen_option) and compute per-dim stats.

    Only responses with ALL three behavioral fields present are counted — a
    partial row (e.g. letter parsed but behavioral tail missing) is dropped so
    means and variances are over the same n.

    Returns:
        { question_id: { option_text: BehavioralStats, ... }, ... }
    """
    out: dict[str, dict[str, BehavioralStats]] = {}
    for q in questions:
        buckets: dict[str, tuple[list, list, list]] = {
            opt: ([], [], []) for opt in q.options
        }
        for r in results.responses:
            if r.question_id != q.id:
                continue
            if r.chosen_option is None:
                continue
            if None in (r.post_likelihood, r.argue_likelihood, r.debate_frequency):
                continue
            if r.chosen_option not in buckets:
                continue  # unexpected option text
            p_list, a_list, d_list = buckets[r.chosen_option]
            p_list.append(r.post_likelihood)
            a_list.append(r.argue_likelihood)
            d_list.append(r.debate_frequency)
        out[q.id] = {
            opt: BehavioralStats.from_samples(p, a, d)
            for opt, (p, a, d) in buckets.items()
        }
    return out


def behavioral_raw_samples(
    results: SurveyResults,
    question: SurveyQuestion,
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """
    Raw per-respondent (post, argue, debate) samples grouped by chosen_option.
    Used for R^3 scatter plots.
    """
    buckets: dict[str, tuple[list, list, list]] = {
        opt: ([], [], []) for opt in question.options
    }
    for r in results.responses:
        if r.question_id != question.id:
            continue
        if r.chosen_option is None:
            continue
        if None in (r.post_likelihood, r.argue_likelihood, r.debate_frequency):
            continue
        if r.chosen_option not in buckets:
            continue
        p, a, d = buckets[r.chosen_option]
        p.append(r.post_likelihood)
        a.append(r.argue_likelihood)
        d.append(r.debate_frequency)
    return buckets


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

    # Chi-squared: need observed/expected counts with matching totals.
    # Ground-truth percentages may not sum to exactly 1.0 due to rounding,
    # and we may drop zero-expected categories below.
    obs_counts = np.array([pred_dist.get(o, 0.0) for o in question.options]) * n_valid
    exp_counts = np.array([question.ground_truth[o] for o in question.options]) * n_valid

    # Avoid chi2 with zero expected counts.
    mask = exp_counts > 0
    if mask.sum() > 1 and n_valid > 0:
        obs_masked = obs_counts[mask]
        exp_masked = exp_counts[mask]

        # scipy.stats.chisquare requires sums to match exactly (within tolerance).
        exp_total = exp_masked.sum()
        obs_total = obs_masked.sum()
        if exp_total > 0 and obs_total > 0:
            exp_masked = exp_masked * (obs_total / exp_total)
            chi2, chi2_p = stats.chisquare(obs_masked, exp_masked)
        else:
            chi2, chi2_p = 0.0, 1.0
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
    # Only count questions with at least one valid response — otherwise the
    # epsilon-smoothed distribution gives a meaningless "winner."
    plurality_correct = 0
    plurality_denom = 0
    for qm, q in zip(per_q, questions):
        if qm.n_valid == 0:
            continue
        plurality_denom += 1
        pred_top = max(qm.predicted_dist, key=qm.predicted_dist.get)
        true_top = max(q.ground_truth, key=q.ground_truth.get)
        if pred_top == true_top:
            plurality_correct += 1

    # Valid-response rate = total valid responses / total responses we asked for.
    # Guards selection against methods/temps that all-errored (n=0 on every q).
    total_asked = sum(
        1 for r in results.responses if r.question_id in {q.id for q in questions}
    )
    total_valid = sum(qm.n_valid for qm in per_q)
    valid_rate = (total_valid / total_asked) if total_asked > 0 else 0.0

    return MethodMetrics(
        method_name=results.method_name,
        per_question=per_q,
        mean_jsd=float(np.mean([qm.jsd for qm in per_q])),
        mean_tvd=float(np.mean([qm.tvd for qm in per_q])),
        mean_mae_pp=float(np.mean([qm.mae_pp for qm in per_q])),
        plurality_accuracy=(plurality_correct / plurality_denom) if plurality_denom else 0.0,
        valid_rate=valid_rate,
    )
