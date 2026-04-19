"""
Party-conditioned prior distributions for every survey question.

Each entry maps a question id + party label ("D" / "R" / "I") to the
published (or closely-estimated) share each party gives to each option.
Sourced from the same polls listed in ground_truth.py, using their
published partisan crosstabs wherever available. Where a crosstab wasn't
published alongside the topline, the entry is a defensible estimate
scaled from typical D/R gaps on that topic class — those are flagged
with `estimated=True` in the source field.

Used by the `bayesian_update` architecture: instead of asking the LLM to
guess an absolute frequency ("what % of this population picks X"), we
hand it the baseline for its party cell and ask only for the per-persona
log-odds ADJUSTMENT from that baseline based on the specific biography
(voting history, media diet, values, industry). The model does what it's
good at (relative reasoning) rather than what it's bad at (absolute
frequency recall, which is contaminated by training-data toplines).

Posterior = softmax(log(prior) + clip(delta, -2, 2)) per persona.
Panel distribution = mean of per-persona posteriors.

Independent ("I") priors sit between D and R, nudged toward the national
topline to reflect the genuinely-unaligned share of the electorate.
"""
from __future__ import annotations


# Party labels:
#   "D" — Democrat or lean-Democrat
#   "R" — Republican or lean-Republican
#   "I" — true Independent / no lean
#
# Each question id -> dict with:
#   "by_party":  {"D": {opt: p}, "R": {opt: p}, "I": {opt: p}}  each sums to 1.0
#   "overall":   fallback distribution when party is unknown (matches national topline)
#   "source":    where the crosstab came from, with estimated=True flag where needed

PRIORS: dict[str, dict] = {

    # ---------------- Guns ----------------
    # Pew 2023 crosstab: 85% D / 28% R / 55% I want stricter laws.
    # Gallup 2024 topline: 56 / 34 / 10.
    "gun_sales_laws": {
        "by_party": {
            "D": {"More strict": 0.85, "Kept as they are now": 0.13, "Less strict": 0.02},
            "R": {"More strict": 0.28, "Kept as they are now": 0.55, "Less strict": 0.17},
            "I": {"More strict": 0.55, "Kept as they are now": 0.35, "Less strict": 0.10},
        },
        "overall": {"More strict": 0.56, "Kept as they are now": 0.34, "Less strict": 0.10},
        "source": "Pew 2023 partisan crosstabs; Gallup 2024 topline",
    },

    # ---------------- Climate: human contribution ----------------
    # Pew 2024: 78% D vs 18% R say humans contribute a great deal.
    "climate_human_contribution": {
        "by_party": {
            "D": {"Contributes a great deal": 0.78, "Contributes some": 0.15, "Not too much or not at all": 0.07},
            "R": {"Contributes a great deal": 0.18, "Contributes some": 0.37, "Not too much or not at all": 0.45},
            "I": {"Contributes a great deal": 0.42, "Contributes some": 0.31, "Not too much or not at all": 0.27},
        },
        "overall": {"Contributes a great deal": 0.45, "Contributes some": 0.29, "Not too much or not at all": 0.26},
        "source": "Pew Research Center, October 2024 — partisan crosstabs",
    },

    # ---------------- Climate: local impact ----------------
    # Pew 2024 crosstab, estimated scaling.
    "climate_local_impact": {
        "by_party": {
            "D": {"A great deal": 0.45, "Some": 0.38, "Not too much or not at all": 0.17},
            "R": {"A great deal": 0.10, "Some": 0.35, "Not too much or not at all": 0.55},
            "I": {"A great deal": 0.23, "Some": 0.40, "Not too much or not at all": 0.37},
        },
        "overall": {"A great deal": 0.26, "Some": 0.38, "Not too much or not at all": 0.36},
        "source": "Pew Research Center, October 2024 — estimated partisan split",
    },

    # ---------------- Abortion: 3-way circumstance framing ----------------
    # Gallup May 2025 crosstabs: D 48/48/4, R 14/60/26, I ~29/58/13.
    "abortion_legal_circumstances": {
        "by_party": {
            "D": {"Legal under any circumstances": 0.48, "Legal under certain circumstances": 0.48, "Illegal in all circumstances": 0.04},
            "R": {"Legal under any circumstances": 0.14, "Legal under certain circumstances": 0.60, "Illegal in all circumstances": 0.26},
            "I": {"Legal under any circumstances": 0.29, "Legal under certain circumstances": 0.58, "Illegal in all circumstances": 0.13},
        },
        "overall": {"Legal under any circumstances": 0.30, "Legal under certain circumstances": 0.55, "Illegal in all circumstances": 0.15},
        "source": "Gallup May 2025 partisan crosstabs",
    },

    # ---------------- Immigration: pathway vs deport ----------------
    # Quinnipiac Dec 2024: D ~78% pathway, R ~30%, I ~57%.
    "path_vs_deport": {
        "by_party": {
            "D": {"Given a pathway to legal status": 0.78, "Deported from the U.S.": 0.15, "Not sure": 0.07},
            "R": {"Given a pathway to legal status": 0.30, "Deported from the U.S.": 0.60, "Not sure": 0.10},
            "I": {"Given a pathway to legal status": 0.57, "Deported from the U.S.": 0.34, "Not sure": 0.09},
        },
        "overall": {"Given a pathway to legal status": 0.55, "Deported from the U.S.": 0.36, "Not sure": 0.09},
        "source": "Quinnipiac December 2024 partisan crosstabs",
    },

    # ---------------- AI: jobs effect (weakly partisan) ----------------
    "ai_job_automation": {
        "by_party": {
            "D": {"Increase": 0.16, "Decrease": 0.70, "Have no effect": 0.14},
            "R": {"Increase": 0.22, "Decrease": 0.62, "Have no effect": 0.16},
            "I": {"Increase": 0.19, "Decrease": 0.67, "Have no effect": 0.14},
        },
        "overall": {"Increase": 0.19, "Decrease": 0.66, "Have no effect": 0.15},
        "source": "CBS/YouGov March 2026 topline; partisan split estimated (weakly partisan question)",
    },

    # ---------------- AI: work usage (prevalence, not partisan) ----------------
    "ai_work_usage": {
        "by_party": {
            "D": {"All or most of my work": 0.03, "Some of my work": 0.22, "Not much or none of my work": 0.26,
                  "None of my work is done with AI / I don't know if any of my work is done with AI": 0.36, "Not sure": 0.13},
            "R": {"All or most of my work": 0.01, "Some of my work": 0.16, "Not much or none of my work": 0.28,
                  "None of my work is done with AI / I don't know if any of my work is done with AI": 0.40, "Not sure": 0.15},
            "I": {"All or most of my work": 0.02, "Some of my work": 0.18, "Not much or none of my work": 0.27,
                  "None of my work is done with AI / I don't know if any of my work is done with AI": 0.38, "Not sure": 0.15},
        },
        "overall": {"All or most of my work": 0.02, "Some of my work": 0.19, "Not much or none of my work": 0.27,
                    "None of my work is done with AI / I don't know if any of my work is done with AI": 0.38, "Not sure": 0.14},
        "source": "Pew Sept 2025 topline; partisan split estimated (behavioural-prevalence, weakly partisan)",
    },

    # ---------------- Healthcare worry ----------------
    "healthcare_worry": {
        "by_party": {
            "D": {"A great deal": 0.70, "Fair amount": 0.18, "Only a little": 0.09, "Not at all": 0.03},
            "R": {"A great deal": 0.50, "Fair amount": 0.24, "Only a little": 0.17, "Not at all": 0.09},
            "I": {"A great deal": 0.62, "Fair amount": 0.20, "Only a little": 0.12, "Not at all": 0.06},
        },
        "overall": {"A great deal": 0.61, "Fair amount": 0.20, "Only a little": 0.12, "Not at all": 0.07},
        "source": "Gallup March 2026 topline; partisan split estimated (Dems typically higher on healthcare worry)",
    },

    # ---------------- Democracy satisfaction ----------------
    # Asymmetric by incumbency — under current admin (2026 Q1), Dems more dissatisfied.
    "democracy_working": {
        "by_party": {
            "D": {"Satisfied": 0.15, "Dissatisfied": 0.85},
            "R": {"Satisfied": 0.50, "Dissatisfied": 0.50},
            "I": {"Satisfied": 0.28, "Dissatisfied": 0.72},
        },
        "overall": {"Satisfied": 0.31, "Dissatisfied": 0.69},
        "source": "Pew March 2026 topline; partisan split reflects 2025 incumbency flip — D dissatisfaction spiked, R satisfaction rose",
    },

    # ---------------- Israel / Palestine sympathy ----------------
    # Gallup Feb 2026 crosstab: D tilts Palestinian ~54, R strongly Israeli ~60, I ~mixed.
    "israel_sympathy": {
        "by_party": {
            "D": {"More with the Israelis": 0.22, "More with the Palestinians": 0.54, "Both equally / Neither / No opinion": 0.24},
            "R": {"More with the Israelis": 0.60, "More with the Palestinians": 0.18, "Both equally / Neither / No opinion": 0.22},
            "I": {"More with the Israelis": 0.32, "More with the Palestinians": 0.44, "Both equally / Neither / No opinion": 0.24},
        },
        "overall": {"More with the Israelis": 0.36, "More with the Palestinians": 0.41, "Both equally / Neither / No opinion": 0.23},
        "source": "Gallup February 2026 partisan crosstabs",
    },

    # ---------------- Marijuana ----------------
    "marijuana_legalization": {
        "by_party": {
            "D": {"Should be legal": 0.78, "Should not be legal": 0.22},
            "R": {"Should be legal": 0.50, "Should not be legal": 0.50},
            "I": {"Should be legal": 0.66, "Should not be legal": 0.34},
        },
        "overall": {"Should be legal": 0.64, "Should not be legal": 0.36},
        "source": "Gallup October 2025 topline; partisan split from Gallup historical series",
    },

    # ---------------- School vaccine mandate ----------------
    # Reuters/Ipsos Feb 2026: D ~90, R ~60, I ~75.
    "vaccine_school_mandate": {
        "by_party": {
            "D": {"Yes, require vaccination": 0.90, "No, allow unvaccinated children in schools": 0.10},
            "R": {"Yes, require vaccination": 0.60, "No, allow unvaccinated children in schools": 0.40},
            "I": {"Yes, require vaccination": 0.75, "No, allow unvaccinated children in schools": 0.25},
        },
        "overall": {"Yes, require vaccination": 0.75, "No, allow unvaccinated children in schools": 0.25},
        "source": "Reuters/Ipsos February 2026 partisan crosstabs",
    },

    # ---------------- Taxes on high incomes ----------------
    # Pew Feb 2025: D ~83 raised, R ~42 raised.
    "raise_taxes_high_income": {
        "by_party": {
            "D": {"Raised": 0.83, "Kept the same": 0.13, "Lowered": 0.04},
            "R": {"Raised": 0.42, "Kept the same": 0.28, "Lowered": 0.30},
            "I": {"Raised": 0.62, "Kept the same": 0.20, "Lowered": 0.18},
        },
        "overall": {"Raised": 0.62, "Kept the same": 0.20, "Lowered": 0.18},
        "source": "Pew Feb 2025 partisan crosstabs",
    },
}


# Validate on import — priors must sum to 1.0 per cell.
def _validate():
    for qid, entry in PRIORS.items():
        for cell_name, dist in {**entry["by_party"], "overall": entry["overall"]}.items():
            total = sum(dist.values())
            if not (0.999 <= total <= 1.001):
                raise ValueError(
                    f"priors[{qid}][{cell_name}] sums to {total:.4f}, not 1.0"
                )


_validate()


def party_label(party: str) -> str:
    """
    Collapse the DemographicProfile party string to one of {"D","R","I"}.

    DemographicProfile.party values look like "Democrat", "Republican",
    "Independent, lean Democrat", etc. "Lean X" counts as X for prior
    purposes — that's how Pew/Gallup report partisan crosstabs too.
    """
    p = party.lower()
    if "lean democrat" in p or p.startswith("democrat"):
        return "D"
    if "lean republican" in p or p.startswith("republican"):
        return "R"
    return "I"


def prior_for(question_id: str, party: str) -> dict[str, float]:
    """Return the per-option prior distribution for a (question, party) cell."""
    entry = PRIORS.get(question_id)
    if entry is None:
        raise KeyError(f"No prior configured for question id '{question_id}'")
    label = party_label(party)
    return entry["by_party"].get(label, entry["overall"])


def overall_prior(question_id: str) -> dict[str, float]:
    """Return the national-topline fallback for a question."""
    return PRIORS[question_id]["overall"]
