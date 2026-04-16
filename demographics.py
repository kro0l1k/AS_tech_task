"""
Demographic distributions for sampling representative US adult personas.

Data sources: US Census Bureau (ACS 2023), Gallup party identification surveys,
Bureau of Labor Statistics. Distributions are approximate but sufficient for
constructing a representative-ish synthetic panel.

Key design decision: We sample demographics as INDEPENDENT marginals rather than
from a joint distribution. This introduces some unrealistic combinations (e.g.,
a 22-year-old retiree) but is much simpler and still captures marginal
demographic balance. A more sophisticated approach would use microdata (PUMS)
to sample correlated attributes — noted as a future improvement.
"""
import random
from dataclasses import dataclass


@dataclass
class DemographicProfile:
    age_bracket: str
    age: int  # specific age within bracket
    gender: str
    race: str
    education: str
    income_bracket: str
    region: str
    community: str
    party: str

    def to_dict(self) -> dict:
        return {
            "age": self.age,
            "gender": self.gender,
            "race": self.race,
            "education": self.education,
            "income": self.income_bracket,
            "region": self.region,
            "community": self.community,
            "political_leaning": self.party,
        }

    def to_text(self) -> str:
        return (
            f"Age: {self.age}\n"
            f"Gender: {self.gender}\n"
            f"Race/Ethnicity: {self.race}\n"
            f"Education: {self.education}\n"
            f"Household income: {self.income_bracket}\n"
            f"Region: {self.region}\n"
            f"Community type: {self.community}\n"
            f"Political leaning: {self.party}"
        )


# --- Marginal distributions ---
# Each is a dict of {category: proportion}

AGE_DIST = {
    "18-29": 0.21,
    "30-44": 0.25,
    "45-59": 0.25,
    "60+": 0.29,
}

AGE_RANGES = {
    "18-29": (18, 29),
    "30-44": (30, 44),
    "45-59": (45, 59),
    "60+": (60, 80),
}

GENDER_DIST = {
    "Male": 0.49,
    "Female": 0.51,
}

RACE_DIST = {
    "White": 0.59,
    "Hispanic or Latino": 0.19,
    "Black or African American": 0.13,
    "Asian": 0.06,
    "Other or multiracial": 0.03,
}

EDUCATION_DIST = {
    "High school diploma or less": 0.38,
    "Some college or associate's degree": 0.27,
    "Bachelor's degree": 0.22,
    "Graduate or professional degree": 0.13,
}

INCOME_DIST = {
    "Under $30,000": 0.20,
    "$30,000 to $50,000": 0.17,
    "$50,000 to $75,000": 0.18,
    "$75,000 to $100,000": 0.14,
    "Over $100,000": 0.31,
}

REGION_DIST = {
    "South": 0.38,
    "West": 0.24,
    "Midwest": 0.21,
    "Northeast": 0.17,
}

COMMUNITY_DIST = {
    "Suburban": 0.55,
    "Urban": 0.31,
    "Rural": 0.14,
}

# Party ID including leaners (Gallup 2024 averages)
PARTY_DIST = {
    "Democrat or lean Democrat": 0.45,
    "Republican or lean Republican": 0.43,
    "Independent, no lean": 0.12,
}


def _weighted_choice(dist: dict[str, float], rng: random.Random) -> str:
    """Sample one category from a distribution dict."""
    categories = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(categories, weights=weights, k=1)[0]


def sample_demographic(rng: random.Random) -> DemographicProfile:
    """Sample a single demographic profile from marginal distributions."""
    age_bracket = _weighted_choice(AGE_DIST, rng)
    lo, hi = AGE_RANGES[age_bracket]
    age = rng.randint(lo, hi)

    return DemographicProfile(
        age_bracket=age_bracket,
        age=age,
        gender=_weighted_choice(GENDER_DIST, rng),
        race=_weighted_choice(RACE_DIST, rng),
        education=_weighted_choice(EDUCATION_DIST, rng),
        income_bracket=_weighted_choice(INCOME_DIST, rng),
        region=_weighted_choice(REGION_DIST, rng),
        community=_weighted_choice(COMMUNITY_DIST, rng),
        party=_weighted_choice(PARTY_DIST, rng),
    )


def sample_panel(n: int, seed: int = 42) -> list[DemographicProfile]:
    """Sample a panel of n demographic profiles."""
    rng = random.Random(seed)
    return [sample_demographic(rng) for _ in range(n)]
