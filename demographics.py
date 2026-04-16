"""
Demographic distributions for sampling representative US adult personas.

Now includes:
- Political alignment basket (progressive / neo_liberal / neo_conservative / maga / alt_right)
- Top 5 information influences (3 from basket pool + 2 from general mainstream)

Basket assignment is conditioned on party lean, reflecting empirical patterns in
media consumption research (Reuters Institute Digital News Report, Pew Media Attitudes).
"""
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Political basket definitions
# ---------------------------------------------------------------------------

POLITICAL_BASKETS = ["progressive", "neo_liberal", "neo_conservative", "maga", "alt_right"]

# Basket distribution conditioned on party lean.
# Proportions are approximate, drawn from Pew political typology research.
BASKET_BY_PARTY: dict[str, dict[str, float]] = {
    "Democrat or lean Democrat": {
        "progressive":     0.30,
        "neo_liberal":     0.65,
        "neo_conservative": 0.04,
        "maga":            0.01,
        "alt_right":       0.00,
    },
    "Republican or lean Republican": {
        "progressive":     0.00,
        "neo_liberal":     0.05,
        "neo_conservative": 0.35,
        "maga":            0.52,
        "alt_right":       0.08,
    },
    "Independent, no lean": {
        "progressive":     0.12,
        "neo_liberal":     0.28,
        "neo_conservative": 0.28,
        "maga":            0.22,
        "alt_right":       0.10,
    },
}

# Influence pools: sources each basket actively consumes.
# Top 3 per persona are drawn from this basket-specific pool.
INFLUENCE_POOLS: dict[str, list[str]] = {
    "progressive": [
        "Democracy Now!",
        "The Young Turks (YouTube/podcast)",
        "AOC's social media / newsletters",
        "Jacobin Magazine",
        "Chapo Trap House podcast",
        "The Intercept",
        "Bernie Sanders email updates",
        "MSNBC (Rachel Maddow / Joy Reid)",
        "Pod Save America",
        "Noam Chomsky / Naomi Klein books",
    ],
    "neo_liberal": [
        "New York Times",
        "NPR Morning Edition",
        "The Atlantic",
        "The Ezra Klein Show",
        "CNN",
        "The New Yorker",
        "The Daily (NYT podcast)",
        "Vox",
        "Pod Save America",
        "Obama / Biden Democratic Party content",
    ],
    "neo_conservative": [
        "Wall Street Journal (opinion pages)",
        "National Review",
        "The Dispatch (David French / Jonah Goldberg)",
        "Fox News (straight news side)",
        "Commentary magazine",
        "The Bulwark",
        "Ben Shapiro Show (policy commentary)",
        "Dennis Prager / PragerU",
        "Hugh Hewitt radio",
        "Heritage Foundation publications",
    ],
    "maga": [
        "Fox News (Hannity / Ingraham / Carlson)",
        "Breitbart News",
        "Trump's Truth Social",
        "Charlie Kirk / Turning Point USA",
        "Newsmax",
        "Dan Bongino Show",
        "Mark Levin radio",
        "OAN",
        "Sebastian Gorka America First",
        "Gateway Pundit",
    ],
    "alt_right": [
        "Infowars / Alex Jones",
        "Andrew Tate content",
        "4chan /pol/ and fringe forums",
        "Far-right X/Twitter accounts (post-Musk)",
        "Nick Fuentes / America First streams",
        "Joe Rogan Experience (edgier episodes)",
        "Rumble alternative media",
        "Gavin McInnes content",
        "Telegram far-right channels",
        "Daily Wire provocateurs",
    ],
}

# General mainstream sources everyone might encounter regardless of basket.
# Bottom 2 influences per persona are drawn from here.
GENERAL_INFLUENCES: list[str] = [
    "local TV news",
    "Facebook (friends and family posts)",
    "network evening news (ABC / NBC / CBS)",
    "conversations with coworkers or neighbors",
    "church or community group discussions",
    "Reddit (general browsing)",
    "YouTube (general algorithm)",
    "local talk radio",
]


# ---------------------------------------------------------------------------
# Demographic profile
# ---------------------------------------------------------------------------

@dataclass
class DemographicProfile:
    age_bracket: str
    age: int
    gender: str
    race: str
    education: str
    income_bracket: str
    region: str
    community: str
    party: str
    political_basket: str        # one of POLITICAL_BASKETS
    influences: list = field(default_factory=list)  # list[str], top 5

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
            "political_basket": self.political_basket,
            "influences": self.influences,
        }

    def to_text(self) -> str:
        influence_str = "\n".join(f"  {i+1}. {src}" for i, src in enumerate(self.influences))
        return (
            f"Age: {self.age}\n"
            f"Gender: {self.gender}\n"
            f"Race/Ethnicity: {self.race}\n"
            f"Education: {self.education}\n"
            f"Household income: {self.income_bracket}\n"
            f"Region: {self.region}\n"
            f"Community type: {self.community}\n"
            f"Political identity: {self.political_basket} "
            f"(broadly {self.party})\n"
            f"Top 5 information sources:\n{influence_str}"
        )


# ---------------------------------------------------------------------------
# Marginal distributions
# ---------------------------------------------------------------------------

AGE_DIST = {
    "18-29": 0.21,
    "30-44": 0.25,
    "45-59": 0.25,
    "60+":   0.29,
}

AGE_RANGES = {
    "18-29": (18, 29),
    "30-44": (30, 44),
    "45-59": (45, 59),
    "60+":   (60, 80),
}

GENDER_DIST = {
    "Male":   0.49,
    "Female": 0.51,
}

RACE_DIST = {
    "White":                      0.59,
    "Hispanic or Latino":         0.19,
    "Black or African American":  0.13,
    "Asian":                      0.06,
    "Other or multiracial":       0.03,
}

EDUCATION_DIST = {
    "High school diploma or less":        0.38,
    "Some college or associate's degree": 0.27,
    "Bachelor's degree":                  0.22,
    "Graduate or professional degree":    0.13,
}

INCOME_DIST = {
    "Under $30,000":        0.20,
    "$30,000 to $50,000":   0.17,
    "$50,000 to $75,000":   0.18,
    "$75,000 to $100,000":  0.14,
    "Over $100,000":        0.31,
}

REGION_DIST = {
    "South":     0.38,
    "West":      0.24,
    "Midwest":   0.21,
    "Northeast": 0.17,
}

COMMUNITY_DIST = {
    "Suburban": 0.55,
    "Urban":    0.31,
    "Rural":    0.14,
}

PARTY_DIST = {
    "Democrat or lean Democrat":    0.45,
    "Republican or lean Republican": 0.43,
    "Independent, no lean":         0.12,
}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _weighted_choice(dist: dict[str, float], rng: random.Random) -> str:
    categories = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(categories, weights=weights, k=1)[0]


def sample_demographic(rng: random.Random) -> DemographicProfile:
    """Sample a single demographic profile from marginal distributions."""
    party = _weighted_choice(PARTY_DIST, rng)

    # Assign political basket conditioned on party
    basket = _weighted_choice(BASKET_BY_PARTY[party], rng)

    # Sample influences: 3 from basket pool + 2 from general mainstream
    basket_pool = INFLUENCE_POOLS[basket]
    basket_picks = rng.sample(basket_pool, min(3, len(basket_pool)))
    general_picks = rng.sample(GENERAL_INFLUENCES, 2)
    influences = basket_picks + general_picks

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
        party=party,
        political_basket=basket,
        influences=influences,
    )


def sample_panel(n: int, seed: int = 42) -> list[DemographicProfile]:
    """Sample a panel of n demographic profiles."""
    rng = random.Random(seed)
    return [sample_demographic(rng) for _ in range(n)]
