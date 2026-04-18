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
    age_bracket: str             # narrative bucket: 18-29 / 30-44 / 45-59 / 60+
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
    state: str | None = None     # set only when the pop spec constrains state

    def to_dict(self) -> dict:
        d = {
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
        if self.state:
            d["state"] = self.state
        return d

    def to_text(self) -> str:
        influence_str = "\n".join(f"  {i+1}. {src}" for i, src in enumerate(self.influences))
        location = f"{self.region}" + (f" ({self.state})" if self.state else "")
        return (
            f"Age: {self.age}\n"
            f"Gender: {self.gender}\n"
            f"Race/Ethnicity: {self.race}\n"
            f"Education: {self.education}\n"
            f"Household income: {self.income_bracket}\n"
            f"Region: {location}\n"
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
    "High school diploma or less":        0.0,
    "Some college or associate's degree": 0.0,
    "Bachelor's degree":                  0.52,
    "Graduate or professional degree":    0.48,
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


# ---------------------------------------------------------------------------
# Focused panel: 28-year-olds with a degree
# ---------------------------------------------------------------------------
#
# Motivation: independent marginal sampling from the full US distribution
# produces incoherent personas (e.g. "22-year-old retired graduate"). For this
# study we narrow the panel to a single coherent cohort — 28-year-olds holding
# a bachelor's or graduate degree — and let everything else (gender, race,
# region, community, party, political basket, income, influences) vary freely.
#
# Distributions below are tuned for this cohort (ages 25-29, college-educated)
# based on Pew / Census patterns: higher Dem lean, higher urban share, lower
# bottom-income tail, no rural retirees.

FOCUSED_EDUCATION_DIST = {
    "Bachelor's degree":               0.80,
    "Graduate or professional degree": 0.20,
}

# Early-career income for a 28yo with a degree.
FOCUSED_INCOME_DIST = {
    "Under $30,000":        0.10,
    "$30,000 to $50,000":   0.25,
    "$50,000 to $75,000":   0.30,
    "$75,000 to $100,000":  0.20,
    "Over $100,000":        0.15,
}

# College-educated 28yo skew urban/suburban.
FOCUSED_COMMUNITY_DIST = {
    "Urban":    0.40,
    "Suburban": 0.50,
    "Rural":    0.10,
}

# Young college grads skew Democratic (Pew 2024).
FOCUSED_PARTY_DIST = {
    "Democrat or lean Democrat":     0.58,
    "Republican or lean Republican": 0.30,
    "Independent, no lean":          0.12,
}


def sample_focused_demographic(rng: random.Random, age: int = 28) -> DemographicProfile:
    """Sample one 28-year-old with a degree. Other attributes vary."""
    party = _weighted_choice(FOCUSED_PARTY_DIST, rng)
    basket = _weighted_choice(BASKET_BY_PARTY[party], rng)

    basket_pool = INFLUENCE_POOLS[basket]
    basket_picks = rng.sample(basket_pool, min(3, len(basket_pool)))
    general_picks = rng.sample(GENERAL_INFLUENCES, 2)
    influences = basket_picks + general_picks

    return DemographicProfile(
        age_bracket="18-29",
        age=age,
        gender=_weighted_choice(GENDER_DIST, rng),
        race=_weighted_choice(RACE_DIST, rng),
        education=_weighted_choice(FOCUSED_EDUCATION_DIST, rng),
        income_bracket=_weighted_choice(FOCUSED_INCOME_DIST, rng),
        region=_weighted_choice(REGION_DIST, rng),
        community=_weighted_choice(FOCUSED_COMMUNITY_DIST, rng),
        party=party,
        political_basket=basket,
        influences=influences,
    )


def sample_focused_panel(
    n: int, seed: int = 42, age: int = 28
) -> list[DemographicProfile]:
    """Coherent panel of n 28-year-olds with degrees; other attrs vary."""
    rng = random.Random(seed)
    return [sample_focused_demographic(rng, age=age) for _ in range(n)]


# ---------------------------------------------------------------------------
# Population specs — three underlying groups the personas can model
# ---------------------------------------------------------------------------
#
# Each PopulationSpec bundles the marginal distributions used to sample one
# persona. Distributions are grounded in Census / ACS / Pew figures (sources
# cited inline). Use sample_population_panel("name", n, seed) to draw a panel.

def _narrative_bucket(age: int) -> str:
    """Map a numeric age to the narrative-experience bucket used in personas.py."""
    if age <= 29:
        return "18-29"
    if age <= 44:
        return "30-44"
    if age <= 59:
        return "45-59"
    return "60+"


@dataclass
class PopulationSpec:
    name: str
    description: str
    age_dist: dict[str, float]
    age_ranges: dict[str, tuple[int, int]]
    gender_dist: dict[str, float]
    race_dist: dict[str, float]
    education_dist: dict[str, float]
    income_dist: dict[str, float]
    region_dist: dict[str, float]
    community_dist: dict[str, float]
    party_dist: dict[str, float]
    state_dist: dict[str, float] | None = None  # restricts to specific states


# --- College-educated adults across the USA (bachelor's+, ages 22-65) -------
# Sources:
#   - Census ACS 2023, Table S1501 (educational attainment by age)
#   - Pew 2024 "Party identification by education" — college grads lean Dem
#   - BLS Current Population Survey (income by education)
COLLEGE_EDUCATED = PopulationSpec(
    name="college_educated",
    description="College-educated adults across the USA (bachelor's+, ages 22-65)",
    age_dist={"22-34": 0.30, "35-44": 0.25, "45-54": 0.22, "55-65": 0.23},
    age_ranges={
        "22-34": (22, 34), "35-44": (35, 44),
        "45-54": (45, 54), "55-65": (55, 65),
    },
    gender_dist={"Male": 0.48, "Female": 0.52},
    race_dist={
        "White":                     0.63,
        "Hispanic or Latino":        0.10,
        "Black or African American": 0.10,
        "Asian":                     0.13,
        "Other or multiracial":      0.04,
    },
    education_dist={
        "High school diploma or less":        0.0,
        "Some college or associate's degree": 0.0,
        "Bachelor's degree":                  0.65,
        "Graduate or professional degree":    0.35,
    },
    income_dist={
        "Under $30,000":       0.08,
        "$30,000 to $50,000":  0.15,
        "$50,000 to $75,000":  0.20,
        "$75,000 to $100,000": 0.20,
        "Over $100,000":       0.37,
    },
    region_dist={"South": 0.35, "West": 0.25, "Midwest": 0.19, "Northeast": 0.21},
    community_dist={"Urban": 0.38, "Suburban": 0.52, "Rural": 0.10},
    party_dist={
        "Democrat or lean Democrat":     0.58,
        "Republican or lean Republican": 0.37,
        "Independent, no lean":          0.05,
    },
)

# --- Seniors (65+) in FL / AL / LA / TX ------------------------------------
# State weights follow Census senior-population counts (2023): FL 4.9M,
# TX 4.1M, AL 0.9M, LA 0.8M → rescaled to sum to 1.
# Race is a weighted blend of state-level 65+ race distributions (ACS 2023).
# Party leans R — Cook PVI: all four states are red-leaning.
# Education and income reflect the 65+ cohort nationally, adjusted slightly
# toward lower levels typical of the deep South.
SENIORS_SOUTH = PopulationSpec(
    name="seniors_south",
    description="Seniors (65+) in Florida, Alabama, Louisiana, and Texas",
    age_dist={"65-74": 0.58, "75-84": 0.30, "85+": 0.12},
    age_ranges={"65-74": (65, 74), "75-84": (75, 84), "85+": (85, 92)},
    gender_dist={"Male": 0.45, "Female": 0.55},
    race_dist={
        "White":                     0.72,
        "Hispanic or Latino":        0.14,
        "Black or African American": 0.12,
        "Asian":                     0.02,
        "Other or multiracial":      0.00,
    },
    education_dist={
        "High school diploma or less":        0.40,
        "Some college or associate's degree": 0.28,
        "Bachelor's degree":                  0.20,
        "Graduate or professional degree":    0.12,
    },
    income_dist={
        "Under $30,000":       0.32,
        "$30,000 to $50,000":  0.28,
        "$50,000 to $75,000":  0.20,
        "$75,000 to $100,000": 0.11,
        "Over $100,000":       0.09,
    },
    region_dist={"South": 1.00, "West": 0.0, "Midwest": 0.0, "Northeast": 0.0},
    community_dist={"Urban": 0.22, "Suburban": 0.52, "Rural": 0.26},
    party_dist={
        "Democrat or lean Democrat":     0.36,
        "Republican or lean Republican": 0.55,
        "Independent, no lean":          0.09,
    },
    state_dist={"Florida": 0.45, "Texas": 0.38, "Alabama": 0.09, "Louisiana": 0.08},
)

# --- General US adult population (matches poll sample frames) ---------------
# Uses the same marginals as the existing AGE_DIST / RACE_DIST / … set, which
# reflect Census 2023 national adult distributions. Education matches ACS
# national attainment for adults 25+. This is the frame Gallup / Pew polls
# are typically weighted to.
GENERAL_US = PopulationSpec(
    name="general_us",
    description="General US adult population — matches national poll sample frames",
    age_dist={"18-29": 0.21, "30-44": 0.25, "45-59": 0.25, "60+": 0.29},
    age_ranges={
        "18-29": (18, 29), "30-44": (30, 44),
        "45-59": (45, 59), "60+": (60, 80),
    },
    gender_dist={"Male": 0.49, "Female": 0.51},
    race_dist={
        "White":                     0.59,
        "Hispanic or Latino":        0.19,
        "Black or African American": 0.13,
        "Asian":                     0.06,
        "Other or multiracial":      0.03,
    },
    education_dist={
        "High school diploma or less":        0.37,
        "Some college or associate's degree": 0.25,
        "Bachelor's degree":                  0.22,
        "Graduate or professional degree":    0.16,
    },
    income_dist={
        "Under $30,000":       0.20,
        "$30,000 to $50,000":  0.17,
        "$50,000 to $75,000":  0.18,
        "$75,000 to $100,000": 0.14,
        "Over $100,000":       0.31,
    },
    region_dist={"South": 0.38, "West": 0.24, "Midwest": 0.21, "Northeast": 0.17},
    community_dist={"Suburban": 0.55, "Urban": 0.31, "Rural": 0.14},
    party_dist={
        "Democrat or lean Democrat":     0.45,
        "Republican or lean Republican": 0.43,
        "Independent, no lean":          0.12,
    },
)

POPULATION_SPECS: dict[str, PopulationSpec] = {
    "college_educated": COLLEGE_EDUCATED,
    "seniors_south":    SENIORS_SOUTH,
    "general_us":       GENERAL_US,
}


def sample_from_spec(spec: PopulationSpec, rng: random.Random) -> DemographicProfile:
    """Draw one persona from a PopulationSpec."""
    party  = _weighted_choice(spec.party_dist, rng)
    basket = _weighted_choice(BASKET_BY_PARTY[party], rng)

    basket_pool = INFLUENCE_POOLS[basket]
    basket_picks  = rng.sample(basket_pool, min(3, len(basket_pool)))
    general_picks = rng.sample(GENERAL_INFLUENCES, 2)
    influences = basket_picks + general_picks

    age_bucket = _weighted_choice(spec.age_dist, rng)
    lo, hi = spec.age_ranges[age_bucket]
    age = rng.randint(lo, hi)

    # region_dist may concentrate (e.g. South=1.0 for seniors_south) — that's fine.
    region_choices = {r: w for r, w in spec.region_dist.items() if w > 0}
    region = _weighted_choice(region_choices, rng)

    state = None
    if spec.state_dist:
        state = _weighted_choice(spec.state_dist, rng)

    return DemographicProfile(
        age_bracket=_narrative_bucket(age),
        age=age,
        gender=_weighted_choice(spec.gender_dist, rng),
        race=_weighted_choice(spec.race_dist, rng),
        education=_weighted_choice(spec.education_dist, rng),
        income_bracket=_weighted_choice(spec.income_dist, rng),
        region=region,
        community=_weighted_choice(spec.community_dist, rng),
        party=party,
        political_basket=basket,
        influences=influences,
        state=state,
    )


def sample_population_panel(
    population: str, n: int, seed: int = 42,
) -> list[DemographicProfile]:
    """
    Draw a panel of n personas for the named population.

    population ∈ {"college_educated", "seniors_south", "general_us"}
    """
    if population not in POPULATION_SPECS:
        raise ValueError(
            f"Unknown population '{population}'. "
            f"Choices: {list(POPULATION_SPECS)}"
        )
    spec = POPULATION_SPECS[population]
    rng = random.Random(seed)
    return [sample_from_spec(spec, rng) for _ in range(n)]
