"""
Five methods for constructing LLM persona prompts.

Each method takes a list of DemographicProfiles and returns a list of system
prompts that will be used to condition the LLM before asking survey questions.

The methods represent different hypotheses about what makes LLM personas
produce realistic opinion distributions:

1. DEMOGRAPHIC BASELINE  — demographics alone as direct attributes
2. NARRATIVE BACKSTORY   — rich biographical context grounding the persona
3. VALUE-BELIEF ANCHORING — moral foundations & values instead of demographics
4. DISTRIBUTION-AWARE    — meta-prompt informing model of its ensemble role
5. COGNITIVE DELIBERATION — chain-of-thought reasoning before answering

Design philosophy: each method tests a different lever for steering LLM opinion
distributions. Methods 1-3 vary the CONTENT of persona conditioning. Method 4
varies the FRAMING (individual vs. population). Method 5 varies the PROCESS
(direct answer vs. reasoned answer).
"""
import random
from dataclasses import dataclass
from demographics import DemographicProfile


# ---------------------------------------------------------------------------
# Method 1: Demographic Baseline
# ---------------------------------------------------------------------------
# Hypothesis: Explicit demographic attributes activate the model's learned
# correlations between demographics and opinions. Simple, transparent, and
# serves as the control condition.

def create_demographic_personas(profiles: list[DemographicProfile]) -> list[str]:
    """Minimal demographic conditioning — the null hypothesis baseline."""
    prompts = []
    for p in profiles:
        prompt = (
            "You are an American adult participating in a national opinion survey. "
            "Here is your demographic profile:\n\n"
            f"{p.to_text()}\n\n"
            "Answer each survey question as this person genuinely would. "
            "Do not hedge or give a balanced answer — commit to one position."
        )
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# Method 2: Narrative Backstory
# ---------------------------------------------------------------------------
# Hypothesis: A rich biographical narrative grounds the persona in specific
# lived experience, making it harder for the model to fall back on shallow
# stereotypes or default to its training-distribution center of mass.
# The backstory creates a coherent identity that constrains responses more
# tightly than bare demographics.
#
# Implementation: We generate backstories programmatically from templates
# rather than via a second LLM call. This keeps the experiment reproducible
# and avoids confounding the results with backstory-generation quality.

# Occupation pools conditional on education level
_OCCUPATIONS = {
    "High school diploma or less": [
        "construction worker", "retail cashier", "warehouse worker",
        "truck driver", "home health aide", "landscaper", "cook",
        "janitor", "factory line worker", "auto mechanic",
        "electrician's apprentice", "security guard", "farmworker",
    ],
    "Some college or associate's degree": [
        "dental hygienist", "real estate agent", "police officer",
        "firefighter", "small business owner", "office administrator",
        "IT support technician", "paralegal", "insurance agent",
        "sales representative", "restaurant manager",
    ],
    "Bachelor's degree": [
        "software developer", "marketing manager", "high school teacher",
        "accountant", "registered nurse", "financial analyst",
        "human resources manager", "journalist", "civil engineer",
        "graphic designer", "sales director", "project manager",
    ],
    "Graduate or professional degree": [
        "attorney", "physician", "college professor", "psychologist",
        "data scientist", "architect", "pharmacist", "school principal",
        "management consultant", "research scientist", "social worker",
    ],
}

# Life experiences that add texture
_EXPERIENCES = {
    "18-29": [
        "recently graduated and carrying student debt",
        "working their first full-time job",
        "living with roommates to afford rent",
        "active on social media and engaged with online communities",
        "considering whether to go back to school",
    ],
    "30-44": [
        "raising two young children",
        "recently bought their first home with a 30-year mortgage",
        "balancing career ambitions with family responsibilities",
        "switched careers after their industry went through layoffs",
        "coaching their kid's little league team on weekends",
    ],
    "45-59": [
        "helping their aging parents while supporting teenage kids",
        "been at the same company for over 15 years",
        "recently went through a divorce",
        "worried about having enough saved for retirement",
        "active in their local church or community organization",
    ],
    "60+": [
        "recently retired after 35 years in the workforce",
        "living on Social Security and a modest pension",
        "spending more time with grandchildren",
        "dealing with rising healthcare costs",
        "volunteering at a local food bank twice a week",
    ],
}

_COMMUNITY_DETAILS = {
    "Urban": [
        "in a mid-rise apartment", "in a rowhouse neighborhood",
        "near downtown", "in a gentrifying neighborhood",
    ],
    "Suburban": [
        "in a single-family home in a subdivision",
        "in a cul-de-sac neighborhood with good schools",
        "in a planned community near a highway",
        "in an older suburban neighborhood",
    ],
    "Rural": [
        "on a few acres outside a small town",
        "in a small town with one main street",
        "in a farming community", "in an unincorporated area",
    ],
}

_FIRST_NAMES_M = [
    "James", "Robert", "Michael", "William", "David", "Carlos", "Jose",
    "Marcus", "Tyler", "Wei", "Jamal", "Brandon", "Kevin", "Brian",
    "Anthony", "Daniel", "Matthew", "Andrew", "Ryan", "Justin",
]
_FIRST_NAMES_F = [
    "Mary", "Patricia", "Jennifer", "Linda", "Maria", "Ashley", "Keisha",
    "Sarah", "Jessica", "Mei", "Lisa", "Karen", "Nancy", "Betty",
    "Sandra", "Margaret", "Dorothy", "Emily", "Olivia", "Sophia",
]


def create_narrative_personas(
    profiles: list[DemographicProfile], seed: int = 42
) -> list[str]:
    """Rich biographical backstory personas."""
    rng = random.Random(seed)
    prompts = []

    for p in profiles:
        # Pick name
        if p.gender == "Male":
            name = rng.choice(_FIRST_NAMES_M)
        else:
            name = rng.choice(_FIRST_NAMES_F)

        occupation = rng.choice(_OCCUPATIONS[p.education])
        experience = rng.choice(_EXPERIENCES[p.age_bracket])
        housing = rng.choice(_COMMUNITY_DETAILS[p.community])

        backstory = (
            f"You are {name}, a {p.age}-year-old {p.race} {p.gender.lower()} "
            f"who works as a {occupation}. You live {housing} in the {p.region} "
            f"United States. Your household income is {p.income_bracket} per year. "
            f"You are {experience}. "
            f"Politically, you identify as {p.party}."
        )

        prompt = (
            f"{backstory}\n\n"
            "You are participating in a national opinion survey. Answer each "
            "question as this person genuinely would, based on your life "
            "situation, values, and experiences. Commit to one clear position."
        )
        prompts.append(prompt)

    return prompts


# ---------------------------------------------------------------------------
# Method 3: Value-Belief Anchoring
# ---------------------------------------------------------------------------
# Hypothesis: Opinions are more directly predicted by values and moral
# foundations than by demographics. By anchoring personas on VALUES rather
# than demographics, we may better capture the actual causal drivers of
# opinion variation.
#
# Theoretical basis: Jonathan Haidt's Moral Foundations Theory identifies
# five (later six) moral foundations. Research shows liberals weight
# Care and Fairness heavily while conservatives weight all foundations
# more evenly. We sample value profiles that reflect this structure.

@dataclass
class ValueProfile:
    """Scores from 1 (not important) to 5 (extremely important) on each foundation."""
    care: int        # Care/Harm — sensitivity to suffering
    fairness: int    # Fairness/Cheating — justice, rights, equality
    loyalty: int     # Loyalty/Betrayal — patriotism, in-group solidarity
    authority: int   # Authority/Subversion — respect for tradition, hierarchy
    sanctity: int    # Sanctity/Degradation — purity, disgust, sacredness
    liberty: int     # Liberty/Oppression — autonomy, freedom from control

    def to_text(self) -> str:
        labels = {
            1: "not important to you",
            2: "somewhat unimportant to you",
            3: "moderately important to you",
            4: "very important to you",
            5: "one of your deepest convictions",
        }
        return (
            f"- Caring for others and preventing harm: {labels[self.care]}\n"
            f"- Fairness, justice, and equal rights: {labels[self.fairness]}\n"
            f"- Loyalty to your community and country: {labels[self.loyalty]}\n"
            f"- Respect for authority, tradition, and social order: {labels[self.authority]}\n"
            f"- Moral purity and upholding sacred values: {labels[self.sanctity]}\n"
            f"- Personal freedom and liberty from government control: {labels[self.liberty]}"
        )


def _sample_value_profile(party: str, rng: random.Random) -> ValueProfile:
    """
    Sample a value profile conditioned on political leaning.

    Based on Moral Foundations research:
    - Liberals: high Care + Fairness, lower Loyalty/Authority/Sanctity
    - Conservatives: moderate-to-high on ALL foundations
    - Independents: mixed
    """
    def _clamp(x):
        return max(1, min(5, x))

    if "Democrat" in party:
        return ValueProfile(
            care=_clamp(rng.gauss(4.2, 0.7)),
            fairness=_clamp(rng.gauss(4.3, 0.7)),
            loyalty=_clamp(rng.gauss(2.5, 1.0)),
            authority=_clamp(rng.gauss(2.3, 1.0)),
            sanctity=_clamp(rng.gauss(2.0, 1.0)),
            liberty=_clamp(rng.gauss(3.5, 1.0)),
        )
    elif "Republican" in party:
        return ValueProfile(
            care=_clamp(rng.gauss(3.2, 0.8)),
            fairness=_clamp(rng.gauss(3.3, 0.8)),
            loyalty=_clamp(rng.gauss(4.0, 0.8)),
            authority=_clamp(rng.gauss(4.0, 0.8)),
            sanctity=_clamp(rng.gauss(3.8, 0.9)),
            liberty=_clamp(rng.gauss(4.3, 0.7)),
        )
    else:  # Independent
        return ValueProfile(
            care=_clamp(rng.gauss(3.5, 1.0)),
            fairness=_clamp(rng.gauss(3.5, 1.0)),
            loyalty=_clamp(rng.gauss(3.0, 1.0)),
            authority=_clamp(rng.gauss(3.0, 1.0)),
            sanctity=_clamp(rng.gauss(2.8, 1.0)),
            liberty=_clamp(rng.gauss(3.8, 1.0)),
        )


def _clamp_int(x):
    return max(1, min(5, round(x)))


def create_value_personas(
    profiles: list[DemographicProfile], seed: int = 42
) -> list[str]:
    """Personas anchored on moral foundations and values, not demographics."""
    rng = random.Random(seed)
    prompts = []

    for p in profiles:
        vp = _sample_value_profile(p.party, rng)
        # Round to ints for clean labels
        vp = ValueProfile(
            care=_clamp_int(vp.care),
            fairness=_clamp_int(vp.fairness),
            loyalty=_clamp_int(vp.loyalty),
            authority=_clamp_int(vp.authority),
            sanctity=_clamp_int(vp.sanctity),
            liberty=_clamp_int(vp.liberty),
        )

        prompt = (
            "You are an American adult participating in a national opinion survey. "
            "Your worldview is shaped by the following core values:\n\n"
            f"{vp.to_text()}\n\n"
            "These values deeply inform how you see political and social issues. "
            "Answer each survey question based on what someone with YOUR values "
            "would genuinely believe. Commit to one clear position."
        )
        prompts.append(prompt)

    return prompts


# ---------------------------------------------------------------------------
# Method 4: Distribution-Aware Ensemble
# ---------------------------------------------------------------------------
# Hypothesis: Instead of asking the LLM to be one individual, we inform it
# of its ROLE in reproducing a population distribution. This leverages the
# model's knowledge of aggregate opinion data directly, rather than hoping
# individual persona simulation aggregates correctly.
#
# This is a meta-approach: the model knows it's part of an ensemble and is
# asked to contribute to distributional fidelity. It's philosophically
# different from Methods 1-3, which pretend the model IS an individual.

def create_distribution_aware_personas(
    profiles: list[DemographicProfile],
) -> list[str]:
    """Personas that know they're part of a representative ensemble."""
    n = len(profiles)
    prompts = []

    for i, p in enumerate(profiles):
        prompt = (
            f"You are helping simulate a nationally representative opinion survey "
            f"of {n} US adults. You are respondent #{i+1} of {n}.\n\n"
            f"Your demographic profile:\n{p.to_text()}\n\n"
            "IMPORTANT CONTEXT:\n"
            f"- The goal is for the combined responses of all {n} respondents to "
            "approximate the genuine distribution of opinion among US adults.\n"
            "- You must commit to ONE position — the position that a real person "
            "with your profile would authentically hold.\n"
            "- Do NOT default to the most popular or most 'acceptable' answer. "
            "Real Americans hold genuinely diverse views.\n"
            "- Minority opinions are valid and must be represented in the sample.\n\n"
            "Answer each survey question as your assigned respondent."
        )
        prompts.append(prompt)

    return prompts


# ---------------------------------------------------------------------------
# Method 5: Cognitive Deliberation (Chain-of-Thought)
# ---------------------------------------------------------------------------
# Hypothesis: Adding a reasoning step before answering activates deeper
# simulation of the persona's thought process. Instead of pattern-matching
# demographics to a stereotypical response, the model is forced to construct
# a rationale, which may produce more authentic and varied responses.
#
# Risk: CoT might actually REDUCE diversity by giving the model more room
# to rationalize toward its default training distribution. This is an
# empirical question this experiment helps answer.

def create_cognitive_personas(profiles: list[DemographicProfile]) -> list[str]:
    """Personas that deliberate before answering."""
    prompts = []

    for p in profiles:
        prompt = (
            "You are an American adult participating in a national opinion survey. "
            "Here is your demographic profile:\n\n"
            f"{p.to_text()}\n\n"
            "For each survey question, FIRST think through your perspective:\n"
            "- What life experiences, given your background, might shape your view?\n"
            "- What do people in your community and social circle tend to think?\n"
            "- What values matter most to you on this issue?\n"
            "- What would you say if someone asked you this at the dinner table?\n\n"
            "Think through your reasoning briefly, then give your final answer. "
            "Commit to one clear position — do not hedge."
        )
        prompts.append(prompt)

    return prompts


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHODS = {
    "demographic_baseline": {
        "fn": create_demographic_personas,
        "description": "Simple demographic attribute conditioning",
        "needs_profiles": True,
    },
    "narrative_backstory": {
        "fn": create_narrative_personas,
        "description": "Rich biographical backstory grounding",
        "needs_profiles": True,
    },
    "value_anchored": {
        "fn": create_value_personas,
        "description": "Moral foundations & value-based anchoring",
        "needs_profiles": True,
    },
    "distribution_aware": {
        "fn": create_distribution_aware_personas,
        "description": "Meta-prompt: model knows its ensemble role",
        "needs_profiles": True,
    },
    "cognitive_deliberation": {
        "fn": create_cognitive_personas,
        "description": "Chain-of-thought reasoning before answering",
        "needs_profiles": True,
    },
}
