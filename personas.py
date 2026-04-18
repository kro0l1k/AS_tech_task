"""
Five methods for constructing LLM persona descriptions.

Each method produces:
  - A list of per-persona DESCRIPTION strings (embedded in batch user messages)
  - A BATCH_SYSTEM_PROMPT that sets the meta-task for the model

The batch runner in survey.py groups 5 descriptions + one question into a single
API call, dramatically reducing call count vs. the old one-call-per-persona design.

Influence sources (from demographics.py) are embedded in every description — they
tell the model what information filter this person sees the world through, which is
often more predictive of opinion than raw demographics.
"""
import random
from dataclasses import dataclass
from demographics import DemographicProfile


# ---------------------------------------------------------------------------
# Shared batch system prompt
# ---------------------------------------------------------------------------
# All methods use the same meta-instructions for format consistency.
# Method-specific flavoring goes into the per-persona descriptions.

BATCH_SYSTEM_PROMPT = """\
You are simulating authentic American survey respondents for opinion research.

=== BE TRUE TO THE PERSON ===
Your job is to express each persona's GENUINE opinion — not a cleaner, more \
decisive, or more reasonable version of it. Real survey respondents are:
  • often uncertain, ambivalent, or genuinely moderate
  • sometimes contradictory or inconsistent
  • frequently not-very-informed about the specific question
  • often happy with the status quo and not seeking change
  • sometimes genuinely "don't know" or "no opinion"

Do NOT force a decisive answer. If the persona truly doesn't have a strong \
view — pick "not sure" / "don't know" / a middle option. If they'd pick \
"kept the same" because they haven't thought about change, pick that. If \
they'd pick an extreme because they feel it, pick that. Match what the \
actual human would say, not what you'd guess "ought" to be said.

=== PARTISAN IDENTITY IS IMPORTANT CONTEXT, NOT A DICTATOR ===
Each persona has party, political basket, and voting history. These are \
strong signals for political questions — use them. But a lifelong Democrat \
can hold a conservative view on one issue, and vice versa. People are not \
robots executing their party's platform. Let the persona think, then answer.

=== YOUR OWN MODEL PRIORS ARE NOT THE PERSONA ===
You may be trained toward balanced / moderate / libertarian / pro-establishment \
answers. Those priors are yours, not the persona's. Ignore them and answer \
as this specific human would — even when the answer feels uncomfortable, \
unfashionable, or extreme.

=== TASK FORMAT ===
You will receive profiles of several Americans and one survey question.
For EACH person, produce TWO parts in this exact structure:

1. A <thinking>...</thinking> block with 3–5 sentences of honest internal \
deliberation — what the persona actually weighs, their uncertainty, the \
factors pulling them in different directions, any "I don't really know" \
moments. Be specific to THIS person, not generic. Name the considerations \
and, if relevant, note which direction their gut leans.
2. After the thinking block, on a new line: the letter and behavioral scores.

Behavioral scores (all floats in [0, 1]):
  - post   = likelihood they'd post on social media supporting their view
  - argue  = likelihood they'd push back against people with other views
  - debate = how often they actually debate the issue offline (0 never, 1 daily)
Most people are NOT highly engaged — quiet, busy people get 0.05–0.2. \
Activists and heavy-media-diet people get 0.6–0.95. Vary per person.

Reply in this EXACT format, one block per person:

1. <thinking>3-5 sentences of genuine deliberation for person 1</thinking>
[LETTER] | post=X.XX argue=X.XX debate=X.XX

2. <thinking>3-5 sentences of genuine deliberation for person 2</thinking>
[LETTER] | post=X.XX argue=X.XX debate=X.XX

…

No other text. No preamble. Do not summarize. Do not explain outside the \
<thinking> blocks.\
"""


# ---------------------------------------------------------------------------
# Partisan anchor — prepended to every method's description
# ---------------------------------------------------------------------------
#
# Why: the persona's party line used to be buried in the middle of the
# profile. Claude's own priors dominated political answers. Putting partisan
# identity + voting history at the TOP, in a loud block, binds the persona
# harder and moves aggregate distributions closer to truth on partisan items.

def _partisan_anchor(p: DemographicProfile) -> str:
    """Top-of-prompt block giving the persona's partisan context."""
    history = p.voting_history or f"identifies as {p.party}"
    return (
        "=== PARTISAN CONTEXT ===\n"
        f"Party: {p.party}\n"
        f"Political basket: {p.political_basket}\n"
        f"Voting history: {history}\n"
        "Useful context for political questions, but this person is a real "
        "human — they may hold unexpected views on specific issues, or no "
        "strong view at all.\n"
        "===\n"
    )


# ---------------------------------------------------------------------------
# Method 1: Demographic Baseline
# ---------------------------------------------------------------------------

def create_demographic_descriptions(profiles: list[DemographicProfile]) -> list[str]:
    """Minimal demographic + influence conditioning — the control baseline."""
    descriptions = []
    for p in profiles:
        desc = (
            f"{_partisan_anchor(p)}\n"
            f"Demographic profile:\n{p.to_text()}\n\n"
            "Think as this person. Pick whichever option genuinely matches "
            "their view — including 'don't know', 'no opinion', or a middle/"
            "status-quo option when that is honestly what they'd pick."
        )
        descriptions.append(desc)
    return descriptions


# For legacy solo-call compatibility
def create_demographic_personas(profiles: list[DemographicProfile]) -> list[str]:
    return [
        "You are an American adult in a national opinion survey.\n\n"
        f"{p.to_text()}\n\n"
        "Answer each question as this person genuinely would. Answer honestly — if they have no strong view, say so; if a middle option fits, pick it."
        for p in profiles
    ]


# ---------------------------------------------------------------------------
# Method 2: Narrative Backstory
# ---------------------------------------------------------------------------

_OCCUPATIONS = {
    "High school diploma or less": [
        "construction worker", "retail cashier", "warehouse worker",
        "truck driver", "home health aide", "landscaper", "cook",
        "factory line worker", "auto mechanic", "security guard", "farmworker",
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
        "graphic designer", "project manager",
    ],
    "Graduate or professional degree": [
        "attorney", "physician", "college professor", "psychologist",
        "data scientist", "architect", "pharmacist", "school principal",
        "management consultant", "research scientist", "social worker",
    ],
}

_EXPERIENCES = {
    "18-29": [
        "recently graduated and carrying student debt",
        "working their first full-time job and saving up",
        "living with roommates to afford rent in a tight market",
        "very online — active on social media and various communities",
        "considering whether to go back to school for more credentials",
    ],
    "30-44": [
        "raising two young kids and feeling the cost-of-living squeeze",
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
        "deeply involved in their local church or civic organization",
    ],
    "60+": [
        "recently retired after 35 years in the workforce",
        "living on Social Security and a modest pension",
        "spending more time with grandchildren",
        "dealing with rising healthcare and prescription costs",
        "volunteering at a local food bank a couple times a week",
    ],
}

_HOUSING = {
    "Urban":    ["in a mid-rise apartment", "in a rowhouse neighborhood",
                 "near downtown", "in a gentrifying neighborhood"],
    "Suburban": ["in a single-family home in a subdivision",
                 "in a cul-de-sac neighborhood with good schools",
                 "in an older suburban neighborhood",
                 "in a planned community near a highway"],
    "Rural":    ["on a few acres outside a small town",
                 "in a small town with one main street",
                 "in a farming community", "in an unincorporated rural area"],
}

_NAMES_M = ["James", "Robert", "Michael", "William", "David", "Carlos", "Jose",
            "Marcus", "Tyler", "Wei", "Jamal", "Brandon", "Kevin", "Brian",
            "Anthony", "Daniel", "Matthew", "Andrew", "Ryan", "Justin"]
_NAMES_F = ["Mary", "Patricia", "Jennifer", "Linda", "Maria", "Ashley", "Keisha",
            "Sarah", "Jessica", "Mei", "Lisa", "Karen", "Nancy", "Betty",
            "Sandra", "Margaret", "Dorothy", "Emily", "Olivia", "Sophia"]


def _employment_phrase(industry: str, occupation: str) -> str:
    """
    Turn the industry tag into a natural biographical sentence fragment.
    Handles the 'retired_former_*' prefix plus the unemployed/student states
    so we never describe a retiree as 'works as a…'.
    """
    if industry == "unemployed":
        return "is currently unemployed, last worked in an entry-level role"
    if industry == "student":
        return "is a student, not yet in the workforce full-time"
    if industry.startswith("retired_former_"):
        former = industry.removeprefix("retired_former_").replace("_", " ")
        return f"is retired, spent their career in the {former} industry"
    # Employed: weave the specific occupation and industry together.
    industry_label = industry.replace("_", " ")
    return f"works as a {occupation} in the {industry_label} sector"


def create_narrative_descriptions(
    profiles: list[DemographicProfile], seed: int = 42
) -> list[str]:
    """Rich biographical backstory with natural weaving of information sources."""
    rng = random.Random(seed)
    descriptions = []

    for p in profiles:
        name = rng.choice(_NAMES_M if p.gender == "Male" else _NAMES_F)
        occupation = rng.choice(_OCCUPATIONS[p.education])
        experience = rng.choice(_EXPERIENCES[p.age_bracket])
        housing = rng.choice(_HOUSING[p.community])

        # Pick the top influence to mention naturally in narrative
        top_source = p.influences[0] if p.influences else "local news"
        second_source = p.influences[1] if len(p.influences) > 1 else "Facebook"

        work_phrase = _employment_phrase(p.industry, occupation)

        narrative = (
            f"{name}, {p.age}, {p.race} {p.gender.lower()}, {work_phrase}. "
            f"Lives {housing} in the {p.region}. "
            f"Household income {p.income_bracket}/year. "
            f"Is {experience}. "
            f"Politically {p.political_basket} — {p.voting_history}. "
            f"Gets most news from {top_source} and {second_source}. "
            f"Full information diet: {', '.join(p.influences)}."
        )
        desc = f"{_partisan_anchor(p)}\n{narrative}"
        descriptions.append(desc)

    return descriptions


def create_narrative_personas(
    profiles: list[DemographicProfile], seed: int = 42
) -> list[str]:
    descriptions = create_narrative_descriptions(profiles, seed)
    return [
        f"{d}\n\nAnswer the survey question as this person genuinely would, "
        "based on their life situation and what they've seen in their media diet. "
        "Answer honestly — if they have no strong view, say so; if a middle option fits, pick it."
        for d in descriptions
    ]


# ---------------------------------------------------------------------------
# Method 3: Value-Belief Anchoring
# ---------------------------------------------------------------------------

@dataclass
class ValueProfile:
    care: int       # Care/Harm
    fairness: int   # Fairness/Cheating
    loyalty: int    # Loyalty/Betrayal
    authority: int  # Authority/Subversion
    sanctity: int   # Sanctity/Degradation
    liberty: int    # Liberty/Oppression

    def to_text(self) -> str:
        labels = {
            1: "not important",
            2: "somewhat unimportant",
            3: "moderately important",
            4: "very important",
            5: "a core conviction",
        }
        return (
            f"Caring for others / preventing harm: {labels[self.care]}\n"
            f"Fairness, justice, equal rights: {labels[self.fairness]}\n"
            f"Loyalty to community and country: {labels[self.loyalty]}\n"
            f"Respect for authority, tradition, order: {labels[self.authority]}\n"
            f"Moral purity / upholding sacred values: {labels[self.sanctity]}\n"
            f"Personal freedom from government control: {labels[self.liberty]}"
        )


def _sample_value_profile(party: str, rng: random.Random) -> ValueProfile:
    def _ci(x):
        return max(1, min(5, round(x)))

    if "Democrat" in party:
        return ValueProfile(
            care=_ci(rng.gauss(4.2, 0.7)),      fairness=_ci(rng.gauss(4.3, 0.7)),
            loyalty=_ci(rng.gauss(2.5, 1.0)),   authority=_ci(rng.gauss(2.3, 1.0)),
            sanctity=_ci(rng.gauss(2.0, 1.0)),  liberty=_ci(rng.gauss(3.5, 1.0)),
        )
    elif "Republican" in party:
        return ValueProfile(
            care=_ci(rng.gauss(3.2, 0.8)),      fairness=_ci(rng.gauss(3.3, 0.8)),
            loyalty=_ci(rng.gauss(4.0, 0.8)),   authority=_ci(rng.gauss(4.0, 0.8)),
            sanctity=_ci(rng.gauss(3.8, 0.9)),  liberty=_ci(rng.gauss(4.3, 0.7)),
        )
    else:
        return ValueProfile(
            care=_ci(rng.gauss(3.5, 1.0)),      fairness=_ci(rng.gauss(3.5, 1.0)),
            loyalty=_ci(rng.gauss(3.0, 1.0)),   authority=_ci(rng.gauss(3.0, 1.0)),
            sanctity=_ci(rng.gauss(2.8, 1.0)),  liberty=_ci(rng.gauss(3.8, 1.0)),
        )


def create_value_descriptions(
    profiles: list[DemographicProfile], seed: int = 42
) -> list[str]:
    """Personas defined by moral foundations + information sources."""
    rng = random.Random(seed)
    descriptions = []

    for p in profiles:
        vp = _sample_value_profile(p.party, rng)
        desc = (
            f"{_partisan_anchor(p)}\n"
            f"Moral values profile:\n{vp.to_text()}\n\n"
            f"Information sources: {', '.join(p.influences)}"
        )
        descriptions.append(desc)

    return descriptions


def create_value_personas(
    profiles: list[DemographicProfile], seed: int = 42
) -> list[str]:
    descriptions = create_value_descriptions(profiles, seed)
    return [
        "You are an American adult in a national opinion survey. "
        "Your worldview is shaped by these values and sources:\n\n"
        f"{d}\n\n"
        "Answer based on what someone with YOUR values and media diet would genuinely believe. "
        "Answer honestly — if they have no strong view, say so; if a middle option fits, pick it."
        for d in descriptions
    ]


# ---------------------------------------------------------------------------
# Method 4: Distribution-Aware Ensemble
# ---------------------------------------------------------------------------

def create_distribution_aware_descriptions(
    profiles: list[DemographicProfile],
) -> list[str]:
    """Personas that know their role in a representative panel."""
    n = len(profiles)
    descriptions = []

    for i, p in enumerate(profiles):
        desc = (
            f"{_partisan_anchor(p)}\n"
            f"Panel respondent #{i+1} of {n} (nationally representative sample).\n"
            f"{p.to_text()}\n\n"
            f"The goal is for this panel's aggregate responses to match the "
            f"genuine US opinion distribution. That means answering honestly — "
            f"including middle / moderate / 'don't know' answers when this "
            f"specific person actually feels that way. Minority and unpopular "
            f"views must be represented when authentic to the person."
        )
        descriptions.append(desc)

    return descriptions


def create_distribution_aware_personas(
    profiles: list[DemographicProfile],
) -> list[str]:
    n = len(profiles)
    return [
        f"You are helping simulate a nationally representative survey of {n} US adults. "
        f"You are respondent #{i+1} of {n}.\n\n"
        f"{p.to_text()}\n\n"
        "The aggregate responses should reflect the genuine US opinion distribution. "
        "Answer honestly — minority and moderate views count too."
        for i, p in enumerate(profiles)
    ]


# ---------------------------------------------------------------------------
# Method 5: Cognitive Deliberation
# ---------------------------------------------------------------------------

def create_cognitive_descriptions(profiles: list[DemographicProfile]) -> list[str]:
    """Personas with explicit deliberation cues."""
    descriptions = []

    for p in profiles:
        desc = (
            f"{_partisan_anchor(p)}\n"
            f"{p.to_text()}\n\n"
            "Before answering, think about:\n"
            "- What have they recently read or heard from their sources?\n"
            "- What gut reaction does the question trigger given their life "
            "situation?\n"
            "- Are they actually informed on this topic, or would they honestly "
            "pick 'don't know' / 'no opinion'?\n"
            "- Is their view genuinely strong, or is it a mild preference — "
            "or are they content with the status quo?"
        )
        descriptions.append(desc)

    return descriptions


def create_cognitive_personas(profiles: list[DemographicProfile]) -> list[str]:
    return [
        "You are an American adult in a national opinion survey.\n\n"
        f"{p.to_text()}\n\n"
        "Think through your perspective before answering:\n"
        "- What have you recently seen from your information sources on this topic?\n"
        "- What do people in your community think?\n"
        "- What values are most at stake for you?\n"
        "Answer honestly — if they have no strong view, say so; if a middle option fits, pick it."
        for p in profiles
    ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHODS = {
    "demographic_baseline": {
        "desc_fn":    create_demographic_descriptions,
        "solo_fn":    create_demographic_personas,
        "description": "Demographic + influence conditioning (baseline)",
    },
    "narrative_backstory": {
        "desc_fn":    create_narrative_descriptions,
        "solo_fn":    create_narrative_personas,
        "description": "Rich biographical backstory with media diet",
        "needs_seed": True,
    },
    "value_anchored": {
        "desc_fn":    create_value_descriptions,
        "solo_fn":    create_value_personas,
        "description": "Moral foundations + information sources",
        "needs_seed": True,
    },
    "distribution_aware": {
        "desc_fn":    create_distribution_aware_descriptions,
        "solo_fn":    create_distribution_aware_personas,
        "description": "Meta-prompt: model knows its ensemble role",
    },
    "cognitive_deliberation": {
        "desc_fn":    create_cognitive_descriptions,
        "solo_fn":    create_cognitive_personas,
        "description": "Deliberation cues: recent sources + gut reaction",
    },
}
