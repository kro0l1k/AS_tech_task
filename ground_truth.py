"""
Survey questions with ground-truth response distributions from major US polls.

Each question uses real polling data so we can measure how well our LLM personas
reproduce actual public opinion. Sources are cited per question.

Design choices:
- Single-select only (forced choice) for clean distribution comparison
- Mix of 2-option, 3-option, and 4-option questions to test varying granularity
- Topics span social, economic, and policy dimensions
- Ground truth is approximate — polls have MOE of ~2-4pp — but good enough for
  rank-ordering persona methods
"""
from dataclasses import dataclass, field


@dataclass
class SurveyQuestion:
    id: str
    text: str
    options: list[str]
    ground_truth: dict[str, float]  # option text -> proportion (sums to 1.0)
    source: str
    url: str | None = None

    def __post_init__(self):
        total = sum(self.ground_truth.values())
        assert abs(total - 1.0) < 0.001, f"Ground truth for '{self.id}' sums to {total}"
        for opt in self.options:
            assert opt in self.ground_truth, f"Option '{opt}' missing from ground truth"


QUESTIONS: list[SurveyQuestion] = [

    SurveyQuestion(
        id="gun_sales_laws",
        text=(
            "Which of the following comes closest to your view about laws covering the sale of firearms in general?"
        ),
        options=[
            "More strict",
            "Kept as they are now",
            "Less strict",
        ],
        ground_truth={
            "More strict": 0.56,
            "Kept as they are now": 0.34,
            "Less strict": 0.10,
        },
        source="Gallup Crime poll, October 1–12, 2024",
        url="https://news.gallup.com/poll/653489/majorities-back-stricter-gun-laws-assault-weapons-ban.aspx",
    ),
    SurveyQuestion(
        id="climate_human_contribution",
        text=(
            "From what you have heard or read, how much do you think human activity, "
            "such as the burning of fossil fuels, contributes to climate change?"
        ),
        options=[
            "Contributes a great deal",
            "Contributes some",
            "Not too much or not at all",
        ],
        ground_truth={
            "Contributes a great deal": 0.45,
            "Contributes some": 0.29,
            "Not too much or not at all": 0.26,
        },
        source="Pew Research Center, October 2024",
        url="https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue/",
    ),
    SurveyQuestion(
        id="climate_local_impact",
        text=(
            "Have you heard or read about the effects of climate change? "
            "Do you think climate change is affecting your local community…"
        ),
        options=[
            "A great deal",
            "Some",
            "Not too much or not at all",
        ],
        ground_truth={
            "A great deal": 0.26,
            "Some": 0.38,
            "Not too much or not at all": 0.36,
        },
        source="Pew Research Center, October 2024",
        url="https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue/",
    ),
    SurveyQuestion(
        id="abortion_legal_circumstances",
        text=(
            "Thinking about the legality of abortion, "
            "do you think abortion should be legal under any circumstances, "
            "legal under certain circumstances, or illegal in all circumstances?"
        ),
        options=[
            "Legal under any circumstances",
            "Legal under certain circumstances",
            "Illegal in all circumstances",
        ],
        ground_truth={
            "Legal under any circumstances": 0.30,
            "Legal under certain circumstances": 0.55,
            "Illegal in all circumstances": 0.15,
        },
        source="Gallup, May 2025",
        url="https://news.gallup.com/poll/321143/americans-stand-abortion.aspx",
    ),

    SurveyQuestion(
    id="path_vs_deport",
    text=(
        "Which of the following comes closer to your view about most undocumented immigrants "
        "in the United States: should they be given a pathway to legal status or "
        "deported from the U.S.?"
    ),
    options=[
        "Given a pathway to legal status",
        "Deported from the U.S.",
        "Not sure",
    ],
    ground_truth={
        "Given a pathway to legal status": 0.55,
        "Deported from the U.S.": 0.36,
        "Not sure": 0.09,
    },
    source="Quinnipiac University Poll, December 2024",
    url="https://poll.qu.edu/poll-release?releaseid=3926",
),
SurveyQuestion(
    id="ai_job_automation",
    text=(
        "Do you think that AI (Artificial Intelligence) will increase, decrease, or not have any impact on the number of jobs available in the U.S.?"
    ),
    options=[
        "Increase",
        "Decrease",
        "Have no effect",
    ],
    ground_truth={
        "Increase": 0.19,
        "Decrease": 0.66,
        "Have no effect": 0.15,
    },
    source="CBS News/YouGov, March 16-19, 2026 (Q1 2026). Survey of 2,500 U.S. adults, margin of error ±2.2 points.",
    url="https://www.cbsnews.com/news/what-ai-artificial-intelligence-should-do-poll-analysis/",
),
SurveyQuestion(
    id="ai_work_usage",
    text=(
        "Thinking of the tasks you do in your job, how much of your work is done with AI?"
    ),
    options=[
        "All or most of my work",
        "Some of my work",
        "Not much or none of my work",
        "None of my work is done with AI / I don't know if any of my work is done with AI",
        "Not sure",
    ],
    ground_truth={
        "All or most of my work": 0.02,
        "Some of my work": 0.19,
        "Not much or none of my work": 0.27,  # 27% not much + 38% none
        "None of my work is done with AI / I don't know if any of my work is done with AI": 0.38,
        "Not sure": 0.14,
    },
    source="Pew Research Center, Sept 2-8, 2025 (Q3 2025). Survey of 5,010 employed U.S. adults; note 12% had not heard of workplace AI use and are excluded here.",
    url="https://www.pewresearch.org/short-reads/2025/10/06/about-1-in-5-us-workers-now-use-ai-in-their-job-up-since-last-year/",
),
    ## NEW QUESTIONS, more up to date:

## Healthcare Concern
SurveyQuestion(
    id="healthcare_worry",
    text=(
        "How much do you worry about healthcare availability and affordability, a great deal or fair amount?"
    ),
    options=[
        "A great deal",
        "Fair amount",
        "Only a little",
        "Not at all",
    ],
    ground_truth={
        "A great deal": 0.61,
        "Fair amount": 0.20,  
        "Only a little": 0.12,
        "Not at all": 0.07,
    },
    source="Gallup, March 2026 (Q1 2026)",
    url="https://www.livenowfox.com/news/healthcare-concern-gallup-poll.amp",
) ,


## Democracy Satisfaction
SurveyQuestion(
    id="democracy_working",
    text=(
        "Are you satisfied or dissatisfied with the way democracy is working in the US?"
    ),
    options=[
        "Satisfied",
        "Dissatisfied",
    ],
    ground_truth={
        "Satisfied": 0.31,
        "Dissatisfied": 0.69,
    },
    source="Pew Research Center, March 23-29, 2026 (Q1 2026). Survey of 3,507 U.S. adults.",
    url="https://www.pewresearch.org/short-reads/2026/04/15/multiple-indicators-show-a-decline-in-the-health-of-americas-democracy-in-2025/",
),
SurveyQuestion(
    id="israel_sympathy",
    text=(
        "In the Middle East situation, are your sympathies more with the Israelis or more with the Palestinians?"
    ),
    options=[
        "More with the Israelis",
        "More with the Palestinians",
        "Both equally / Neither / No opinion",
    ],
    ground_truth={
        "More with the Israelis": 0.36,
        "More with the Palestinians": 0.41,
        "Both equally / Neither / No opinion": 0.23,
    },
    source="Gallup, Feb 2-16, 2026 (Q1 2026). Random sample of 1,001 U.S. adults.",
    url="https://news.gallup.com/poll/702440/israelis-no-longer-ahead-americans-middle-east-sympathies.aspx",
),

## Drug Legalization (Marijuana)
SurveyQuestion(
    id="marijuana_legalization",
    text=(
        "Do you think the use of marijuana should be legal or not?"
    ),
    options=[
        "Should be legal",
        "Should not be legal",
    ],
    ground_truth={
        "Should be legal": 0.64,
        "Should not be legal": 0.36,
    },
    source="Gallup, October 2025 (Q4 2025). Based on telephone interviews with a random sample of 1,000 U.S. adults.",
    url="https://news.gallup.com/poll/697445/americans-positive-progress-drugs.aspx",
),

## Mandatory Childhood Vaccines for School Attendance
SurveyQuestion(
    id="vaccine_school_mandate",
    text=(
        "Do you think the government should require healthy children to be vaccinated in order to attend public school?"
    ),
    options=[
        "Yes, require vaccination",
        "No, allow unvaccinated children in schools",
    ],
    ground_truth={
        "Yes, require vaccination": 0.75,
        "No, allow unvaccinated children in schools": 0.25,
    },
    source="Reuters/Ipsos, February 2026 (Q1 2026). Online poll of 4,638 U.S. adults, margin of error ±2 percentage points.",
    url="https://www.reuters.com/business/healthcare-pharmaceuticals/americans-trust-vaccines-school-mandates-rejecting-trump-agenda-reutersipsos-poll-2026-02-25/",
),

## Taxes on the Wealthy
SurveyQuestion(
    id="raise_taxes_high_income",
    text=(
        "Do you think tax rates on households with annual incomes over $400,000 should be raised, kept the same, or lowered?"
    ),
    options=[
        "Raised",
        "Kept the same",
        "Lowered",
    ],
    ground_truth={
        "Raised": 0.62,
        "Kept the same": 0.20,
        "Lowered": 0.18,
    },
    source="Pew Research Center, January–February 2025 (Q1 2025). Survey of 5,086 U.S. adults conducted Jan. 27–Feb. 2, 2025.",
    url="https://www.pewresearch.org/short-reads/2025/03/19/most-americans-continue-to-favor-raising-taxes-on-corporations-higher-income-households/",
),

]
