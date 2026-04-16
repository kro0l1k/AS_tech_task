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

    def __post_init__(self):
        total = sum(self.ground_truth.values())
        assert abs(total - 1.0) < 0.02, f"Ground truth for '{self.id}' sums to {total}"
        for opt in self.options:
            assert opt in self.ground_truth, f"Option '{opt}' missing from ground truth"


QUESTIONS: list[SurveyQuestion] = [
    SurveyQuestion(
        id="gun_laws",
        text=(
            "In general, do you think gun laws in this country should be "
            "made more strict, made less strict, or kept as they are now?"
        ),
        options=["More strict", "Less strict", "Kept as they are now"],
        ground_truth={
            "More strict": 0.56,
            "Less strict": 0.10,
            "Kept as they are now": 0.34,
        },
        source="Gallup, October 2024",
    ),

    SurveyQuestion(
        id="climate_cause",
        text=(
            "From what you have heard or read, which of the following best "
            "describes your view of climate change?"
        ),
        options=[
            "Caused mostly by human activities",
            "Caused mostly by natural changes in the environment",
            "There is no solid evidence of climate change",
            "Not sure",
        ],
        ground_truth={
            "Caused mostly by human activities": 0.57,
            "Caused mostly by natural changes in the environment": 0.26,
            "There is no solid evidence of climate change": 0.14,
            "Not sure": 0.03,
        },
        source="Pew Research Center, 2024",
    ),

    SurveyQuestion(
        id="marijuana",
        text="Do you think the use of marijuana should be…",
        options=[
            "Legal for recreational and medical use",
            "Legal for medical use only",
            "Not legal for any use",
        ],
        ground_truth={
            "Legal for recreational and medical use": 0.64,
            "Legal for medical use only": 0.26,
            "Not legal for any use": 0.10,
        },
        source="Gallup, November 2024",
    ),

    SurveyQuestion(
        id="immigration_level",
        text=(
            "Should immigration to this country be kept at its present level, "
            "increased, or decreased?"
        ),
        options=["Increased", "Decreased", "Kept at present level", "Not sure"],
        ground_truth={
            "Increased": 0.21,
            "Decreased": 0.41,
            "Kept at present level": 0.33,
            "Not sure": 0.05,
        },
        source="Gallup, February 2024",
    ),

    # SurveyQuestion(
    #     id="healthcare_govt",
    #     text=(
    #         "Do you think it is the responsibility of the federal government "
    #         "to make sure all Americans have healthcare coverage, or is that "
    #         "not the responsibility of the federal government?"
    #     ),
    #     options=[
    #         "Yes, government responsibility",
    #         "No, not government responsibility",
    #     ],
    #     ground_truth={
    #         "Yes, government responsibility": 0.57,
    #         "No, not government responsibility": 0.43,
    #     },
    #     source="Gallup, November 2024",
    # ),

    # SurveyQuestion(
    #     id="abortion",
    #     text="Do you think abortion should be…",
    #     options=[
    #         "Legal in all cases",
    #         "Legal in most cases",
    #         "Illegal in most cases",
    #         "Illegal in all cases",
    #     ],
    #     ground_truth={
    #         "Legal in all cases": 0.27,
    #         "Legal in most cases": 0.33,
    #         "Illegal in most cases": 0.25,
    #         "Illegal in all cases": 0.15,
    #     },
    #     source="Pew Research Center, May 2024",
    # ),

    # SurveyQuestion(
    #     id="death_penalty",
    #     text=(
    #         "Are you in favor of the death penalty for a person convicted "
    #         "of murder?"
    #     ),
    #     options=["Favor", "Oppose"],
    #     ground_truth={
    #         "Favor": 0.53,
    #         "Oppose": 0.47,
    #     },
    #     source="Gallup, October 2024",
    # ),
]
