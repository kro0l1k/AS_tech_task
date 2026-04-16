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
    # SurveyQuestion(
    #     id="gun_laws",
    #     text=(
    #         "In general, do you think gun laws in this country should be "
    #         "made more strict, made less strict, or kept as they are now?"
    #     ),
    #     options=["More strict", "Less strict", "Kept as they are now"],
    #     ground_truth={
    #         "More strict": 0.56,
    #         "Less strict": 0.10,
    #         "Kept as they are now": 0.34,
    #     },
    #     source="Gallup, October 2024",
    # ),

    # SurveyQuestion(
    #     id="climate_cause",
    #     text=(
    #         "From what you have heard or read, which of the following best "
    #         "describes your view of climate change?"
    #     ),
    #     options=[
    #         "Caused mostly by human activities",
    #         "Caused mostly by natural changes in the environment",
    #         "There is no solid evidence of climate change",
    #         "Not sure",
    #     ],
    #     ground_truth={
    #         "Caused mostly by human activities": 0.57,
    #         "Caused mostly by natural changes in the environment": 0.26,
    #         "There is no solid evidence of climate change": 0.14,
    #         "Not sure": 0.03,
    #     },
    #     source="Pew Research Center, 2024",
    # ),

    # SurveyQuestion(
    #     id="marijuana",
    #     text="Do you think the use of marijuana should be…",
    #     options=[
    #         "Legal for recreational and medical use",
    #         "Legal for medical use only",
    #         "Not legal for any use",
    #     ],
    #     ground_truth={
    #         "Legal for recreational and medical use": 0.64,
    #         "Legal for medical use only": 0.26,
    #         "Not legal for any use": 0.10,
    #     },
    #     source="Gallup, November 2024",
    # ),

    # SurveyQuestion(
    #     id="immigration_level",
    #     text=(
    #         "Should immigration to this country be kept at its present level, "
    #         "increased, or decreased?"
    #     ),
    #     options=["Increased", "Decreased", "Kept at present level", "Not sure"],
    #     ground_truth={
    #         "Increased": 0.21,
    #         "Decreased": 0.41,
    #         "Kept at present level": 0.33,
    #         "Not sure": 0.05,
    #     },
    #     source="Gallup, February 2024",
    # ),

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
        id="abortion_legal_all_or_most",
        text=(
            "When it comes to the legality of abortion, "
            "do you think abortion should be legal in all cases, "
            "legal in most cases, or illegal in all or most cases?"
        ),
        options=[
            "Legal in all cases",
            "Legal in most cases",
            "Illegal in all or most cases",
        ],
        ground_truth={
            "Legal in all cases": 0.28,
            "Legal in most cases": 0.33,
            "Illegal in all or most cases": 0.39,
        },
        source="The 19th/SurveyMonkey poll, 2025",
        url="https://www.surveymonkey.com/curiosity/the-19th-surveymonkey-poll-september-2025/",
    ),
    # SurveyQuestion(
    #     id="immigration_top_priority",
    #     text=(
    #         "Which of the following should be the top priority on immigration for the United States?"
    #     ),
    #     options=[
    #         "Securing the U.S.-Mexico border",
    #         "Offering a path to citizenship for undocumented immigrants now living in the U.S.",
    #         "Deporting those who are in the country illegally",
    #     ],
    #     ground_truth={
    #         "Securing the U.S.-Mexico border": 0.33,
    #         "Offering a path to citizenship for undocumented immigrants now living in the U.S.": 0.20,
    #         "Deporting those who are in the country illegally": 0.18,
    #     },
    #     source="Scripps News/Ipsos, 2024",
    #     url="https://www.ipsos.com/en-us/securing-border-seen-top-immigration-priority",
    # ),
    # SurveyQuestion(
    #     id="border_policy_biden",
    #     text=(
    #         "Do you think the Biden administration should keep its border policies the same "
    #         "or make it tougher to get in the U.S. illegally?"
    #     ),
    #     options=[
    #         "Keep border policies the same",
    #         "Make it tougher to get in the U.S. illegally",
    #     ],
    #     ground_truth={
    #         "Keep border policies the same": 0.27,
    #         "Make it tougher to get in the U.S. illegally": 0.73,
    #     },
    #     source="Harris X / The Harris Poll, 2024",
    #     url="https://cis.org/Arthur/Polls-Show-2024-Shaping-Be-Immigration-Election",
    # ),
#     SurveyQuestion(
#     id="_path_vs_deport",
#     text=(
#         "Which of the following comes closer to your view about most undocumented immigrants "
#         "in the United States: should they be given a pathway to legal status or "
#         "deported from the U.S.?"
#     ),
#     options=[
#         "Given a pathway to legal status",
#         "Deported from the U.S.",
#         "Not sure",
#     ],
#     ground_truth={
#         "Given a pathway to legal status": 0.55,
#         "Deported from the U.S.": 0.36,
#         "Not sure": 0.09,
#     },
#     source="Quinnipiac University Poll, December 2024",
#     url="https://poll.qu.edu/poll-release?releaseid=3926",
# ),
#     SurveyQuestion(
#     id="ai_job_automation",
#     text=(
#         "Compared to today, do you think the use of artificial intelligence will lead "
#         "to more jobs, fewer jobs, or about the same number of jobs in the U.S.?"
#     ),
#     options=[
#         "More jobs",
#         "Fewer jobs",
#         "About the same number of jobs",
#     ],
#     ground_truth={
#         "More jobs": 0.21,
#         "Fewer jobs": 0.51,
#         "About the same number of jobs": 0.28,
#     },
#     source="CBS News/YouGov, March 2026",
#     url="https://www.cbsnews.com/news/what-ai-artificial-intelligence-should-do-poll-analysis/",
# ),
#     SurveyQuestion(
#     id="ai_work_usage",
#     text=(
#         "At your current or most recent job, how much of your work is done with AI tools?"
#     ),
#     options=[
#         "All or most of my work",
#         "Some of my work",
#         "None of my work",
#     ],
#     ground_truth={
#         "All or most of my work": 0.02,
#         "Some of my work": 0.19,
#         "None of my work": 0.79,
#     },
#     source="Pew Research Center, September 2025",
#     url="https://www.pewresearch.org/short-reads/2025/10/06/about-1-in-5-us-workers-now-use-ai-in-their-job-up-since-last-year/",
# ),
    
]
